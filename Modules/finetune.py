import os
# import shutil
import sys
import yaml
import numpy as np
# import pandas as pd
# from datetime import datetime
# import matplotlib.pyplot as plt
# from operator import add

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
# from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error

from dataset.dataset_test import MolTestDatasetWrapper
from utils.plots import plot_losses, parity_plot_2, parity_plot_3, plot_srcc_MAE

apex_support = False
try:
    sys.path.append('./apex')
    from apex import amp
    apex_support = True
except:
    print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False



def _save_config_file(model_checkpoints_folder, config):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        #make th config file. The file dons't already exist there
    with open(os.path.join(model_checkpoints_folder, 'config_finetune.yaml'), 'w') as file:
            yaml.dump(config, file)
        # shutil.copy('./config_finetune.yaml', os.path.join(model_checkpoints_folder, 'config_finetune.yaml'))


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


class FineTune(object):
    '''Fine-tune the model on the downstream task.
    Only use with 'visc', 'cond', 'visc_hc' tasks'''
    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()

        #current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        #dir_name = current_time + '_' + config['task_name'] + '_'
        dir_name = config['name']
        log_dir = os.path.join(config['save_folder'], dir_name)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.dataset = dataset
        if config['dataset']['task'] == 'classification':
            self.criterion = nn.CrossEntropyLoss()
        elif config['dataset']['task'] == 'regression':
            if self.config["task_name"] in ['qm7', 'qm8', 'qm9', 'visc', 'cond', 'visc_hc']:
                self.criterion = nn.L1Loss()
            else:
                self.criterion = nn.MSELoss()

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print("Running on:", device)

        return device

    def _step(self, model, data, n_iter):
        # get the prediction
        __, pred, ln_A, B = model(data)  # [N,C]
        # if self.config['dataset']['task'] == 'classification':
        loss = 1 * self.criterion(pred, data.y) + 0 * self.criterion(ln_A, data.ln_A) + 0 * self.criterion(B, data.B)
        # elif self.config['dataset']['task'] == 'regression':
        #     if self.normalizer:
        #         loss = self.criterion(pred, self.normalizer.norm(data.y))
        #     else:
        #         loss = self.criterion(pred, data.y)

        return loss, pred, ln_A, B

    def train(self, config):
        train_loader, valid_loader, test_loader = self.dataset.get_data_loaders(config['task_name'])

        self.normalizer = None
        if self.config["task_name"] in ['qm7', 'qm9']:
            labels = []
            for d, __ in train_loader:
                labels.append(d.y)
            labels = torch.cat(labels)
            self.normalizer = Normalizer(labels)
            print(self.normalizer.mean, self.normalizer.std, labels.shape)

        if self.config['model_type'] == 'gin':
            from models.ginet_finetune import GINet
            model = GINet(self.config['dataset']['task'], **self.config["model"]).to(self.device)
            model = self._load_pre_trained_weights(model)
        elif self.config['model_type'] == 'gcn':
            from models.gcn_finetune import GCN
            model = GCN(self.config['dataset']['task'], **self.config["model"]).to(self.device)
            model = self._load_pre_trained_weights(model)

        layer_list = []
        for name, param in model.named_parameters():
            if 'pred_head' in name:
                print(name, param.requires_grad)
                layer_list.append(name)

        print(layer_list)

        params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in layer_list, model.named_parameters()))))
        base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in layer_list, model.named_parameters()))))

        optimizer = torch.optim.Adam(
            [{'params': base_params, 'lr': self.config['init_base_lr']}, {'params': params}],
            self.config['init_lr'], weight_decay=eval(self.config['weight_decay'])
        )

        # scheduler = CosineAnnealingLR(
        #     optimizer, T_max=self.config['epochs']-self.config['warmup'], 
        #     eta_min=0, last_epoch=-1
        # )

        if apex_support and self.config['fp16_precision']:
            model, optimizer = amp.initialize(
                model, optimizer, opt_level='O2', keep_batchnorm_fp32=True
            )

        model_checkpoints_folder = self.writer.log_dir

        # save config file
        _save_config_file(model_checkpoints_folder, config)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf
        best_valid_rgr = np.inf
        best_valid_cls = 0

        patience = 0

        train_losses = []
        valid_losses = []

        for epoch_counter in range(self.config['epochs']):
            print(f'Epoch: {epoch_counter +1}')
            train_l = 0
            batch_counter = 0
            for bn, data in enumerate(train_loader):
                batch_counter += 1
                optimizer.zero_grad()

                data = data.to(self.device)
                loss, *_ = self._step(model, data, n_iter)
                train_l += loss.item()

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                    print(f'Train Loss: {loss:.4f}')

                if apex_support and self.config['fp16_precision']:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                n_iter += 1
            train_l /= batch_counter
            train_losses.append(train_l)

            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                if self.config['dataset']['task'] == 'classification': 
                    valid_loss, valid_cls = self._validate(model, valid_loader)
                    
                    if valid_cls > best_valid_cls:
                        # save the model weights
                        
                        best_valid_cls = valid_cls
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
                elif self.config['dataset']['task'] == 'regression': 
                    valid_loss, valid_rgr = self._validate(model, valid_loader)
                    valid_losses.append(valid_rgr)
                    if valid_loss < best_valid_loss:
                        # save the model weights
                        best_valid_rgr = valid_rgr
                        best_valid_loss = valid_loss
                        patience = 0
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
                    else:
                        patience += 1
                    print(f'Val Loss: {valid_loss:.4f} best_val_loss: {best_valid_loss:.4f} patience: {patience}/{self.config["patience"]}')

                    if patience >= self.config['patience']:
                        print("Early stopping.")
                        break

                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1

            # if epoch_counter >= self.config['warmup']:
            #     scheduler.step()

        plot_losses(train_losses, valid_losses, model_checkpoints_folder)

        predictions, labels, A_pred, ln_As, B_pred, Bs = self._test(model, test_loader)

        srcc, mae = parity_plot_3(predictions, labels, model_checkpoints_folder, tag = self.config['task_name'])

        parity_plot_2(A_pred, ln_As, model_checkpoints_folder, tag = 'ln(A)')

        parity_plot_2(B_pred, Bs, model_checkpoints_folder, tag = 'B')
        return srcc, mae


    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./ckpt', self.config['fine_tune_from'], 'checkpoints')
            # checkpoints_folder = os.path.join('./finetune', self.config['fine_tune_from'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'), map_location=self.device)
            # model.load_state_dict(state_dict)
            model.load_my_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader):
        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            num_data = 0
            for bn, data in enumerate(valid_loader):
                data = data.to(self.device)

                loss, pred, *_ = self._step(model, data, bn)

                valid_loss += loss.item()
                num_data += 1

                if self.normalizer:
                    pred = self.normalizer.denorm(pred)

                if self.config['dataset']['task'] == 'classification':
                    pred = F.softmax(pred, dim=-1)

                if self.device == 'cpu':
                    predictions.extend(pred.detach().numpy())
                    labels.extend(data.y.flatten().numpy())
                else:
                    predictions.extend(pred.cpu().flatten().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())

            valid_loss /= num_data
        
        model.train()
        
        if self.config['dataset']['task'] == 'regression':
            predictions = np.array(predictions)
            labels = np.array(labels)
            if self.config['task_name'] in ['qm7', 'qm8', 'qm9', 'visc', 'cond', 'visc_hc']:
                mae = np.mean(np.abs(labels - predictions))
                print('Validation loss:', valid_loss, 'MAE:', mae)
                return valid_loss, mae
            else:
                rmse = mean_squared_error(labels, predictions, squared=False)
                print('Validation loss:', valid_loss, 'RMSE:', rmse)
                return valid_loss, rmse

        elif self.config['dataset']['task'] == 'classification': 
            predictions = np.array(predictions)
            labels = np.array(labels)
            roc_auc = roc_auc_score(labels, predictions[:,1])
            print('Validation loss:', valid_loss, 'ROC AUC:', roc_auc)
            return valid_loss, roc_auc

    def _test(self, model, test_loader):
        model_path = os.path.join(self.writer.log_dir, 'model.pth')
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        print("Loaded trained model with success.")

        # test steps
        predictions = []
        labels = []
        ln_As = []
        A_pred = []
        Bs = []
        B_pred = []
        with torch.no_grad():
            model.eval()

            test_loss = 0.0
            num_data = 0
            for bn, data in enumerate(test_loader):
                data = data.to(self.device)

                
                loss, pred, ln_A, B = self._step(model, data, bn)

                test_loss += loss.item()
                num_data += 1

                if self.normalizer:
                    pred = self.normalizer.denorm(pred)

                if self.config['dataset']['task'] == 'classification':
                    pred = F.softmax(pred, dim=-1)

                if self.device == 'cpu':
                    predictions.extend(pred.detach().numpy())
                    labels.extend(data.y.flatten().numpy())
                else:
                    predictions.extend(pred.cpu().flatten().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())
                    data.ln_A = data.ln_A.expand(-1, 5)
                    data.B = data.B.expand(-1, 5)
                    ln_As.extend(data.ln_A.cpu().flatten().numpy())
                    A_pred.extend(ln_A.cpu().flatten().numpy())
                    Bs.extend(data.B.cpu().flatten().numpy())
                    B_pred.extend(B.cpu().flatten().numpy())

            test_loss /= num_data
        
        model.train()

        if self.config['dataset']['task'] == 'regression':
            predictions = np.array(predictions)
            labels = np.array(labels)
            if self.config['task_name'] in ['qm7', 'qm8', 'qm9', 'visc', 'cond', 'visc_hc']:
                self.mae = mean_absolute_error(labels, predictions)
                print('Test loss:', test_loss, 'Test MAE:', self.mae)
            else:
                self.rmse = mean_squared_error(labels, predictions, squared=False)
                print('Test loss:', test_loss, 'Test RMSE:', self.rmse)

        elif self.config['dataset']['task'] == 'classification': 
            predictions = np.array(predictions)
            labels = np.array(labels)
            self.roc_auc = roc_auc_score(labels, predictions[:,1])
            print('Test loss:', test_loss, 'Test ROC AUC:', self.roc_auc)

        
        predictions = np.array(predictions).reshape(-1, 5)
        labels = np.array(labels).reshape(-1, 5)
        A_pred = np.array(A_pred).reshape(-1, 5)
        ln_As = np.array(ln_As).reshape(-1, 5)
        Bs = np.array(Bs).reshape(-1, 5)
        B_pred = np.array(B_pred).reshape(-1, 5)
        
        return predictions, labels, A_pred, ln_As, B_pred, Bs


def main(config):
    dataset = MolTestDatasetWrapper(config['batch_size'], **config['dataset'])

    fine_tune = FineTune(dataset, config)

    srcc, mae = fine_tune.train(config)

    return srcc, mae
    
    # if config['dataset']['task'] == 'classification':
    #     return fine_tune.roc_auc
    # if config['dataset']['task'] == 'regression':
    #     if config['task_name'] in ['qm7', 'qm8', 'qm9', 'visc']:
    #         return fine_tune.mae
    #     else:
    #         return fine_tune.rmse


def run(task, main_folder, tune_from, runs, targets):
    '''Run the fine-tuning experiments for the given task.
    task: str, the name of the task
    main_folder: str, the folder to save the results
    tune_from: str, the name of the pre-trained model
    runs: int, the number of runs
    targets: int, the number of train splits to fine-tune on'''

    config = yaml.load(open("config_finetune.yaml", "r"), Loader=yaml.FullLoader)

    config['task_name'] = task

    if config['task_name'] == 'BBBP':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/bbbp/BBBP.csv'
        target_list = ["p_np"]

    elif config['task_name'] == 'Tox21':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/tox21/tox21.csv'
        target_list = [
            "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD", 
            "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
        ]

    elif config['task_name'] == 'ClinTox':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/clintox/clintox.csv'
        target_list = ['CT_TOX', 'FDA_APPROVED']

    elif config['task_name'] == 'HIV':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/hiv/HIV.csv'
        target_list = ["HIV_active"]

    elif config['task_name'] == 'BACE':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/bace/bace.csv'
        target_list = ["Class"]

    elif config['task_name'] == 'SIDER':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/sider/sider.csv'
        target_list = [
            "Hepatobiliary disorders", "Metabolism and nutrition disorders", "Product issues", 
            "Eye disorders", "Investigations", "Musculoskeletal and connective tissue disorders", 
            "Gastrointestinal disorders", "Social circumstances", "Immune system disorders", 
            "Reproductive system and breast disorders", 
            "Neoplasms benign, malignant and unspecified (incl cysts and polyps)", 
            "General disorders and administration site conditions", "Endocrine disorders", 
            "Surgical and medical procedures", "Vascular disorders", 
            "Blood and lymphatic system disorders", "Skin and subcutaneous tissue disorders", 
            "Congenital, familial and genetic disorders", "Infections and infestations", 
            "Respiratory, thoracic and mediastinal disorders", "Psychiatric disorders", 
            "Renal and urinary disorders", "Pregnancy, puerperium and perinatal conditions", 
            "Ear and labyrinth disorders", "Cardiac disorders", 
            "Nervous system disorders", "Injury, poisoning and procedural complications"
        ]
    
    elif config['task_name'] == 'MUV':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/muv/muv.csv'
        target_list = [
            'MUV-692', 'MUV-689', 'MUV-846', 'MUV-859', 'MUV-644', 'MUV-548', 'MUV-852',
            'MUV-600', 'MUV-810', 'MUV-712', 'MUV-737', 'MUV-858', 'MUV-713', 'MUV-733',
            'MUV-652', 'MUV-466', 'MUV-832'
        ]

    elif config['task_name'] == 'FreeSolv':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/freesolv/freesolv.csv'
        target_list = ["expt"]
    
    elif config["task_name"] == 'ESOL':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/esol/esol.csv'
        target_list = ["measured log solubility in mols per litre"]

    elif config["task_name"] == 'Lipo':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/lipophilicity/Lipophilicity.csv'
        target_list = ["exp"]
    
    elif config["task_name"] == 'qm7':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/qm7/qm7.csv'
        target_list = ["u0_atom"]

    elif config["task_name"] == 'qm8':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/qm8/qm8.csv'
        target_list = [
            "E1-CC2", "E2-CC2", "f1-CC2", "f2-CC2", "E1-PBE0", "E2-PBE0", 
            "f1-PBE0", "f2-PBE0", "E1-CAM", "E2-CAM", "f1-CAM","f2-CAM"
        ]
    
    elif config["task_name"] == 'qm9':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/qm9/qm9.csv'
        target_list = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'cv']

    elif config["task_name"] == 'visc':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'GNN_BT_Data/all_data_visc.csv'
        target_list = [['Viscosity_1', 'Viscosity_2', 'Viscosity_3', 'Viscosity_4', 'Viscosity_5', 'T1', 'T2', 'T3', 'T4', 'T5', 'Ln(A)', 'Ea/R', 'smiles']]
    elif config["task_name"] == 'cond':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'GNN_BT_Data/all_data_cond.csv'
        target_list = [['K1', 'K2', 'K3', 'K4', 'K5', 'T1', 'T2', 'T3', 'T4', 'T5', 'Intercept', 'Coefficients', 'SMILES']]
    elif config["task_name"] == 'visc_hc':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'GNN_BT_Data/hydrocarbon_visc.csv'
        target_list = [['Viscosity_1', 'Viscosity_2', 'Viscosity_3', 'Viscosity_4', 'Viscosity_5', 'T1', 'T2', 'T3', 'T4', 'T5', 'Intercept (A)', 'Coefficient (B)', 'smiles']]
    else:
        raise ValueError('Undefined downstream task!')

    print(config)

    # results_list = []
    # for target in target_list:
    #     config['dataset']['target'] = target
    #     result = main(config)
    #     results_list.append([target, result])


    # save all the srcc and mae values in a dictionary
    # save results in a yaml file
    s_avg = np.zeros((targets, 5))
    s_std = np.zeros((targets, 5))
    m_avg = np.zeros((targets, 5))
    m_std = np.zeros((targets, 5))

    BT_s_avg = np.zeros((targets, 5))
    BT_s_std = np.zeros((targets, 5))
    BT_m_avg = np.zeros((targets, 5))
    BT_m_std = np.zeros((targets, 5))

    results = {
        'scratch': np.zeros((runs, targets, 5)).tolist(),
        'BT': np.zeros((runs, targets, 5)).tolist(),
        'scratch_mae': np.zeros((runs, targets, 5)).tolist(),
        'BT_mae': np.zeros((runs, targets, 5)).tolist(),
        'axis': np.zeros(targets).tolist()
    }

    for j in range(runs):
        folder = os.path.join(main_folder, f'test_{j+1}')
        os.makedirs(folder, exist_ok=False)
        
        scratch = np.zeros((targets, 5))
        BT = np.zeros((targets, 5))
        scratch_mae = np.zeros((targets, 5))
        BT_mae = np.zeros((targets, 5))
        axis = np.zeros(targets)

        a = 0
        for i in range(6, 6-targets, -1):
            print("start")
            config['dataset']['target'] = target_list[0]
            config['dataset']['train_size'] = (i+1)*0.1
            config['save_folder'] = folder
            config['fine_tune_from'] = tune_from
            config['name'] = f'BT_{(i+1)*0.1:.1f}'
            
            srcc_BT, mae_BT = main(config)
            BT[a] = srcc_BT
            BT_mae[a] = mae_BT

            config['fine_tune_from'] = 'None'
            config['name'] = f'Scratch_{(i+1)*0.1:.1f}'
            srcc_s, mae_s = main(config)
            scratch[a] = srcc_s
            scratch_mae[a] = mae_s
            

            if config["task_name"] == 'visc':
                axis[a] = int((i+1)*0.1*477)
            elif config["task_name"] == 'cond':
                axis[a] = int((i+1)*0.1*1222)
            elif config["task_name"] == 'visc_hc':
                axis[a] = int((i+1)*0.1*182)

            
            print("done")
        
        for i in range(5):
            plot_srcc_MAE(scratch[:, i], BT[:, i], s_std[:, i], BT_s_std[:, i], axis, f'{i+1}', folder, tag2='srcc')
            plot_srcc_MAE(scratch_mae[:, i], BT_mae[:, i], m_std[:, i], BT_m_std[:, i], axis, f'{i+1}', folder, tag2='mae')

        results['scratch'][j] = scratch.tolist()
        results['BT'][j] = BT.tolist()
        results['scratch_mae'][j] = scratch_mae.tolist()
        results['BT_mae'][j] = BT_mae.tolist()
        results['axis'] = axis.tolist()

        s_avg += scratch
        m_avg += scratch_mae
        BT_s_avg += BT
        BT_m_avg += BT_mae

    s_avg /= runs
    m_avg /= runs
    BT_s_avg /= runs
    BT_m_avg /= runs

    if runs > 1:
        s_std = np.std(np.array(results['scratch']), axis=0)
        m_std = np.std(np.array(results['scratch_mae']), axis=0)
        BT_s_std = np.std(np.array(results['BT']), axis=0)
        BT_m_std = np.std(np.array(results['BT_mae']), axis=0)

        for i in range(5):
            plot_srcc_MAE(s_avg[:, i], BT_s_avg[:, i], s_std[:, i], BT_s_std[:, i], axis, f'{i+1}', main_folder, tag2='srcc', tag3 = 'avg')
            plot_srcc_MAE(m_avg[:, i], BT_m_avg[:, i], m_std[:, i], BT_m_std[:, i], axis, f'{i+1}', main_folder, tag2='mae', tag3 = 'avg')

    # Save results to YAML file
    with open(f"{main_folder}/results.yaml", "w") as file:
        yaml.dump(results, file)

    

