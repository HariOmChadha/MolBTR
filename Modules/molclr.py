import os
import shutil
import sys
import yaml

import torch
import numpy as np
# from datetime import datetime

# import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

# from utils.nt_xent import NTXentLoss
from utils.BT_loss import BT_loss
from utils.plots import plot_losses


apex_support = False
try:
    sys.path.append('./apex')
    from apex import amp
    apex_support = True
except:
    print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False

def _save_config_file(model_checkpoints_folder):
    '''Save the config file in the model folder'''
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))


class MolBTR(object):
    '''Barlow Twin Learning: https://arxiv.org/abs/2103.03230
        Training 2 models with the same architecture and using the Barlow Twin Loss.'''
    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()
        #dir_name = datetime.now().strftime('%b%d_%H-%M-%S')
        dir_name = config['name']
        os.makedirs(os.path.join('./ckpt', dir_name), exist_ok=False)
        log_dir = os.path.join('ckpt', dir_name)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.dataset = dataset
        #self.nt_xent_criterion = NTXentLoss(self.device, config['batch_size'], **config['loss'])
        

    def _get_device(self):
        '''Get the device (GPU or CPU)'''
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print("Running on:", device)
        return device

    def _step(self, model, model2, xis, xjs, n_iter):
        # get the representations and the projections
        ris, zis = model(xis)  # [N,C]
        # get the representations and the projections
        rjs, zjs = model2(xjs)  # [N,C]

        # normalize projection feature vectors
        # zis = F.normalize(zis, dim=1)
        # zjs = F.normalize(zjs, dim=1)

        # compute the loss
        on_diag, off_diag, loss = BT_loss(zis, zjs, [1 , self.config['loss_lambda']])
        return loss, on_diag, off_diag

    def train(self):
        '''Train the model'''
        train_loader, valid_loader = self.dataset.get_data_loaders()

        if self.config['model_type'] == 'gin':
            from models.ginet_molclr import GINet
            model = GINet(**self.config["model"]).to(self.device)
            model2 = GINet(**self.config["model"]).to(self.device)
            model = self._load_pre_trained_weights(model, model2)
        elif self.config['model_type'] == 'gcn':
            from models.gcn_molclr import GCN
            model = GCN(**self.config["model"]).to(self.device)
            model2 = GCN(**self.config["model"]).to(self.device)
            model = self._load_pre_trained_weights(model, model2)
        else:
            raise ValueError('Undefined GNN model.')
        print(model)
        print(model2)
        
        # define the optimizers and schedulers
        optimizer = torch.optim.Adam(
            model.parameters(), self.config['init_lr'], 
            weight_decay=eval(self.config['weight_decay'])
        )
        optimizer2 = torch.optim.Adam(
            model2.parameters(), self.config['init_lr'], 
            weight_decay=eval(self.config['weight_decay'])
        )
        
        scheduler = CosineAnnealingLR(
            optimizer, T_max=self.config['epochs']-self.config['warm_up'], 
            eta_min=0, last_epoch=-1
        )
        scheduler2 = CosineAnnealingLR(
            optimizer2, T_max=self.config['epochs']-self.config['warm_up'], 
            eta_min=0, last_epoch=-1
        )

        if apex_support and self.config['fp16_precision']:
            model, optimizer = amp.initialize(
                model, optimizer, opt_level='O2', keep_batchnorm_fp32=True
            )

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config file
        _save_config_file(model_checkpoints_folder)

        # train loop
        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf
        patience = 0

        train_losses = []
        valid_losses = []


        for epoch_counter in range(self.config['epochs']):
            print(f'Epoch: {epoch_counter +1}')
            train_l = 0
            batch_counter = 0
            for bn, (xis, xjs) in enumerate(train_loader):
                batch_counter += 1
                optimizer.zero_grad()
                optimizer2.zero_grad()

                xis = xis.to(self.device)
                xjs = xjs.to(self.device)
                # zis = model(xis)
                # zjs = model2(xjs)

                # compute the loss
                loss, on_diag, off_diag = self._step(model, model2, xis, xjs, n_iter)
                train_l += loss.item()
                
                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                    self.writer.add_scalar('cosine_lr_decay', scheduler.get_last_lr()[0], global_step=n_iter)
                    self.writer.add_scalar('cosine_lr_decay2', scheduler2.get_last_lr()[0], global_step=n_iter)
                    # print(epoch_counter, bn, loss.item())
                    print(f'Train Loss: {loss:.4f}, on_diag: {on_diag:.4f}, off_diag: {off_diag:.4f}')

                if apex_support and self.config['fp16_precision']:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                optimizer2.step()
                n_iter += 1
                
            train_l /= batch_counter
            train_losses.append(train_l)
            
            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss, c, d = self._validate(model, model2, valid_loader)
                # print(epoch_counter, bn, valid_loss, '(validation)')
                valid_losses.append(valid_loss)
                
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    patience = 0
                    torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
                    torch.save(model2.state_dict(), os.path.join(model_checkpoints_folder, 'model2.pth'))
                else:
                    patience += 1
                print(f'Val Loss: {valid_loss:.4f}, on_diag: {c:.4f}, off_diag: {d:.4f} \nbest_val_loss: {best_valid_loss:.4f} patience: {patience}/{self.config["patience"]}')
                
                if patience >= self.config['patience']:
                    print("Early stopping.")
                    break

                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1
            
            if (epoch_counter+1) % self.config['save_every_n_epochs'] == 0:
                torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model_{}.pth'.format(str(epoch_counter + 1))))
                torch.save(model2.state_dict(), os.path.join(model_checkpoints_folder, 'model2_{}.pth'.format(str(epoch_counter+ 1))))

            # warmup for the first few epochs
            if epoch_counter >= self.config['warm_up']:
                scheduler.step()
                scheduler2.step()
        
        plot_losses(train_losses, valid_losses, model_checkpoints_folder)

    def _load_pre_trained_weights(self, model, model2):
        '''Load the pre-trained weights if available'''
        try:
            checkpoints_folder = os.path.join('./ckpt', self.config['load_model'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
            state_dict2 = torch.load(os.path.join(checkpoints_folder, 'model2.pth'))
            model.load_state_dict(state_dict)
            model2.load_state_dict(state_dict2)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")
        return model

    def _validate(self, model, model2, valid_loader):
        # validation steps
        with torch.no_grad():
            model.eval()
            model2.eval()

            valid_loss = 0.0
            a = 0
            b = 0
            counter = 0
            for (xis, xjs) in valid_loader:
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                loss, on_diag, off_diag = self._step(model, model2, xis, xjs, counter)
                valid_loss += loss.item()
                a += on_diag.item()
                b += off_diag.item()
                counter += 1
            valid_loss /= counter
            a /= counter
            b /= counter
        
        model.train()
        model2.train()
        return valid_loss, a, b


def main(config):
    # change the augmentation in the config file
    if config['aug'] == 'node':
        from dataset.dataset import MoleculeDatasetWrapper
    elif config['aug'] == 'subgraph':
        from dataset.dataset_subgraph import MoleculeDatasetWrapper
    elif config['aug'] == 'mix':
        from dataset.dataset_mix import MoleculeDatasetWrapper
    else:
        raise ValueError('Not defined molecule augmentation!')

    # make sure that the smiles in the dataset are in the first column of the csv file
    dataset = MoleculeDatasetWrapper(config['batch_size'], **config['dataset'])

    molclr = MolBTR(dataset, config)
    molclr.train()


# if __name__ == "__main__":
#     main()
