import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import os

def plot_losses(test_loss, val_loss, model_folder_path):
    '''Plots the test and validation loss and saves the plot in the model folder'''
    plt.rcParams.update({'font.size': 12})
    # Plot the loss
    plt.plot(test_loss, label='Test')
    plt.plot(val_loss, label='Validation')
    plt.title('Model loss')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    # save the plot
    plt.savefig(f"{model_folder_path}/loss_plot.png")
    #plt.show()
    plt.close()
    return

# plot 5 seperate plots (1 for every temperature), NO subplots, and save them in a folder called "parity_plots"
def parity_plot_2(predictions, label_test, model_folder_path, tag):
    '''Plots the parity plot for the ln_A and B values and saves the plot in the model folder'''
    plt.rcParams.update({'font.size': 12})
    # make the folder for the parity plots
    os.makedirs(f"{model_folder_path}/parity_plots", exist_ok=True) 
    min_val  = min(np.min(label_test), np.min(predictions))
    max_val = max(np.max(label_test), np.max(predictions))
    print(min_val, max_val)
    for i in range(5):
        plt.scatter(label_test[:, i], predictions[:, i], s=5, color='blue')
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='dashed')
        srcc = scipy.stats.spearmanr(label_test[:, i].flatten(), predictions[:, i].flatten())[0]
        mae = np.mean(np.abs(predictions[:, i].flatten() - label_test[:, i].flatten()))
        # print(srcc, mae)
        if tag == 'B':
            plt.text(min_val, max_val - (max_val/10), f"SRCC: {srcc:.2f}, MAE: {mae: .2f}", bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
            plt.xlabel('True Activation')
            plt.ylabel('Predicted Activation')
            plt.title(f'Parity plot for Activation')
        elif tag == 'ln(A)':
            plt.text(min_val, max_val + (max_val/15), f"SRCC: {srcc:.2f}, MAE: {mae: .2f}", bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
            plt.xlabel('True ln(A)')
            plt.ylabel('Predicted ln(A)')
            plt.title(f'Parity plot for ln(A)')
        plt.savefig(f"{model_folder_path}/parity_plots/parity_plot_{tag}.png")
        #plt.show()
        plt.close()
        break
    return

def parity_plot_3(predictions, label_test, model_folder_path, tag):
    '''Plots the parity plot for the predictions and labels and saves the plot in the model folder'''
    # make the folder for the parity plots
    plt.rcParams.update({'font.size': 12})
    os.makedirs(f"{model_folder_path}/parity_plots", exist_ok=True) 
    min_val  = min(np.min(label_test), np.min(predictions))
    max_val = max(np.max(label_test), np.max(predictions))
    s = []
    m = []
    print(min_val, max_val)
    for i in range(5):
        plt.scatter(label_test[:, i], predictions[:, i], s=5, color='blue')
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='dashed')
        srcc = scipy.stats.spearmanr(label_test[:, i].flatten(), predictions[:, i].flatten())[0]
        mae = np.mean(np.abs(predictions[:, i].flatten() - label_test[:, i].flatten()))
        s.append(srcc)
        m.append(mae)
        # print(srcc, mae)
        if tag == 'cond':
            plt.text(min_val, max_val + (max_val/20), f"SRCC: {srcc:.2f}, MAE: {mae: .2f}", bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
            plt.xlabel('True ln(Thermal conductivity)')
            plt.ylabel('Predicted ln(Thermal conductivity)')
            plt.title(f'Parity plot at ' + f'$T_{i+1}$')
            plt.savefig(f"{model_folder_path}/parity_plots/parity_plot_T{i+1}.png")
        elif tag == 'visc':
            plt.text(min_val, max_val - (max_val/20), f"SRCC: {srcc:.2f}, MAE: {mae: .2f}", bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
            plt.xlabel('True ln(Dynamic Viscosity)')
            plt.ylabel('Predicted ln(Dynamic Viscosity)')
            plt.title(f'Parity plot at ' + f'$T_{i+1}$')
            plt.savefig(f"{model_folder_path}/parity_plots/parity_plot_T{i+1}.png")
        #plt.show()
        plt.close()
    return s, m

def plot_colour(y, x):
    plt.figure(figsize=(8, 6))
    hb = plt.hexbin(x, y, gridsize = 70, cmap = 'viridis', bins = 'log')
    cb = plt.colorbar(hb, label = 'Number of points')
    cb.set_label('Number of points', fontsize = 13)
    cb.ax.tick_params(labelsize = 13, size = 10)
    plt.plot(x, x, 'r')
    plt.xlabel('Actual ln(Thermal Conductivity)', fontsize = 13)
    plt.ylabel('Predicted ln(Thermal Conductivity)', fontsize = 13)
    plt.xlim([np.min(x)-0.1, np.max(x)+0.1])
    plt.ylim([np.min(x)-0.1, np.max(x)+0.1])
    plt.tick_params(labelsize = 14)
    plt.title("Parity Plot", fontsize=13)
    srcc = scipy.stats.spearmanr(x[:, 3].flatten(), y[:, 3].flatten())[0]
    mae = np.mean(np.abs(y[:, 3].flatten() - x[:, 3].flatten()))
    textstr = f'MAE: {mae:.2f}\nSRCC: {srcc:.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=14, verticalalignment='top', bbox=props)
    plt.grid(True)
    plt.savefig('plot2.png', transparent = True)
    plt.close()
    return

def plot_srcc_MAE(scratch, BT, s_std, BT_std, axis, tag, folder, tag2, tag3=None):
    '''Plots the SRCC and MAE values for the scratch and BT models and saves the plot in the folder.
    Tag1: Temperature
    Tag2: srcc or mae
    Tag3: avg over multiple tries or not'''
    plt.rcParams.update({'font.size': 12})
    if tag2 == 'srcc':
        plt.plot(axis, scratch, label='random weights - GCN', color='red', linestyle='dashed', marker='o')
        plt.fill_between(axis, scratch - s_std, scratch + s_std, color='red', alpha=0.2)
        plt.plot(axis, BT, label='pretrained weights using BT - GCN', color='blue', linestyle='dashed', marker='o')
        plt.fill_between(axis, BT - BT_std, BT + BT_std, color='blue', alpha=0.2)
        plt.xlabel('Number of data points')
        if tag3 == 'avg':
            plt.ylabel('Average SRCC')
            plt.title(f'Average SRCC vs Number of train data points at ' + f'$T_{tag}$')
            folder = os.path.join(folder, 'srcc_avg_plots')
        else:
            plt.ylabel('SRCC')
            plt.title(f'SRCC vs Number of train data points at ' + f'$T_{tag}$')
            folder = os.path.join(folder, 'srcc_plots')
        plt.legend()
        os.makedirs(folder, exist_ok=True)
        plt.savefig(f"{folder}/srcc_T{tag}.png")
        plt.close()
    elif tag2 == 'mae':
        plt.plot(axis, scratch, label='random weights', color='red', linestyle='dashed', marker='o')
        plt.fill_between(axis, scratch - s_std, scratch + s_std, color='red', alpha=0.2)
        plt.plot(axis, BT, label='pretrained weights using BT', color='blue', linestyle='dashed', marker='o')
        plt.fill_between(axis, BT - BT_std, BT + BT_std, color='blue', alpha=0.2)
        plt.xlabel('Number of data points')
        if tag3 == 'avg':
            plt.ylabel('Average MAE')
            plt.title(f'Average MAE vs Number of train data points at ' + f'$T_{tag}$')
            folder = os.path.join(folder, 'mae_avg_plots')
        else:
            plt.ylabel('MAE')
            plt.title(f'MAE vs Number of train data points at ' + f'$T_{tag}$')
            folder = os.path.join(folder, 'mae_plots')
        plt.legend()
        os.makedirs(folder, exist_ok=True)
        plt.savefig(f"{folder}/mae_T{tag}.png")
        plt.close()

def plot_srcc_MAE_2(scratch, s_std, axis, tag, folder, tag2, tag3=None):
    plt.rcParams.update({'font.size': 12})
    if tag2 == 'srcc':
        plt.plot(axis, scratch, label='random weights', color='red', linestyle='dashed', marker='o')
        plt.fill_between(axis, scratch - s_std, scratch + s_std, color='red', alpha=0.2)
        # plt.plot(axis, BT, label='weights transfered from visc ANN', color='blue', linestyle='dashed', marker='o')
        # plt.fill_between(axis, BT - BT_std, BT + BT_std, color='blue', alpha=0.2)
        plt.xlabel('Number of data points')
        if tag3 == 'avg':
            plt.ylabel('Average SRCC')
            plt.title(f'Average SRCC vs Number of train data points at ' + f'$T_{tag}$')
            folder = os.path.join(folder, 'srcc_avg_plots')
        else:
            plt.ylabel('SRCC')
            plt.title(f'SRCC vs Number of train data points at ' + f'$T_{tag}$')
            folder = os.path.join(folder, 'srcc_plots')
        plt.legend()
        os.makedirs(folder, exist_ok=True)
        plt.savefig(f"{folder}/srcc_T{tag}.png")
        plt.close()
    elif tag2 == 'mae':
        plt.plot(axis, scratch, label='random weights', color='red', linestyle='dashed', marker='o')
        plt.fill_between(axis, scratch - s_std, scratch + s_std, color='red', alpha=0.2)
        # plt.plot(axis, BT, label='weights transfered from visc ANN', color='blue', linestyle='dashed', marker='o')
        # plt.fill_between(axis, BT - BT_std, BT + BT_std, color='blue', alpha=0.2)
        plt.xlabel('Number of data points')
        if tag3 == 'avg':
            plt.ylabel('Average MAE')
            plt.title(f'Average MAE vs Number of train data points at ' + f'$T_{tag}$')
            folder = os.path.join(folder, 'mae_avg_plots')
        else:
            plt.ylabel('MAE')
            plt.title(f'MAE vs Number of train data points at ' + f'$T_{tag}$')
            folder = os.path.join(folder, 'mae_plots')
        plt.legend()
        os.makedirs(folder, exist_ok=True)
        plt.savefig(f"{folder}/mae_T{tag}.png")
        plt.close()
    
def plot_srcc_MAE_3(scratch, BT, s_std, BT_std, axis2, scratch_2, BT_2, s_std_2, BT_std_2, axis, tag, folder, tag2, tag3=None):
    plt.rcParams.update({'font.size': 12})
    if tag2 == 'srcc':
        plt.plot(axis2, scratch, label='random weights', color='red', linestyle='dashed', marker='o')
        plt.fill_between(axis2, scratch - s_std, scratch + s_std, color='red', alpha=0.2)
        plt.plot(axis2, BT, label='weights transfered from visc ANN', color='blue', linestyle='dashed', marker='o')
        plt.fill_between(axis2, BT - BT_std, BT + BT_std, color='blue', alpha=0.2)
        plt.plot(axis, scratch_2, label='random weights - GNN', color='green', linestyle='dashed', marker='o')
        plt.fill_between(axis, scratch_2 - s_std_2, scratch_2 + s_std_2, color='green', alpha=0.2)
        plt.plot(axis, BT_2, label='weights transfered from visc GNN', color='orange', linestyle='dashed', marker='o')
        plt.fill_between(axis, BT_2 - BT_std_2, BT_2 + BT_std_2, color='orange', alpha=0.2)
        plt.xlabel('Number of data points')
        if tag3 == 'avg':
            plt.ylabel('Average SRCC')
            plt.title(f'Average SRCC vs Number of train data points at ' + f'$T_{tag}$')
            folder = os.path.join(folder, 'srcc_avg_plots')
        else:
            plt.ylabel('SRCC')
            plt.title(f'SRCC vs Number of train data points at ' + f'$T_{tag}$')
            folder = os.path.join(folder, 'srcc_plots')
        plt.legend()
        os.makedirs(folder, exist_ok=True)
        plt.savefig(f"{folder}/srcc_T{tag}.png")
        plt.close()
    elif tag2 == 'mae':
        plt.plot(axis2, scratch, label='random weights', color='red', linestyle='dashed', marker='o')
        plt.fill_between(axis2, scratch - s_std, scratch + s_std, color='red', alpha=0.2)
        plt.plot(axis2, BT, label='weights transfered from visc ANN', color='blue', linestyle='dashed', marker='o')
        plt.fill_between(axis2, BT - BT_std, BT + BT_std, color='blue', alpha=0.2)
        plt.plot(axis, scratch_2, label='random weights - GNN', color='green', linestyle='dashed', marker='o')
        plt.fill_between(axis, scratch_2 - s_std_2, scratch_2 + s_std_2, color='green', alpha=0.2)
        plt.plot(axis, BT_2, label='weights transfered from visc GNN', color='orange', linestyle='dashed', marker='o')
        plt.fill_between(axis, BT_2 - BT_std_2, BT_2 + BT_std_2, color='orange', alpha=0.2)
        plt.xlabel('Number of data points')
        if tag3 == 'avg':
            plt.ylabel('Average MAE')
            plt.title(f'Average MAE vs Number of train data points at ' + f'$T_{tag}$')
            folder = os.path.join(folder, 'mae_avg_plots')
        else:
            plt.ylabel('MAE')
            plt.title(f'MAE vs Number of train data points at ' + f'$T_{tag}$')
            folder = os.path.join(folder, 'mae_plots')
        plt.legend()
        os.makedirs(folder, exist_ok=True)
        plt.savefig(f"{folder}/mae_T{tag}.png")
        plt.close()
