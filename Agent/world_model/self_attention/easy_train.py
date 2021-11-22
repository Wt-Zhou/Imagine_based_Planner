from modeling.vectornet import HGNN
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
import pandas as pd
# from utils.viz_utils import show_predict_result
import matplotlib.pyplot as plt
import numpy as np
import pdb
import os
from torch_geometric.data import DataLoader, DataListLoader, Data
# from utils.eval import get_eval_metric_results
from tqdm import tqdm
import torch_geometric.nn as nn
import time
# from dataset import VirtualDataset

# %%
# train related
TRAIN_DIR = os.path.join('interm_data', 'train_intermediate')
VAL_DIR = os.path.join('interm_data', 'val_intermediate')
gpus = [2, 3]
SEED = 13
epochs = 25
batch_size = 4096 * len(gpus)
decay_lr_factor = 0.3
decay_lr_every = 5
lr = 0.01
in_channels, out_channels = 4, 4
show_every = 20
val_every = 5
small_dataset = False
end_epoch = 0
save_dir = 'trained_params'
best_minade = float('inf')
global_step = 0
date = '200621'
# eval related
max_n_guesses = 1
horizon = 30
miss_threshold = 2.0


#%%
def save_checkpoint(checkpoint_dir, model, optimizer, end_epoch, val_minade, date):
    # state_dict: a Python dictionary object that:
    # - for a model, maps each layer to its parameter tensor;
    # - for an optimizer, contains info about the optimizerâ€™s states and hyperparameters used.
    os.makedirs(checkpoint_dir, exist_ok=True)
    state = {
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'end_epoch' : end_epoch,
        'val_minade': val_minade
        }
    checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{end_epoch}.valminade_{val_minade:.3f}.{date}.{"xkhuang"}.pth')
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)
    
def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)
    return checkpoint_path['end_epoch']

#%%
if __name__ == "__main__":
    # training envs
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device(f'cuda:{gpus[0]}' if torch.cuda.is_available() else 'cpu')



    # virtual data for gnn
    x = torch.tensor([[ 0.8231,  0.5371,  0.2737,  0.0750,  0.4939],  [ 0.9797, -0.0893,  0.8225,  0.0374,  0.4955]], dtype=torch.float)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    y = torch.tensor([[ 0.8325,  0.5130,  0.3191,  0.0887,  0.4940],
        [ 0.9799, -0.1144,  0.8309,  0.0374,  0.4956]], dtype=torch.float)
    cluster = torch.tensor([1,2], dtype=torch.float)
    valid_len = torch.tensor([[2], [2]], dtype=torch.float)
    time_step_len = torch.tensor([[1], [1]], dtype=torch.float)
    virtual_data = Data(x=x, edge_index=edge_index, y=y, cluster=cluster, valid_len=valid_len, time_step_len=time_step_len)
    # for i in range(0,20):
    #     virual_data.append([i+1,i+2,i+3,i+4,2*i,3*i,4*i,5*i])

    model = HGNN(in_channels, out_channels)
    # model = nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])
    model = model.to(device=device)
    virtual_data = virtual_data.to(device=device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=decay_lr_every, gamma=decay_lr_factor)
    model.train()
    for j in range(0,100):
        y = virtual_data.y          
        optimizer.zero_grad()
        out = model(virtual_data)
        loss = F.mse_loss(out, y)
        # print("out",out)
        print("Test_Loss",loss)
        loss.backward()
        # acc_loss += batch_size * loss.item()
        # num_samples += y.shape[0]
        optimizer.step()



# %%
