# imports
import sys

import torch
import wandb
from myutils.plot import show_tensor

sys.path.append('../.')

from torch.utils.data import DataLoader,Dataset
from torch import nn, optim
from torch.functional import F
from myutils.template import train_model
from myutils.python_and_os import ignore_warnings

ignore_warnings()


# config

config = {
    'project': 'LSTM',
    'lr': 1e-3,
    'epoch': 30,
    'batch_size':1,
    'dataloader_shuffle': True,
    'momentum': 0.9,
    'device': 'cuda',
    'save': True,
    'save_dir': "epoch_{}_{:.3f}.pth",
    'enable_wandb': False,
    'LSTM_window': 10,
    'DEBUG': False,
    'LOG': False,
    'LOG_DIR': './log.txt',
}

# log relocation

import os

file_path = config['LOG_DIR']

if config['LOG']:
    if os.path.exists(file_path):
      os.remove(file_path)
    sys.stdout = open(file_path, "w")

# Model definition

class MyLSTM(nn.Module):

    def __init__(self,input_dem:int,hs_dem:int):

        super(MyLSTM,self).__init__()

        self.hs_dem = hs_dem

        # none grad parameter
        self.cell_state = torch.zeros(hs_dem).to(config['device'])
        self.hidden_state = torch.zeros(hs_dem).to(config['device'])

        # require grad parameter
        self.w_xi = nn.Parameter(torch.randn(input_dem,hs_dem,dtype=float)*0.01,requires_grad=True)
        self.w_xf = nn.Parameter(torch.randn(input_dem,hs_dem,dtype=float)*0.01,requires_grad=True)
        self.w_xo = nn.Parameter(torch.randn(input_dem,hs_dem,dtype=float)*0.01,requires_grad=True)
        self.w_xc = nn.Parameter(torch.randn(input_dem,hs_dem,dtype=float)*0.01,requires_grad=True)

        self.w_hi = nn.Parameter(torch.randn(hs_dem,hs_dem,dtype=float)*0.01,requires_grad=True)
        self.w_hf = nn.Parameter(torch.randn(hs_dem,hs_dem,dtype=float)*0.01,requires_grad=True)
        self.w_ho = nn.Parameter(torch.randn(hs_dem,hs_dem,dtype=float)*0.01,requires_grad=True)
        self.w_hc = nn.Parameter(torch.randn(hs_dem,hs_dem,dtype=float)*0.01,requires_grad=True)

        self.b_i = nn.Parameter(torch.zeros(hs_dem,dtype=float),requires_grad=True)
        self.b_f = nn.Parameter(torch.zeros(hs_dem,dtype=float),requires_grad=True)
        self.b_o = nn.Parameter(torch.zeros(hs_dem,dtype=float),requires_grad=True)
        self.b_c = nn.Parameter(torch.zeros(hs_dem,dtype=float),requires_grad=True)

    def _forward(self,x):

        def sigmond_and_matmul(o1,o2,o3,o4,o5):
            t1 = torch.matmul(o1,o2)
            t2 = torch.matmul(o3.double().to(config['device']),o4.double().to(config['device']))
            t3 = o5
            t4 = t1 + t2 + t3
            t5 = F.sigmoid(t4)
            return t5.to(config['device'])
        i = sigmond_and_matmul(x, self.w_xi.data, self.hidden_state.data, self.w_hi.data, self.b_i.data)
        f = sigmond_and_matmul(x, self.w_xf.data, self.hidden_state.data, self.w_hf.data, self.b_f.data)
        o = sigmond_and_matmul(x, self.w_xo.data, self.hidden_state.data, self.w_ho.data, self.b_o.data)
        c = sigmond_and_matmul(x, self.w_xc.data, self.hidden_state.data, self.w_hc.data, self.b_c.data)

        self.cell_state = torch.mul(f,self.cell_state.data) + torch.dot(i,c)
        self.cell_state = (self.cell_state - self.cell_state.mean())/ (self.cell_state.std() + 1e-5)
        self.hidden_state = torch.mul( o , torch.tanh(self.cell_state.data))
        if config['DEBUG']:
            print("four middle var: ", i, f, o, c)
            print("two states shape: ", self.cell_state.shape, self.hidden_state.shape)
            print("two state: ",self.cell_state, self.hidden_state)
        return self.hidden_state

    def forward(self,x):
        return self._forward(x)

    def clean_states(self):
        self.cell_state = torch.zeros(self.hs_dem).to(config['device'])
        self.hidden_state = torch.zeros(self.hs_dem).to(config['device'])

class MySinPreModel(nn.Module):
    def __init__(self,input_num:int,input_dem:int,hs_dem:int):
        super(MySinPreModel,self).__init__()
        self.input_num = input_num
        self.lstm = MyLSTM(input_dem=input_dem,hs_dem=hs_dem)
        self.mlp = nn.Sequential(
            nn.Linear(hs_dem,1000),
            nn.LayerNorm([1000]),
            nn.ReLU(),
            nn.Linear(1000,100),
            nn.LayerNorm([100]),
            nn.ReLU(),
            nn.Linear(100,1)
        )

    def forward(self,x):
        x = x.reshape(config['LSTM_window'],1)
        for index in range(self.input_num):
            self.lstm(x[index])

        return_value = self.mlp(self.lstm.hidden_state.float())
        self.lstm.clean_states()
        return return_value

# Preprocess

# Dataset Definition

class MySinDataSet(Dataset):
    def __init__(self,input_size:int,dataset_size:int,start:float = None,end:float = None):
        import numpy as np
        self.dataset_size = dataset_size
        self.input_size = input_size
        self.x = np.linspace(start=start,stop=end,num=dataset_size+input_size+1)
        self.data = np.sin(self.x)
    def __len__(self):
        return self.dataset_size
    def __getitem__(self, item):
        return torch.asarray(self.data[item:item+self.input_size]),torch.asarray([self.data[item+self.input_size+1]])

# DataLoader definition

train_dataloader = DataLoader(dataset=MySinDataSet(config['LSTM_window'],10000,start=0,end=100),batch_size=config['batch_size'],shuffle=True)
test_dataloader = DataLoader(dataset=MySinDataSet(config['LSTM_window'],1000,start=101,end=111),batch_size=config['batch_size'],shuffle=True)
valid_dataloader = DataLoader(dataset=MySinDataSet(config['LSTM_window'],1000,start=112,end=122),batch_size=config['batch_size'],shuffle=True)
# Model initialization

model = MySinPreModel(input_num=config['LSTM_window'],input_dem=1,hs_dem=100)

def init_model(model:nn.Module):
    for i in model.parameters():
        if isinstance(i,nn.Linear):
            nn.init.xavier_uniform_(i.weight)
            nn.init.constant_(i.bias,0)

init_model(model)

# Loss function and optimizer

## Fine tune param select

loss_func = nn.MSELoss()
loss_func = loss_func.to(config['device']) # using cuda
optimizer = optim.SGD(model.parameters(),config['lr'],momentum=0.9,weight_decay=0.001)


# Train loop

def test_model(config:dict,model:nn.Module,test_dataLoader:DataLoader):
    model.eval()
    from tqdm import tqdm
    with tqdm(total=len(test_dataLoader.dataset), desc="Valid") as pbar:
        with torch.no_grad():
            miss_total = []
            for i,(input,target) in enumerate(test_dataLoader):
                input = input.to(config['device'])
                output:torch.Tensor = model(input)
                target = target.to(config['device'])
                miss_total.append(abs(output - target).item())
                pbar.update(1)

            if config['enable_wandb']:
                wandb.log({
                    'test_miss': torch.asarray(miss_total).mean().item()
                })
            return torch.asarray(miss_total).mean().item()

def valid_model(config:dict,model:nn.Module,valid_dataloader:DataLoader) -> float:
    model.eval()
    from tqdm import tqdm
    with tqdm(total=len(valid_dataloader.dataset), desc="Valid") as pbar:
        with torch.no_grad():
            miss_total = []
            labels = []
            pres = []
            for i,(input,target) in enumerate(valid_dataloader):
                input = input.to(config['device'])
                output:torch.Tensor = model(input)
                target = target.to(config['device'])
                miss_total.append(abs(output - target).item())
                labels.append(target.item())
                pres.append(output.item())
                pbar.update(1)

            if config['enable_wandb']:
                wandb.log({
                    'valid_miss': torch.asarray(miss_total).mean().item()
                })

            from myutils.plot import plot_y

            plot_y([labels[0:50],pres[0:50]],title="valid")

            return torch.asarray(miss_total).mean().item()


# process after model output and before loss func
def output_process(output:torch.Tensor):
    return output

# if config['enable_wandb']:
#     wandb.init(project=config['project'],config=config)
# for epoch in range(config['epoch']):
#     epoch_loss = train_model(config=config,model=model,data_loader=train_dataloader,loss_func=loss_func,optimizer=optimizer,epoch_num=epoch,output_process=output_process)
#     test_model(config=config,model=model,test_dataLoader=test_dataloader)

model.to(config['device'])
model.load_state_dict(torch.load('./checkpoints/epoch_9_410.152.pth'))
print(valid_model(config=config,model=model,valid_dataloader=valid_dataloader))
