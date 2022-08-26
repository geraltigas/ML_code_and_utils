# Import

import sys

import torch
import wandb

sys.path.append('../.')

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.functional import F
from myutils.template import train_model
from myutils.python_and_os import ignore_warnings, PIL_load_cuncate
from myutils.plot import show_tensor


ignore_warnings()
PIL_load_cuncate()

# config

config = {
    'project': 'Resnet',
    'lr': 1e-3,
    'epoch': 30,
    'batch_size':32,
    'dataloader_shuffle': True,
    'momentum': 0.9,
    'device': 'cuda',
    'save': True,
    'save_dir': "epoch_{}_{:.3f}.pth",
    'enable_wandb': False,
}
# Model definition

class MyResnet(nn.Module):

    def __init__(self):
        from torchvision.models import resnet18, ResNet18_Weights
        super(MyResnet,self).__init__()
        raw_resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        raw_resnet_fc_inc = raw_resnet.fc.in_features
        raw_resnet.fc = nn.Linear(raw_resnet_fc_inc,12)
        self.resnet = raw_resnet

    def forward(self,x):
        return self.resnet(x)

# Preprocess

train_transform = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5517767 , 0.52218217, 0.4580852 ],[0.22576344, 0.22649726, 0.23176326])
])
test_transform = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5510003 , 0.51879793, 0.45186004],[0.22197686, 0.22260648, 0.22860827])
])
valid_transform = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5586525 , 0.5275879 , 0.46598247],[0.22403228, 0.22568305, 0.23144639])
])

# Dataset Definition

train_dataset = ImageFolder(root='data/train',transform=train_transform)
test_dataset = ImageFolder(root='data/test',transform=test_transform)
valid_dataset = ImageFolder(root='data/valid',transform=valid_transform)

# DataLoader definition

train_dataloader = DataLoader(train_dataset,batch_size=config['batch_size'],shuffle=True)
test_dataloader = DataLoader(test_dataset)
valid_dataloader = DataLoader(valid_dataset)

# Model initialization

model = MyResnet()

# Loss function and optimizer

## Fine tune param select

other_param = []
layer4_param = []
fc_param = []
i = 0

for name, param in  model.named_parameters():
    if "layer4" in name:
        layer4_param.append(param)
    elif "fc" in name:
        fc_param.append(param)
    else:
        other_param.append(param)
    i += 1


loss_func = nn.CrossEntropyLoss()
loss_func = loss_func.to(config['device']) # using cuda
optimizer = optim.SGD([{
    'params': other_param,
    'lr':0
},{
    'params': fc_param,
    'lr': config['lr']*10
},{
    'params': layer4_param,
    'lr':config['lr']
}], config['lr'],momentum=0.9)


# Train loop

def test_model(config:dict,model:nn.Module,test_dataLoader:DataLoader): #TODO: complete the test template code
    model.eval()
    with torch.no_grad():

        num_total = 0
        acc_total = 0
        for i,(input,target) in enumerate(test_dataLoader):
            input = input.to(config['device'])
            output:torch.Tensor = model(input)
            out_index = F.softmax(output).argmax(dim=1)
            target = target.to(config['device'])
            if (out_index == target)[0].item():
                acc_total += 1
            num_total += 1

        if config['enable_wandb']:
            wandb.log({
                'test_acc': acc_total/num_total
            })
        return acc_total/num_total

def valid_model(config:dict,model:nn.Module,test_dataLoader:DataLoader):
    model.eval()
    import os

    reflect = ['dog','dragon','goat','horse','monkey','ox','pig','rabbit','ratt','rooster','snake','tiger']

    from tqdm import tqdm
    with tqdm(total=len(test_dataLoader.dataset)) as pbar:
        with torch.no_grad():

            num_total = 0
            acc_total = 0
            for i,(input,target) in enumerate(valid_dataset):
                output:torch.Tensor = model(input.reshape(1,3,224,224))
                out_index = F.softmax(output).argmax(dim=1)
                if (out_index == target)[0].item():
                    acc_total += 1
                num_total += 1
                pbar.update(1)
                if i%20 == 0:
                    show_tensor(input[0],reflect[out_index[0].item()])

            if config['enable_wandb']:
                wandb.log({
                    'valid_acc': acc_total/num_total
                })
            return acc_total/num_total

def output_process(output:torch.Tensor):

    if len(output.shape) == 2:
        # print("prediction.shape",output.shape)
        # print("prediction",output)
        return F.softmax(output,dim=1)
    else:
        # print("target.shape", output.shape)
        # print("target", output)
        return F.one_hot(output,12)

if config['enable_wandb']:
    wandb.init(project=config['project'],config=config)
# for epoch in range(config['epoch']):
#     epoch_loss = train_model(config=config,model=model,data_loader=train_dataloader,loss_func=loss_func,optimizer=optimizer,epoch_num=epoch,output_process=output_process)
#     test_model(config=config,model=model,test_dataLoader=test_dataloader)

model.load_state_dict(torch.load("./checkpoints/epoch_7_376.182.pth"))
print(valid_model(config=config,model=model,test_dataLoader=test_dataloader))

