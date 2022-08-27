from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch import Tensor,save
import os
from tqdm import tqdm
from collections.abc import Callable
import wandb

def train_model(config:dict,model:Module,data_loader:DataLoader,loss_func:Module,optimizer:Optimizer,epoch_num:int,output_process:Callable[[Tensor],Tensor]) -> int:

    loss_total = 0
    model.train()
    model.to(config['device'])
    with tqdm(total=len(data_loader.dataset),desc='Epoch {}'.format(epoch_num)) as pbar:
        for index,data_pair in enumerate(data_loader):
            featrues,labels = data_pair
            featrues:Tensor = featrues.to(config['device'])
            labels:Tensor = labels.to(config['device'])

            pres = model.forward(featrues)

            pres = output_process(pres).float()
            labels = output_process(labels).float()

            batch_loss:Tensor = loss_func(pres,labels)

            if config['DEBUG']:
                print("pres and labels : ",pres,labels)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            pbar.update(featrues.shape[0])

            loss_total += batch_loss.item()

    if config['save']:
        os.path.exists('checkpoints') or os.mkdir('checkpoints')
        save(model.state_dict(),('checkpoints/'+config['save_dir']).format(epoch_num,loss_total))

    if config['enable_wandb']:
        wandb.log({
            'epoch': epoch_num,
            'loss': loss_total
        })

    return loss_total


