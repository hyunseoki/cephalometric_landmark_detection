import sys
import tqdm
import torch
import os
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        prob, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def cutmix(data, target, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]

    lam = np.clip(np.random.beta(alpha, alpha),0.3,0.4)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    new_data = data.clone()
    new_data[:, :, bby1:bby2, bbx1:bbx2] = data[indices, :, bby1:bby2, bbx1:bbx2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
    targets = (target, shuffled_target, lam)

    return new_data, targets


def train(train_loader, model, loss_func, metric_func, device, optimizer, use_amp):
    running_metric = AverageMeter()
    running_loss = AverageMeter()

    if use_amp:
        scaler = torch.cuda.amp.GradScaler()

    model.train()       

    with tqdm.tqdm(train_loader, total=len(train_loader), desc="Train", file=sys.stdout) as iterator:
        for sample in iterator:
            train_x, train_y = sample['input'], sample['target']
            train_x, train_y = train_x.to(device), train_y.to(device)
            
            optimizer.zero_grad()

            if use_amp:
                with torch.cuda.amp.autocast():
                    output = model(train_x)
                    loss = loss_func(output, train_y)
                    loss_value = loss.detach().cpu().numpy()
                    metric_value = metric_func(output.detach(), train_y.detach()).cpu().numpy()

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()  

            else:
                output = model(train_x)
                loss = loss_func(output, train_y)
                loss_value = loss.detach().cpu().numpy()
                metric_value = metric_func(output.detach(), train_y.detach()).cpu().numpy()

                loss.backward()
                optimizer.step()
                
            running_loss.update(loss_value.item(), train_x.size(0))
            running_metric.update(metric_value.item(), train_x.size(0))                
            
            log = 'loss - {:.5f}, metric - {:.5f}'.format(running_loss.avg, running_metric.avg)
            iterator.set_postfix_str(log)

    return running_loss.avg, running_metric.avg

def validate(valid_loader, model, loss_func, metric_func, device):
    running_metric = AverageMeter()
    running_loss = AverageMeter()
    model.eval()

    with tqdm.tqdm(valid_loader, total=len(valid_loader), desc="Valid", file=sys.stdout) as iterator:
        for sample in iterator:
            train_x, train_y = sample['input'], sample['target']
            train_x, train_y = train_x.to(device), train_y.to(device)

            with torch.no_grad():
                output = model.forward(train_x)
            
            loss = loss_func(output, train_y)
            loss_value = loss.detach().cpu().numpy()
            metric_value = metric_func(output.detach(), train_y.detach()).cpu().numpy() 

            running_loss.update(loss_value, train_x.size(0))
            running_metric.update(metric_value, train_x.size(0))                
            
            log = 'loss - {:.5f}, metric - {:.5f}'.format(running_loss.avg, running_metric.avg)
            iterator.set_postfix_str(log)

    return running_loss.avg, running_metric.avg


class ModelTrainer:
    def __init__(self, model, train_loader, valid_loader, loss_func, metric_func, optimizer, device, save_dir, 
                       mode='max', scheduler=None, num_epochs=25, num_snapshops=None, parallel=False, use_amp=True, use_wandb=True):

        assert mode in ['min', 'max']

        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.loss_func = loss_func
        self.metric_func = metric_func
        self.optimizer = optimizer
        self.device = device
        self.mode = mode
        self.save_path = str(os.path.join(save_dir, datetime.now().strftime("%m%d%H%M%S")))        

        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.num_snapshops = num_snapshops
        self.parallel = parallel
        self.use_amp = use_amp
        self.use_wandb = use_wandb

        self.elapsed_time = None

        self.train_loss = list()
        self.train_metric = list()

        self.valid_loss = list()
        self.valid_metric = list()

        self.lr_curve = list()

    def initWandb(self, project_name, run_name, args):
        assert self.use_wandb == True      
    
        wandb.init(project=project_name)
        wandb.run.name = run_name
        wandb.config.update(args)
        wandb.watch(self.model)

    def train(self):
        """fit a model"""

        if self.device == 'cpu':
            print('[info msg] Start training the model on CPU')
        elif self.parallel and torch.cuda.device_count() > 1:
            print(f'[info msg] Start training the model on {torch.cuda.device_count()} '
                  f'{torch.cuda.get_device_name(torch.cuda.current_device())} in parallel')
            self.model = torch.nn.DataParallel(self.model)
        else:
            print(f'[info msg] Start training the model on {torch.cuda.get_device_name(torch.cuda.current_device())}')
        print('=' * 50)              
        
        if self.mode =='max':
            best_metric = -float('inf')
        else:
            best_metric = float('inf')

        if self.num_snapshops is not None:
            best_snap_metric = best_metric
            snapshop_period = self.num_epochs // self.num_snapshops
            cur_num_snapshop = 0            
            snapshop = None

        self.model = self.model.to(self.device)
        startTime = datetime.now()     

        print('[info msg] training start !!')
        for epoch in range(self.num_epochs):        
            print('Epoch {}/{}'.format(epoch+1, self.num_epochs))
            train_epoch_loss, train_epoch_metric = train(
                train_loader=self.train_loader,
                model=self.model,
                loss_func=self.loss_func,
                metric_func=self.metric_func,
                device=self.device,
                optimizer=self.optimizer,
                use_amp=self.use_amp
                )
            self.train_loss.append(train_epoch_loss)
            self.train_metric.append(train_epoch_metric)

            valid_epoch_loss, valid_epoch_metric = validate(
                valid_loader=self.valid_loader,
                model=self.model,
                loss_func=self.loss_func,
                metric_func=self.metric_func,
                device=self.device,
                )                
            self.valid_loss.append(valid_epoch_loss)        
            self.valid_metric.append(valid_epoch_metric)
            self.lr_curve.append(self.optimizer.param_groups[0]['lr'])

            if self.use_wandb:
                wandb.log({
                    "Train Acc": train_epoch_metric,
                    "Valid Acc": valid_epoch_metric,
                    "Train Loss": train_epoch_loss,
                    "Valid Loss": valid_epoch_loss,
                    })

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(valid_epoch_metric)
                else:
                    self.scheduler.step()
                    # raise NotImplementedError()

            if (self.mode =='min' and valid_epoch_metric < best_metric) or \
               (self.mode =='max' and valid_epoch_metric > best_metric) :
                best_metric = valid_epoch_metric                
                self.__save_model(param=self.model.state_dict(), fn='model_best.pth')

            if self.num_snapshops is not None:
                if (self.mode =='min' and valid_epoch_metric < best_snap_metric) or \
                   (self.mode =='max' and valid_epoch_metric > best_snap_metric) :
                    best_snap_metric = valid_epoch_metric
                    snapshop = self.model.state_dict()

                ## save snapshop            
                if (epoch + 1) % snapshop_period == 0:                             
                    self.__save_model(param=snapshop, fn=f"snapshop_{cur_num_snapshop}.pth")

                    if self.mode =='max':
                        best_snap_metric = -float('inf')
                    elif self.mode =='min':
                        best_snap_metric = float('inf')
                    
                    cur_num_snapshop+=1

        self.elapsed_time = datetime.now() - startTime
        self.__save_result()

    def __save_model(self, param, fn):
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        full_name = os.path.join(self.save_path, fn)
        torch.save(param, full_name)
        print('MODEL IS SAVED TO {}!!!'.format(full_name))

    def __save_result(self):    
        train_loss = np.array(self.train_loss)
        train_metric = np.array(self.train_metric)
        valid_loss = np.array(self.valid_loss)
        valid_metric = np.array(self.valid_metric)

        if self.mode =='max':
            best_train_metric_pos = np.argmax(train_metric)
            best_train_metric = train_metric[best_train_metric_pos]
            best_train_loss = train_loss[best_train_metric_pos]

            best_val_metric_pos = np.argmax(valid_metric)
            best_val_metric = valid_metric[best_val_metric_pos]
            best_val_loss = valid_loss[best_val_metric_pos]

        if self.mode =='min':
            best_train_metric_pos = np.argmin(train_metric)
            best_train_metric = train_metric[best_train_metric_pos]
            best_train_loss = train_loss[best_train_metric_pos]

            best_val_metric_pos = np.argmin(valid_metric)
            best_val_metric = valid_metric[best_val_metric_pos]      
            best_val_loss = valid_loss[best_val_metric_pos]
        
        print('=' * 50)
        print('[info msg] training is done')
        print("Time taken: {}".format(self.elapsed_time))
        print("best metric is {} w/ loss {} at epoch : {}".format(best_val_metric, best_val_loss, best_val_metric_pos))    

        print('=' * 50)
        print('[info msg] model weight and log is save to {}'.format(self.save_path))

        with open(os.path.join(self.save_path, 'log.txt'), 'w') as f:         
            f.write(f'total ecpochs : {train_loss.shape[0]}\n')
            f.write(f'time taken : {self.elapsed_time}\n')
            f.write(f'best_train_metric {best_train_metric} w/ loss {best_train_loss} at epoch : {best_train_metric_pos}\n')
            f.write(f'best_valid_metric {best_val_metric} w/ loss {best_val_loss} at epoch : {best_val_metric_pos}\n')

        df_learning_curves = pd.DataFrame.from_dict({
                    'loss_train': train_loss,
                    'loss_val': valid_loss,
                    'metric_train': train_metric,
                    'metric_val': valid_metric,
                })

        df_learning_curves.to_csv(os.path.join(self.save_path, 'learning_curves.csv'), sep=',')

        plt.figure(figsize=(15,5))
        plt.subplot(1, 2, 1)
        plt.title('loss')
        plt.plot(train_loss, label='train loss')
        plt.plot(valid_loss, label='valid loss')
        plt.axvline(x=best_val_metric_pos, color='r', linestyle='--', linewidth=1.5, label='best_val_metric')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.title('metric')
        plt.plot(train_metric, label='train metric')
        plt.plot(valid_metric, label='valid metric')
        plt.axvline(x=best_val_metric_pos, color='r', linestyle='--', linewidth=1.5, label='best_val_metric')
        plt.legend()
        plt.savefig(os.path.join(self.save_path, 'history.png'))
        # plt.show()
        
        plt.figure(figsize=(15,5))
        plt.title('lr_rate curve')
        plt.plot(self.lr_curve)
        plt.savefig(os.path.join(self.save_path, 'lr_history.png'))
        # plt.show()

    @property
    def save_dir(self):
        return self.save_path

if __name__ == '__main__':
    pass