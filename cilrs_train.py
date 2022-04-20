import numpy as np
import torch
import torch.nn as nn
import tqdm as tqdm
import yaml
from torch.utils.data import DataLoader

from expert_dataset import ExpertDataset
from models.cilrs import CILRS
import matplotlib.pyplot as plt
import wandb


def train_eval(config, model, dataloaders, criterion, optimizer, titer, viter):
    """Train model on the training dataset for one epoch"""
    # Your code here
    eval_losses = []
    train_losses = []
    for phase in ['eval', 'train']:
        train = True if phase == 'train' else False
        if train:
            model.train()
        else:
            model.eval()
        dataloader = dataloaders[phase]
        with torch.set_grad_enabled(train):
            # for (img, measurements) in tqdm.tqdm(dataloader):
            for i, (img, measurements) in enumerate(dataloader):
                img = img.cuda()
                speed = measurements['speed'].reshape(-1, 1).float().cuda()
                command = list(measurements['command'])
                throttle = measurements['throttle'].reshape(-1, 1).float().cuda()
                brake = measurements['brake'].reshape(-1, 1).float().cuda() / 0.3
                steer = measurements['steer'].reshape(-1, 1).float().cuda()

                pred_actions, pred_speed = model(img, speed, command)

                # p_throttle = torch.sigmoid(pred_actions[:, 0]).reshape(-1, 1)
                # p_brake = torch.sigmoid(pred_actions[:, 1]).reshape(-1, 1)
                # p_steer = torch.tanh(pred_actions[:, 2]).reshape(-1, 1)
                p_throttle = (pred_actions[:, 0]).reshape(-1, 1)
                p_brake = (pred_actions[:, 1]).reshape(-1, 1)
                p_steer = (pred_actions[:, 2]).reshape(-1, 1)

                throttle_loss = criterion(p_throttle, throttle)
                brake_loss = criterion(p_brake, brake)
                steer_loss = criterion(p_steer, steer)
                speed_loss = criterion(pred_speed, speed)

                loss = throttle_loss + \
                       10 * steer_loss + \
                       10 * brake_loss + \
                       speed_loss
                loss_item = loss.item()

                if train:
                    wandb.log({
                        'Throttle - {} {}'.format(phase, config.Loss): throttle_loss.item(),
                        'Steer - {} {}'.format(phase, config.Loss): steer_loss.item(),
                        'Brake - {} {}'.format(phase, config.Loss): brake_loss.item(),
                        'Speed - {} {}'.format(phase, config.Loss): speed_loss.item(),
                        'Weighted Total - {} {}'.format(phase, config.Loss): loss_item,
                        'x_axis': titer

                    })
                    titer += 1
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                else:
                    wandb.log({
                        'Throttle - {} {}'.format(phase, config.Loss): throttle_loss.item(),
                        'Steer - {} {}'.format(phase, config.Loss): steer_loss.item(),
                        'Brake - {} {}'.format(phase, config.Loss): brake_loss.item(),
                        'Speed - {} {}'.format(phase, config.Loss): speed_loss.item(),
                        'Weighted Total - {} {}'.format(phase, config.Loss): loss_item,
                        'x_axis': viter
                    })
                    viter += 1
        if train:
            train_losses.append([throttle_loss.item(), brake_loss.item(), steer_loss.item(), speed_loss.item()])
        else:
            eval_losses.append([throttle_loss.item(), brake_loss.item(), steer_loss.item(), speed_loss.item()])
    return train_losses, eval_losses, titer, viter


def plot_all(train_loss, val_loss, config):
    """Visualize your plots and save them for your report."""
    # Your code here
    train_loss = np.array(train_loss)
    val_loss = np.array(val_loss)
    for i, type in enumerate(['Throttle', 'Brake', 'Steer', 'Speed']):
        plot_losses(train_loss[:, i], val_loss[:, i], type, config=config)


def plot_losses(train_loss, val_loss, type, loss='MAE', config='-'):
    """Visualize your plots and save them for your report."""
    # Your code here
    epoch = range(len(train_loss))
    plt.title(f'{type} Loss Curve ({loss})')
    plt.plot(epoch, train_loss, label='Train')
    plt.plot(epoch, val_loss, label='Validation')
    plt.legend()
    plt.grid()
    # plt.show()
    plt.savefig(f'Config {config} {type} Loss Curve ({loss}).png')


def main():
    config = yaml.load(open('CILTrainConfig.yml', 'r'), Loader=yaml.FullLoader)
    with wandb.init(project=f"CIL Training", config=config):
        wandb.run.name = f"Config {config['ID']}"
        config = wandb.config
        # Change these paths to the correct paths in your downloaded expert dataset
        data_root = "/home/mcokelek21/Desktop/comp523/expert_data"
        train_root = "{}/train".format(data_root)
        val_root = "{}/val".format(data_root)
        model = CILRS().cuda()
        train_dataset = ExpertDataset(train_root)
        val_dataset = ExpertDataset(val_root)

        # You can change these hyper parameters freely, and you can add more
        num_epochs = config.Epochs
        batch_size = config.BatchSize

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  drop_last=True, num_workers=8)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

        train_losses = []
        val_losses = []

        criterion = nn.L1Loss()
        # for k, p in model.named_ parameters():
        #     print(k, p.requires_grad)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), config.Optim['LR'],
                                     weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[15,25], gamma=0.2,
                                                         verbose=True)
        train_iter = 0
        val_iter = 0
        dataloaders = {'train': train_loader,
                       'eval': val_loader
                       }
        for epoch in range(num_epochs):
            save_path = f"Checkpoints/Config_{config.ID}_Epoch-{epoch}"
            train_loss, val_loss, train_iter, val_iter = train_eval(config, model, dataloaders, criterion, optimizer, train_iter, val_iter)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            print('Train Epoch:', epoch, train_loss)
            print('Val Epoch:', epoch, val_loss)
            scheduler.step()
            model.eval()
            torch.save(model, save_path)
        plot_all(train_losses, val_losses, str(config.ID))


if __name__ == "__main__":
    main()
