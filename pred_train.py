import torch
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import yaml
from torch.utils.data import DataLoader

from expert_dataset import ExpertDataset
from models.affordance_predictor import AffordancePredictor
import wandb


def train_eval(config, phase, model, dataloader, criterions, optimizer, iteration):
    train = phase == 'train'
    if train:
        model.train()
    else:
        model.eval()
    """Train model on the training dataset for one epoch"""
    # Your code here
    criterion_r = criterions['regression']
    criterion_c = criterions['classification']
    with torch.set_grad_enabled(train):
        # for (img, measurements) in tqdm.tqdm(dataloader):
        for (img, measurements) in (dataloader):
            img = img.cuda()
            lane_dist_gt = measurements['lane_dist'].reshape(-1, 1).float().cuda()  # 0, 2.5
            route_ang_gt = measurements['route_angle'].reshape(-1, 1).float().cuda()  # -0.3 0.3
            tl_state_gt = measurements['tl_state'].reshape(-1, 1).float().cuda()
            tl_dist_gt = measurements['tl_dist'].reshape(-1, 1).float().cuda()  # -3, 45.0
            command = list(measurements['command'])

            lane_dist_pred, route_ang_pred, tl_dist_pred, tl_state_pred = model(img, command)

            loss_lanedist = criterion_r(lane_dist_pred, lane_dist_gt).mean()
            loss_routeang = criterion_r(route_ang_pred, route_ang_gt).mean()
            # loss_tldist = criterion_r(tl_dist_pred, tl_dist_gt)
            loss_tldist = (tl_state_pred * criterion_r(tl_dist_pred, tl_dist_gt)).mean()  # Masked TL Distance
            loss_tlstate = criterion_c(tl_state_pred, tl_state_gt).mean()

            loss = loss_lanedist + loss_routeang + loss_tldist + loss_tlstate
            loss_item = loss.item()
            if iteration % 8 == 0:
                wandb.log({'Lane Dist - {} {}'.format(phase, 'MAE'): loss_lanedist.item()}, step=iteration)
                wandb.log({'Route Ang - {} {}'.format(phase, 'MAE'): loss_routeang.item()}, step=iteration)
                wandb.log({'TL Dist - {} {}'.format(phase, 'Masked MAE'): loss_tldist.item()}, step=iteration)
                wandb.log({'TL State - {} {} '.format(phase, 'BCE'): loss_tlstate.item()}, step=iteration)
                wandb.log({'Total - {} {}'.format(phase, 'Loss'): loss_item}, step=iteration)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            iteration += 1
        return [loss_lanedist.item(), loss_routeang.item(), loss_tlstate.item(), loss_tldist.item()], iteration


def plot_all(train_loss, val_loss, config):
    """Visualize your plots and save them for your report."""
    # Your code here
    epoch = range(len(train_loss))
    train_loss = np.array(train_loss)
    val_loss = np.array(val_loss)
    for i, type in enumerate(['Lane Dist.', 'Route Ang.', 'TL State', 'TL Dist']):
        if type == 'TL State':
            name = 'BCE'
        else:
            name = 'MAE'
        plot_losses(train_loss[:, i], val_loss[:, i], type, name, config)


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
    plt.savefig(f'CAL Config {config} {type} Loss Curve ({loss}).png')


def main():
    config = yaml.load(open('CALTrainConfig.yml', 'r'), Loader=yaml.FullLoader)
    with wandb.init(project=f"CAL Training - Config {config['ID']}", config=config):
        config = wandb.config

        # Change these paths to the correct paths in your downloaded expert dataset
        data_root = "/home/mcokelek21/Desktop/comp523/expert_data"
        train_root = "{}/train".format(data_root)
        val_root = "{}/val".format(data_root)
        model = AffordancePredictor().cuda()
        train_dataset = ExpertDataset(train_root)
        val_dataset = ExpertDataset(val_root)

        # You can change these hyper parameters freely, and you can add more
        num_epochs = config.Epochs
        batch_size = config.BatchSize

        train_loader = DataLoader(train_dataset, batch_size=config.BatchSize, shuffle=True,
                                  drop_last=True, num_workers=8)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

        train_losses = []
        val_losses = []

        # criterions = {'regression': torch.nn.MSELoss(),
        #               'classification': torch.nn.BCELoss()}

        criterions = {'regression': torch.nn.L1Loss(reduction='none'),
                      'classification': torch.nn.BCELoss(reduction='none')}

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), config.Optim['LR'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=4, gamma=0.25, verbose=True)
        train_iter = 0
        val_iter = 0
        for epoch in range(num_epochs):
            val_loss, val_iter = train_eval(config, 'eval', model, val_loader, criterions, optimizer, val_iter)
            print('Val Epoch:', epoch, val_loss)
            val_losses.append(val_loss)
            train_loss, train_iter = train_eval(config, 'train', model, train_loader, criterions, optimizer, train_iter)
            print('Train Epoch:', epoch, train_loss)
            train_losses.append(val_loss)
            model.eval()
            scheduler.step()
        plot_all(train_losses, val_losses, str(config.ID))


if __name__ == "__main__":
    main()
