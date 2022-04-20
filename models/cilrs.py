import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np


class CILRS(nn.Module):
    """An imitation learning agent with a resnet backbone."""

    def __init__(self):
        super().__init__()

        self.measured_speed = nn.Sequential(nn.Linear(in_features=1, out_features=128), nn.Dropout(0.5), nn.ReLU(),
                                            nn.Linear(in_features=128, out_features=128), nn.Dropout(0.5), nn.ReLU(),
                                            nn.Linear(in_features=128, out_features=128), nn.Dropout(0.5), nn.ReLU(),
                                            )
        self.speed_prediction = nn.Sequential(nn.Linear(in_features=512, out_features=256), nn.Dropout(0.5), nn.ReLU(),
                                              nn.Linear(in_features=256, out_features=256), nn.Dropout(0.5), nn.ReLU(),
                                              nn.Linear(in_features=256, out_features=1)
                                              )

        self.joint_input = nn.Sequential(nn.Linear(in_features=512 + 128, out_features=512),
                                         nn.Dropout(0.3),
                                         nn.ReLU())

        # Switch between independent control branches based on the input command
        self.control = nn.ModuleList(
            [nn.Sequential(nn.Linear(in_features=512, out_features=256), nn.Dropout(0.5), nn.LeakyReLU(),
                           nn.Linear(in_features=256, out_features=256), nn.Dropout(0.5), nn.LeakyReLU(),
                           nn.Linear(in_features=256, out_features=3)
                           # throttle (0,1), steer(-1,1), brake(0,1)
                           ) for _ in range(4)  # L,R,Lane,Straight
             ])

        # for m in self.modules():
            # if isinstance(m, nn.Conv2d):
            #     nn.init.kaiming_normal_(
            #         m.weight, mode='fan_out', nonlinearity='relu')
            # if isinstance(m, nn.Linear):
            #     nn.init.xavier_uniform_(
            #         m.weight)
            #     m.bias.data.fill_(0.01)

        resnet = models.resnet18(pretrained=True)
        self.perception = nn.Sequential(*list(resnet.children())[:-1])  # 512, 1, 1
        # for child in self.perception.children():
        #     for param in child.parameters():
                # param.requires_grad = False

    def forward(self, img, speed, command):
        P_i = self.perception(img).squeeze(2).squeeze(2)
        S_e = self.measured_speed(speed)

        joint_emb = torch.cat([P_i, S_e], dim=1)
        joint_output = self.joint_input(joint_emb)

        predicted_speed = self.speed_prediction(P_i)
        actions = torch.stack([ctrl(joint_output) for ctrl in self.control]).permute(1, 0, 2)


        actions = actions[np.arange(len(command)), np.array(command)]

        return actions, predicted_speed
