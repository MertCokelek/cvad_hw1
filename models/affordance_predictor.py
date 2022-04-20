import torch
import torch.nn as nn
import torchvision.models as models


class MLP(nn.Module):
    def __init__(self, cin, ch, cout):
        super(MLP, self).__init__()
        self.block = nn.Sequential(nn.Linear(cin, ch),
                                   nn.ReLU(),
                                   nn.Linear(ch, cout))

    def forward(self, x):
        x_flattened = x.view(x.size(0), -1)
        return self.block(x_flattened)


class AffordancePredictor(nn.Module):
    """Afforance prediction network that takes images as input"""

    def __init__(self):
        super().__init__()
        # vgg = models.vgg11_bn(pretrained=True)
        # self.feature_extractor = nn.Sequential(*list(vgg.features), nn.AdaptiveMaxPool2d(1))  # x.view(x.size(0), -1)
        resnet = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # 512, 1, 1
        # for child in self.feature_extractor.children():
        #     for param in child.parameters():
        #         param.requires_grad = False
        self.cond_affordances = nn.ModuleList([MLP(512, 256, 2) for _ in range(4)])  # lane_distance, route_angle
        self.uncond_affordances = MLP(512, 256, 2)  # traffic light state & distance

    def forward(self, img, command):
        vgg_features = self.feature_extractor(img)

        cond_affordances = torch.stack([ctrl(vgg_features) for ctrl in self.cond_affordances]).permute(1, 0, 2)
        cond_affordances = cond_affordances[:, command][:, 0]

        uncond_affordances = self.uncond_affordances(vgg_features)

        lane_distance = cond_affordances[..., 0].reshape(-1, 1)
        route_angle = cond_affordances[..., 1].reshape(-1, 1)
        traffic_light_state = uncond_affordances[..., 0].reshape(-1, 1)
        traffic_light_dist = uncond_affordances[..., 1].reshape(-1, 1)

        return lane_distance, route_angle, traffic_light_dist, torch.sigmoid(traffic_light_state)
