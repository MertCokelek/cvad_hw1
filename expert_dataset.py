from torch.utils.data import Dataset
from PIL import Image
import json
import glob
import torchvision.transforms as T
import cv2


class ExpertDataset(Dataset):
    """Dataset of RGB images, driving affordances and expert actions"""

    def __init__(self, data_root):
        self.data_root = data_root
        # Your code here
        # lim = 5000
        # if 'val' in data_root:
        #     lim = lim // 2
        lim = -1

        self.measurements = sorted(glob.glob("{}/measurements/*.json".format(data_root)))[:lim]
        self.image_names = sorted(glob.glob("{}/rgb/*.png".format(data_root)))[:lim]
        self.transform = T.Compose([T.ToTensor(),
                                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])]
                                   )

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        """Return RGB images and measurements"""
        # Your code here
        # img = Image.open(self.image_names[index]).convert('RGB')
        img = cv2.resize(cv2.imread(self.image_names[index]), (224, 224))
        img_tensor = self.transform(img)
        # img_tensor = img_tensor.permute(2, 0, 1)
        measurements = json.load(open(self.measurements[index], 'r'))

        command = measurements['command']  # .reshape(-1, 1)

        # CIL
        # speed = measurements['speed']
        # throttle = measurements['throttle']
        # brake = measurements['brake']
        # steer = measurements['steer']
        #
        # # CAL
        # lane_dist = measurements['lane_dist']
        # route_angle = measurements['route_angle']
        # tl_state = measurements['tl_state']
        # tl_dist = measurements['tl_dist']
        #
        # meas = dict()
        #
        # # CIL
        # meas['speed'] = speed
        # meas['throttle'] = throttle
        # meas['brake'] = brake
        # meas['steer'] = steer
        # meas['command'] = command

        return img_tensor, measurements
