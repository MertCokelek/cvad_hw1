import os

import cv2
import torch
import yaml
import torchvision.transforms as T
from carla_env.env import Env


class Evaluator:
    def __init__(self, env, config, modelconfig):
        self.transform = T.Compose([T.ToTensor(),
                                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])]
                                   )
        self.env = env
        self.config = config
        checkpoint_path = f"Checkpoints/Config_{modelconfig['ID']}_Epoch-{modelconfig['TestEpoch']-1}"
        print("Loading checkpoint:", checkpoint_path)
        self.agent = self.load_agent(checkpoint_path)

    def load_agent(self, checkpoint_path):
        from models.cilrs import CILRS
        # Your code here
        agent = torch.load(checkpoint_path)
        agent.cuda()
        agent.eval()

        return agent

    def generate_action(self, rgb, command, speed):
        # Your code here
        img = cv2.resize(rgb, (224, 224))
        img = self.transform(img).cuda().unsqueeze(0)
        speed_t = torch.tensor(speed).reshape(-1, 1).float().cuda()
        actions, _ = self.agent(img, speed_t, [command])

        throttle = float(actions[:, 0])
        brake = float(actions[:, 1])
        steer = float(actions[:, 2])

        # throttle = actions[:, 0]
        # brake = actions[:, 1]
        # steer = actions[:, 2]
        # throttle = float(torch.sigmoid(throttle))
        # brake = float(torch.sigmoid(brake))
        # steer = float(torch.tanh(steer))

        # if brake < 0.0005:
        #     brake = 0
        # else:
        #    brake = 0.3
        return throttle, steer, brake

    def take_step(self, state):
        rgb = state["rgb"]
        command = state["command"]
        speed = state["speed"]
        throttle, steer, brake = self.generate_action(rgb, command, speed)
        print(f'Throttle: {throttle}, Steer: {steer}, Brake: {brake}')
        # commands = ['LEFT', 'RIGHT', 'STRAIGHT', 'LANE FOLLOW']
        # print(f"Command:", commands[command])
        action = {
            "throttle": throttle,
            "brake": brake if brake > 0.0005 else 0.0,
            "steer": steer
        }
        state, reward_dict, is_terminal = self.env.step(action)
        return state, is_terminal

    def evaluate(self, num_trials=100):

        terminal_histogram = {}
        for i in range(num_trials):
            state, _, is_terminal = self.env.reset()
            for i in range(5000):
                if is_terminal:
                    break
                state, is_terminal = self.take_step(state)
            if not is_terminal:
                is_terminal = ["timeout"]
            terminal_histogram[is_terminal[0]] = (terminal_histogram.get(is_terminal[0], 0)+1)
        print("Evaluation over. Listing termination causes:")
        for key, val in terminal_histogram.items():
            print(f"{key}: {val}/100")


def main():
    with open(os.path.join("configs", "cilrs.yaml"), "r") as f:
        config = yaml.full_load(f)
    modelconfig = yaml.load(open('CILTrainConfig.yml', 'r'), Loader=yaml.FullLoader)
    with Env(config) as env:
        evaluator = Evaluator(env, config, modelconfig)
        evaluator.evaluate()


if __name__ == "__main__":
    main()
