import torch
import torch.nn as nn
import numpy as np

class ActorNetwork(nn.Module):
    def __init__(self, map_dim, robot_dim, task_dim, global_dim, action_dim):
        super(ActorNetwork, self).__init__()
        self.map_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(map_dim), 128),
            nn.ReLU()
        )
        self.robot_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(robot_dim), 128),
            nn.ReLU()
        )
        self.task_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(task_dim), 128),
            nn.ReLU()
        )
        self.global_fc = nn.Sequential(
            nn.Linear(global_dim[0], 64),
            nn.ReLU()
        )
        self.concat_fc = nn.Sequential(
            nn.Linear(128 * 3 + 64, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, map_state, robot_state, task_state, global_state):
        x1 = self.map_fc(map_state)
        x2 = self.robot_fc(robot_state)
        x3 = self.task_fc(task_state)
        x4 = self.global_fc(global_state)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return self.concat_fc(x)

class PPOAgent:
    def __init__(self, model_path='policy.pth', width=25, height=25, num_robots=5, action_dim=12):
        self.action_dim = action_dim
        self.WIDTH = width
        self.HEIGHT = height
        self.NUM_ROBOTS = num_robots

        # 假设你的输入维度如下，可根据你模型实际情况调整
        self.actor = ActorNetwork(
            map_dim=(width, height),
            robot_dim=(num_robots, 4),   # 每个机器人 4 个状态特征？
            task_dim=(10, 5),            # 10 个任务，每个任务 5 个特征？
            global_dim=(10,),            # 全局状态向量维度
            action_dim=action_dim
        )
        self.actor.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.actor.eval()

    def choose_action(self, observation):
        """
        给定当前环境状态，预测每个机器人下一步动作。
        输入：
            observation: 字典，包含grid、robots、tasks、global状态
        输出：
            actions: 长度为NUM_ROBOTS的动作列表，每个动作是0-11的整数
        """
        # 转换为 torch tensor，并增加 batch 维
        map_state = torch.tensor(np.expand_dims(observation['grid'], axis=0), dtype=torch.float32)
        robot_state = torch.tensor(np.expand_dims(observation['robots'], axis=0), dtype=torch.float32)
        task_state = torch.tensor(np.expand_dims(observation['tasks'], axis=0), dtype=torch.float32)
        global_state = torch.tensor(np.expand_dims(observation['global'], axis=0), dtype=torch.float32)

        with torch.no_grad():
            action_probs = self.actor(map_state, robot_state, task_state, global_state)

        # 每个机器人根据其 action 分布 sample
        actions = []
        probs = action_probs.squeeze(0).cpu().numpy()
        for i in range(self.NUM_ROBOTS):
            action = np.random.choice(self.action_dim, p=probs)  # 可替换为每个机器人独立输出
            actions.append(action)

        return actions
