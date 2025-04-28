import numpy as np
from tensorflow import keras

class PPOAgent:
    def __init__(self, actor_path='ppo_actor.h5', critic_path='ppo_critic.h5', width=25, height=25, num_robots=5):
        # 只加载训练好的Actor模型（Critic可以加载但不一定用）
        self.actor = keras.models.load_model(actor_path)
        self.critic = keras.models.load_model(critic_path)
        self.action_dim = 12  # 动作空间（10个任务+充电+空闲）

        # 动态记录环境尺寸
        self.WIDTH = width
        self.HEIGHT = height
        self.NUM_ROBOTS = num_robots

    def choose_action(self, observation):
        """
        给定当前环境状态，预测每个机器人下一步动作。
        输入：
            observation: 字典，包含grid、robots、tasks、global状态
        输出：
            actions: 长度为NUM_ROBOTS的动作列表，每个动作是0-11的整数
        """
        # 处理输入，加维让输入符合神经网络期望的Batch格式
        map_state = np.expand_dims(observation['grid'], axis=0)
        robot_state = np.expand_dims(observation['robots'], axis=0)
        task_state = np.expand_dims(observation['tasks'], axis=0)
        global_state = np.expand_dims(observation['global'], axis=0)

        # 预测动作概率分布
        action_probs = self.actor.predict([map_state, robot_state, task_state, global_state], verbose=0)

        actions = []
        for robot_probs in action_probs:
            robot_probs = robot_probs[0]
            action = np.random.choice(self.action_dim, p=robot_probs)
            actions.append(action)

        return actions
