import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input, Concatenate, LSTM
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import gym
from gym import spaces
from environment import (
    WIDTH, HEIGHT, NUM_ROBOTS, MAX_BATTERY, MOVE_COST,LOW_BATTERY_THRESHOLD,
    CHARGING_STATION, MAX_TIME, TOTAL_TASKS, IDLE_TIMEOUT,
    Robot, generate_map, generate_tasks, a_star, path_cost
)


# =================== 强化学习环境 ===================
class RobotSchedulingEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(RobotSchedulingEnv, self).__init__()
        self.render_mode = render_mode
        self.grid = generate_map()
        self.chargers = [(i, j) for i in range(HEIGHT) for j in range(WIDTH)
                         if self.grid[i][j] == CHARGING_STATION]

        # 每个机器人可以选择的动作：分配给10个任务中的一个，去充电，或保持空闲
        self.action_space = spaces.MultiDiscrete([12] * NUM_ROBOTS)  # 0-9是任务, 10是充电, 11是空闲

        # 观察空间
        self.observation_space = spaces.Dict({
            'grid': spaces.Box(low=0, high=3, shape=(HEIGHT, WIDTH), dtype=np.int32),
            'robots': spaces.Box(low=0, high=1, shape=(NUM_ROBOTS, 6), dtype=np.float32),
            'tasks': spaces.Box(low=0, high=1, shape=(10, 5), dtype=np.float32),
            'global': spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        })

        self.reset()

    def reset(self, *, seed=None, options=None):
        self.time = 0
        self.tasks = generate_tasks(self.grid, self.chargers)
        self.robots = [Robot(i, random.choice(self.chargers), self.chargers)
                       for i in range(NUM_ROBOTS)]
        self.episode_rewards = 0
        self.task_wait_times = []
        self.completed_tasks = 0
        self.energy_used = 0
        self.initial_total_energy = sum(robot.battery for robot in self.robots)

        observation = self._get_state()
        return observation, {}

    def _get_state(self):
        # 将环境状态编码为神经网络的输入
        # 1. 栅格地图状态
        grid_state = np.copy(self.grid)

        # 2. 机器人状态
        robot_states = []
        for robot in self.robots:
            robot_state = [
                robot.pos[0] / HEIGHT,  # 归一化位置
                robot.pos[1] / WIDTH,
                robot.battery / MAX_BATTERY,  # 归一化电量
                1 if robot.task else 0,  # 是否有任务
                1 if robot.state == 'charging' else 0,  # 是否在充电
                robot.idle_counter / IDLE_TIMEOUT if hasattr(robot, 'idle_counter') else 0  # 归一化空闲时间
            ]
            robot_states.append(robot_state)

        # 3. 任务状态
        active_tasks = [t for t in self.tasks
                        if t['arrival_time'] <= self.time and not t['served']]
        task_states = []
        for i in range(min(10, len(active_tasks))):  # 最多考虑10个活跃任务
            task = active_tasks[i]
            task_state = [
                task['location'][0] / HEIGHT,  # 归一化位置
                task['location'][1] / WIDTH,
                task['initial_energy'] / 100,  # 归一化初始能量
                task['required_energy'] / 100,  # 归一化所需能量
                (self.time - task['arrival_time']) / MAX_TIME,  # 归一化等待时间
            ]
            task_states.append(task_state)

        # 如果活跃任务少于10个，用零向量填充
        while len(task_states) < 10:
            task_states.append([0, 0, 0, 0, 0])

        # 4. 全局状态
        global_state = [
            self.time / MAX_TIME,  # 归一化时间
            len(active_tasks) / TOTAL_TASKS,  # 归一化活跃任务数
            self.completed_tasks / TOTAL_TASKS  # 归一化完成任务数
        ]

        return {
            'grid': grid_state.astype(np.int32),
            'robots': np.array(robot_states, dtype=np.float32),
            'tasks': np.array(task_states, dtype=np.float32),
            'global': np.array(global_state, dtype=np.float32)
        }

    def step(self, actions):
        """
        执行动作并返回新状态、奖励、是否结束和额外信息
        actions: 每个机器人的动作 0-9表示分配给对应任务, 10表示去充电, 11表示保持空闲
        """
        prev_completed = self.completed_tasks
        prev_energy_levels = [robot.battery for robot in self.robots]
        prev_positions = [robot.pos for robot in self.robots]

        # 应用动作
        active_tasks = [t for t in self.tasks
                        if t['arrival_time'] <= self.time and not t['served'] and t['assigned_to'] is None]

        for robot_id, action in enumerate(actions):
            robot = self.robots[robot_id]

            # 如果机器人已经有任务或正在充电，跳过
            if robot.task is not None or robot.state == 'charging':
                continue

            if action == 10:  # 特殊动作：去充电
                robot.return_to_charge(self.grid)
            elif action == 11:  # 特殊动作：保持空闲
                pass
            elif action < len(active_tasks):  # 分配任务
                task = active_tasks[action]
                # 检查任务到充电站的可达性
                to_task_path = a_star(self.grid, robot.pos, task['location'])
                if to_task_path:
                    # 检查电量是否足够完成任务并到达最近的充电站
                    to_task_cost = len(to_task_path) * MOVE_COST
                    nearest_charger = min(self.chargers, key=lambda c: path_cost(self.grid, task['location'], c))
                    to_charger_cost = path_cost(self.grid, task['location'], nearest_charger) * MOVE_COST
                    energy_for_task = task['required_energy'] - task['initial_energy']

                    if robot.battery >= (to_task_cost + to_charger_cost + energy_for_task):
                        robot.assign(task, to_task_path)
                        active_tasks.remove(task)

        # 更新机器人状态
        for robot in self.robots:
            old_battery = robot.battery
            robot.step(self.grid, self.time)

            # 计算本步骤消耗的能量（不包括充电获得的能量）
            if robot.battery < old_battery:
                self.energy_used += (old_battery - robot.battery)

        # 更新计数器
        self.time += 1

        # 计算完成的任务数
        self.completed_tasks = sum(1 for t in self.tasks if t['served'])

        # 计算新完成任务的等待时间
        newly_completed = [t for t in self.tasks
                           if t['served'] and t not in self.task_wait_times]
        for task in newly_completed:
            if task['start_time'] is not None:
                wait_time = task['start_time'] - task['arrival_time']
                self.task_wait_times.append(task)

        # 获取新状态
        new_state = self._get_state()

        # 计算奖励
        reward = self._compute_reward(prev_completed, prev_energy_levels, prev_positions)
        self.episode_rewards += reward

        # 检查是否结束
        done = self.time >= 300 or self.completed_tasks == len(self.tasks)

        # 构建信息字典
        info = {
            'completed_tasks': self.completed_tasks,
            'total_tasks': len(self.tasks),
            'avg_wait_time': np.mean(
                [t['start_time'] - t['arrival_time'] for t in self.task_wait_times]) if self.task_wait_times else 0,
            'total_reward': self.episode_rewards,
            'energy_efficiency': 1.0 - (
                        self.energy_used / self.initial_total_energy) if self.initial_total_energy > 0 else 0
        }

        # 渲染
        if self.render_mode == 'human':
            self.render()

        return new_state, reward, done, info

    def _compute_reward(self, prev_completed, prev_energy_levels, prev_positions):
        """计算奖励函数 - 优化版本，更强调任务完成"""
        reward = 0

        # 奖励1: 完成任务奖励（最重要，大幅提高）
        new_completed = self.completed_tasks - prev_completed
        if new_completed > 0:
            reward += 100 * new_completed  # 将任务完成奖励从100提高到300

        # 奖励3: 任务分配奖励 - 大幅提高
        active_tasks = [t for t in self.tasks
                        if t['arrival_time'] <= self.time and not t['served'] and t['assigned_to'] is None]
        for robot in self.robots:
            # 如果机器人刚刚接受了新任务
            if robot.task and robot.task['assigned_to'] == robot.id and robot.task not in active_tasks:
                # 给予高额奖励鼓励任务分配
                reward += 20

        # 奖励5: 能源效率奖励 - 降低惩罚系数
        energy_used = 0
        for i, robot in enumerate(self.robots):
            energy_diff = prev_energy_levels[i] - robot.battery
            if energy_diff > 0:  # 只考虑消耗的能量
                energy_used += energy_diff

        # 降低能源消耗惩罚，使其不会阻碍任务完成
        reward -= 0.2 * energy_used  # 从0.5降低到0.2

        # 奖励6: 低电量时去充电的奖励（略微提高）
        for i, robot in enumerate(self.robots):
            if robot.battery < LOW_BATTERY_THRESHOLD * 1.5:  # 接近低电量阈值
                if robot.state == 'to_charge' or robot.state == 'charging':
                    reward += 5  # 从2提高到5
                elif robot.battery < LOW_BATTERY_THRESHOLD and robot.state != 'to_charge':
                    # 严重低电量但未去充电，给予惩罚
                    reward -= 20

        # 奖励7: 平均等待时间惩罚（降低权重）
        if self.task_wait_times:
            avg_wait = np.mean([t['start_time'] - t['arrival_time'] for t in self.task_wait_times])
            reward -= 0.5 * avg_wait  # 从0.1降低到0.05

        # 奖励8: 死锁惩罚（保持严重）
        for robot in self.robots:
            if robot.state == 'dead':
                reward -= 100  # 从50提高到100

        # 奖励9: 空闲惩罚（新增）
        for robot in self.robots:
            if robot.state == 'idle' and not robot.task:
                # 空闲时间越长，惩罚越大
                idle_time = robot.idle_counter if hasattr(robot, 'idle_counter') else 0
                reward -= 0.5 * idle_time  # 空闲惩罚

        # 奖励10: 全局进度奖励（新增）
        completion_rate = self.completed_tasks / len(self.tasks)
        reward += 300 * (completion_rate - 0.8)  # 根据总体完成率给予额外奖励

        # 奖励11: 长期任务优先（新增）
        waiting_tasks = [t for t in self.tasks
                         if t['arrival_time'] <= self.time and not t['served'] and t['assigned_to'] is None]
        if waiting_tasks:
            # 如果有等待很久的任务仍未分配，给予惩罚
            longest_wait = max([(self.time - t['arrival_time']) for t in waiting_tasks], default=0)
            if longest_wait > 10:  # 等待超过10个时间单位
                reward -= 0.2 * longest_wait  # 惩罚长时间等待的任务未被分配

        return reward


# =================== PPO Agent ===================
class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return np.array(self.states), np.array(self.actions), np.array(self.probs), \
            np.array(self.vals), np.array(self.rewards), np.array(self.dones), batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []


# Actor network for PPO
# Actor network for PPO
def build_actor_network(state_shape, action_dim):
    # 构建Actor网络
    # 输入1: 地图状态
    map_input = Input(shape=(HEIGHT, WIDTH))
    # 使用Reshape层代替tf.reshape
    map_reshape = keras.layers.Reshape((HEIGHT, WIDTH, 1))(map_input)
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(map_reshape)
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    map_flat = Flatten()(conv2)

    # 输入2: 机器人状态
    robot_input = Input(shape=(NUM_ROBOTS, 6))
    robot_flat = Flatten()(robot_input)

    # 输入3: 任务状态
    task_input = Input(shape=(10, 5))  # 最多10个活跃任务
    task_flat = Flatten()(task_input)

    # 输入4: 全局状态
    global_input = Input(shape=(3,))

    # 合并所有输入
    merged = Concatenate()([map_flat, robot_flat, task_flat, global_input])

    # 全连接层
    dense1 = Dense(256, activation='relu')(merged)
    dense2 = Dense(128, activation='relu')(dense1)

    # 为每个机器人输出独立的动作概率分布
    action_outputs = []
    for _ in range(NUM_ROBOTS):
        # 每个机器人可以选择12个动作：10个任务分配，充电，空闲
        action_output = Dense(action_dim, activation='softmax')(dense2)
        action_outputs.append(action_output)

    # 构建模型
    actor = Model(
        inputs=[map_input, robot_input, task_input, global_input],
        outputs=action_outputs
    )

    return actor


# Critic network for PPO
def build_critic_network(state_shape):
    # 构建Critic网络
    # 输入1: 地图状态
    map_input = Input(shape=(HEIGHT, WIDTH))
    # 使用Reshape层代替tf.reshape
    map_reshape = keras.layers.Reshape((HEIGHT, WIDTH, 1))(map_input)
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(map_reshape)
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    map_flat = Flatten()(conv2)

    # 输入2: 机器人状态
    robot_input = Input(shape=(NUM_ROBOTS, 6))
    robot_flat = Flatten()(robot_input)

    # 输入3: 任务状态
    task_input = Input(shape=(10, 5))  # 最多10个活跃任务
    task_flat = Flatten()(task_input)

    # 输入4: 全局状态
    global_input = Input(shape=(3,))

    # 合并所有输入
    merged = Concatenate()([map_flat, robot_flat, task_flat, global_input])

    # 全连接层
    dense1 = Dense(256, activation='relu')(merged)
    dense2 = Dense(128, activation='relu')(dense1)

    # 输出价值估计
    value_output = Dense(1, activation=None)(dense2)

    # 构建模型
    critic = Model(
        inputs=[map_input, robot_input, task_input, global_input],
        outputs=value_output
    )

    return critic


class PPOAgent:
    def __init__(self, n_actions=12, batch_size=64, alpha=0.0003, gamma=0.99,
                 gae_lambda=0.95, policy_clip=0.2, n_epochs=30, entropy_coef=0.01):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.action_dim = n_actions

        self.actor = build_actor_network((HEIGHT, WIDTH), n_actions)
        self.critic = build_critic_network((HEIGHT, WIDTH))
        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic.compile(optimizer=Adam(learning_rate=alpha))

        self.memory = PPOMemory(batch_size)

    def store_transition(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self, actor_path='ppo_actor.h5', critic_path='ppo_critic.h5'):
        print('... 保存模型 ...')
        self.actor.save(actor_path)
        self.critic.save(critic_path)

    def load_models(self, actor_path='ppo_actor.h5', critic_path='ppo_critic.h5'):
        print('... 加载模型 ...')
        self.actor = keras.models.load_model(actor_path)
        self.critic = keras.models.load_model(critic_path)

    def choose_action(self, observation):
        # 准备输入
        map_state = np.expand_dims(observation['grid'], axis=0)
        robot_state = np.expand_dims(observation['robots'], axis=0)
        task_state = np.expand_dims(observation['tasks'], axis=0)
        global_state = np.expand_dims(observation['global'], axis=0)

        # 预测动作概率分布和值函数
        action_probs = self.actor.predict([map_state, robot_state, task_state, global_state], verbose=0)
        value = self.critic.predict([map_state, robot_state, task_state, global_state], verbose=0)

        # 从每个机器人的动作概率分布中采样动作
        actions = []
        all_probs = []

        for robot_probs in action_probs:
            # 提取概率分布
            robot_probs = robot_probs[0]
            # 根据概率分布采样动作
            action = np.random.choice(self.action_dim, p=robot_probs)
            actions.append(action)
            all_probs.append(robot_probs[action])

        return actions, all_probs, value[0][0]

    def learn(self):
        for _ in range(self.n_epochs):
            # 生成训练批次
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            # 计算优势函数
            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] * (1 - int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t

            # 批次训练
            for batch in batches:
                with tf.GradientTape(persistent=True) as tape:
                    # 准备输入
                    states_batch = [
                        np.array([state_arr[i]['grid'] for i in batch]),
                        np.array([state_arr[i]['robots'] for i in batch]),
                        np.array([state_arr[i]['tasks'] for i in batch]),
                        np.array([state_arr[i]['global'] for i in batch])
                    ]

                    # Actor网络输出
                    action_probs_batch = self.actor(states_batch)

                    # Critic网络输出
                    critic_value_batch = self.critic(states_batch)
                    critic_value_batch = tf.squeeze(critic_value_batch, axis=1)

                    # 计算每个机器人的新动作概率
                    new_probs = []
                    for r_idx in range(NUM_ROBOTS):
                        # 提取该机器人的动作概率
                        r_probs = action_probs_batch[r_idx]
                        # 提取该机器人在每个样本中的动作
                        r_actions = np.array([action_arr[i][r_idx] for i in batch])
                        # 提取对应概率
                        r_new_probs = tf.reduce_sum(
                            tf.one_hot(r_actions, self.action_dim) * r_probs,
                            axis=1
                        )
                        new_probs.append(r_new_probs)

                    # 计算每个机器人的旧动作概率
                    old_probs = []
                    for r_idx in range(NUM_ROBOTS):
                        r_old_probs = np.array([old_prob_arr[i][r_idx] for i in batch])
                        old_probs.append(r_old_probs)

                    # 合并所有机器人的概率比率
                    prob_ratio = 1.0
                    for r_idx in range(NUM_ROBOTS):
                        prob_ratio *= new_probs[r_idx] / (old_probs[r_idx] + 1e-10)

                    # 提取该批次的优势
                    advantages = advantage[batch]

                    # PPO更新
                    weighted_probs = prob_ratio * advantages
                    weighted_clipped_probs = tf.clip_by_value(
                        prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip
                    ) * advantages

                    # Actor loss
                    actor_loss = -tf.reduce_mean(tf.minimum(weighted_probs, weighted_clipped_probs))

                    # Critic loss
                    returns = advantages + values[batch]
                    critic_loss = tf.reduce_mean(tf.square(returns - critic_value_batch))

                # 更新Actor和Critic
                actor_params = self.actor.trainable_variables
                critic_params = self.critic.trainable_variables

                actor_grads = tape.gradient(actor_loss, actor_params)
                critic_grads = tape.gradient(critic_loss, critic_params)

                self.actor.optimizer.apply_gradients(zip(actor_grads, actor_params))
                self.critic.optimizer.apply_gradients(zip(critic_grads, critic_params))

        # 清除记忆
        self.memory.clear_memory()


# =================== 训练和评估 ===================
def train_ppo(env, agent, n_episodes=1000, n_steps_per_episode=300):
    best_score = -np.inf
    score_history = []
    completed_tasks_history = []
    energy_efficiency_history = []

    for episode in range(n_episodes):
        observation = env.reset()
        done = False
        score = 0
        step = 0

        while not done and step < n_steps_per_episode:
            # 选择动作
            action, prob, val = agent.choose_action(observation)

            # 执行动作
            new_observation, reward, done, info = env.step(action)
            step += 1
            score += reward

            # 存储转换
            agent.store_transition(observation, action, prob, val, reward, done)

            # 更新观察
            observation = new_observation

            # 如果内存足够或者回合结束，开始学习
            if step % agent.memory.batch_size == 0 or done:
                agent.learn()

        # 记录分数和任务完成率
        score_history.append(score)
        completed_tasks_history.append(info['completed_tasks'] / info['total_tasks'])
        energy_efficiency_history.append(info['energy_efficiency'])

        # 打印进度
        avg_score = np.mean(score_history[-100:])
        avg_completed = np.mean(completed_tasks_history[-100:])
        avg_energy_efficiency = np.mean(energy_efficiency_history[-100:])

        print(
            f'Episode: {episode}, Score: {score:.2f}, Avg Score: {avg_score:.2f}, Tasks: {avg_completed:.2f}, Energy Efficiency: {avg_energy_efficiency:.2f}')

        # 保存最佳模型
        if avg_score > best_score and episode > 20:
            best_score = avg_score
            agent.save_models()

    # 绘制学习曲线
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(score_history)
    plt.title('Reward History')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')

    plt.subplot(1, 3, 2)
    plt.plot(completed_tasks_history)
    plt.title('Task Completion Rate')
    plt.xlabel('Episode')
    plt.ylabel('Completion Rate')

    plt.subplot(1, 3, 3)
    plt.plot(energy_efficiency_history)
    plt.title('Energy Efficiency')
    plt.xlabel('Episode')
    plt.ylabel('Efficiency')

    plt.tight_layout()
    plt.savefig('learning_curves.png')
    plt.show()

    return score_history, completed_tasks_history, energy_efficiency_history


# 评估函数
def evaluate_agent(env, agent, n_episodes=10):
    completion_rates = []
    avg_rewards = []
    avg_wait_times = []
    energy_efficiencies = []

    for _ in range(n_episodes):
        observation = env.reset()
        done = False
        total_reward = 0
        step = 0

        while not done and step < MAX_TIME:
            # 选择动作
            action, _, _ = agent.choose_action(observation)

            # 执行动作
            observation, reward, done, info = env.step(action)
            total_reward += reward
            step += 1

        # 收集统计数据
        completion_rate = info['completed_tasks'] / info['total_tasks']
        completion_rates.append(completion_rate)
        avg_rewards.append(total_reward)
        avg_wait_times.append(info['avg_wait_time'])
        energy_efficiencies.append(info['energy_efficiency'])

    # 计算平均值
    avg_completion_rate = np.mean(completion_rates)
    avg_reward = np.mean(avg_rewards)
    avg_wait_time = np.mean(avg_wait_times)
    avg_energy_efficiency = np.mean(energy_efficiencies)

    print("\n==== 评估结果 ====")
    print(f"平均任务完成率: {avg_completion_rate:.2f}")
    print(f"平均总奖励: {avg_reward:.2f}")
    print(f"平均等待时间: {avg_wait_time:.2f}")
    print(f"平均能源效率: {avg_energy_efficiency:.2f}")

    return {
        'completion_rate': avg_completion_rate,
        'avg_reward': avg_reward,
        'avg_wait_time': avg_wait_time,
        'energy_efficiency': avg_energy_efficiency
    }


if __name__ == "__main__":
    # 设置随机种子
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)

    # 创建环境
    env = RobotSchedulingEnv()

    # 创建优化后的PPO代理
    n_actions = 12  # 0-9是任务, 10是充电, 11是空闲
    agent = PPOAgent(
        n_actions=n_actions,
        batch_size=128,  # 从64增加到128，增加批量大小
        alpha=0.0003,  # 学习率保持适中
        gamma=0.995,  # 提高未来奖励权重，从0.99增加到0.995
        gae_lambda=0.98,  # 优化优势估计
        policy_clip=0.2,  # 保持适当的裁剪范围
        n_epochs=30  # 增加每批次训练轮数，从10增加到15
    )

    # 训练
    print("开始训练...")
    train_ppo(env, agent, n_episodes=500)

    # 评估
    print("\n开始评估...")
    eval_results = evaluate_agent(env, agent)

    # 保存评估结果
    with open('evaluation_results.txt', 'w') as f:
        f.write(f"平均任务完成率: {eval_results['completion_rate']:.2f}\n")
        f.write(f"平均总奖励: {eval_results['avg_reward']:.2f}\n")
        f.write(f"平均等待时间: {eval_results['avg_wait_time']:.2f}\n")
        f.write(f"平均能源效率: {eval_results['energy_efficiency']:.2f}\n")