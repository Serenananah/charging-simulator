# train_ppo.py
# 修改后的 PPO 训练脚本，参数与 app.py 保持一致，支持不同规模训练。粘贴在一行
"""可以自定义要求，以下三种只是举个例子
# 小规模 + 聚簇车位 + 泊松到达
python train_ppo.py  --scale small  --distribution clustered  --arrival poisson

# 中规模 + 均匀车位 + 均匀到达
python train_ppo.py  --scale medium  --distribution uniform  --arrival uniform

# 大规模 + 混合车位 + 正态到达
python train_ppo.py  --scale large  --distribution clustered  --arrival normal
"""
import sys
import os
import shutil
import argparse
import gym
import numpy as np
import random
import torch
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from collections import deque


# 清理日志目录
LOG_DIR = "ppo_logs"
if os.path.exists(LOG_DIR):
    shutil.rmtree(LOG_DIR) if os.path.isdir(LOG_DIR) else os.remove(LOG_DIR)
os.makedirs(LOG_DIR, exist_ok=True)

# 导入环境组件
from environment import (
    generate_map,
    generate_tasks,
    Robot,
    MAX_BATTERY,
    MOVE_COST,
    MAX_TIME,
    CHARGING_STATION,
    LOW_BATTERY_THRESHOLD,
    IDLE_TIMEOUT,
    a_star,
    path_cost,
    is_feasible,
    CHARGE_TRANSFER
)

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
class RobotSchedulingEnv(gym.Env):
    """
    根据 app.py 中的参数逻辑构造环境：
    - scale: small/medium/large
    - parking_distribution: uniform/clustered/mixed
    - arrival_mode: poisson/uniform/normal
    """
    def __init__(self, scale="medium", parking_distribution="uniform", arrival_mode="poisson"):
        super().__init__()
        # 从 app.py 同步的参数配置
        densities = {
            "small":  dict(w=25, h=25, task_density=0.07, robot_density=0.020, group_density=0.012),
            "medium": dict(w=35, h=35, task_density=0.06, robot_density=0.017, group_density=0.011),
            "large":  dict(w=40, h=40, task_density=0.05, robot_density=0.015, group_density=0.010),
        }
        if scale not in densities:
            raise ValueError("Invalid scale (use 'small', 'medium', 'large')")
        cfg = densities[scale]
        w, h = cfg['w'], cfg['h']
        area = w * h
        total_tasks = round(area * cfg['task_density'])
        num_robots  = round(area * cfg['robot_density'])
        parking_groups = round(area * cfg['group_density'])
        num_obstacles  = round(area * 0.15 / 9)
        num_chargers   = max(num_robots, int(num_robots * 0.7))

        # 保存配置
        self.WIDTH   = w
        self.HEIGHT  = h
        self.TOTAL_TASKS = total_tasks
        self.NUM_ROBOTS  = num_robots
        self.NUM_PARKING_GROUPS = parking_groups
        self.NUM_OBSTACLES      = num_obstacles
        self.NUM_CHARGERS       = num_chargers
        self.parking_distribution = parking_distribution
        self.arrival_mode = arrival_mode

        # 定义 Gym 空间
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.NUM_ROBOTS, 3), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.TOTAL_TASKS + 1)

        # 内部状态
        self.timestep = 0
        self.prev_batteries = []
        self.prev_positions = []

    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)

    def reset(self):
        # 生成地图/充电桩
        self.grid = generate_map(
            self.WIDTH, self.HEIGHT,
            self.NUM_CHARGERS,
            self.NUM_PARKING_GROUPS,
            self.NUM_OBSTACLES,
            parking_distribution=self.parking_distribution
        )
        # 收集充电桩位置
        self.chargers = [
            (i, j)
            for i in range(self.HEIGHT)
            for j in range(self.WIDTH)
            if self.grid[i][j] == CHARGING_STATION
        ]
        # 生成任务列表
        self.tasks = generate_tasks(
            self.grid, self.chargers,
            self.WIDTH, self.HEIGHT,
            self.TOTAL_TASKS,
            MAX_TIME,
            mode=self.arrival_mode
        )
        # 重新定义动作空间（任务可能因可达性减少）
        self.action_space = spaces.Discrete(len(self.tasks) + 1)
        # 初始化机器人
        self.robots = [
            Robot(i, self.chargers[i % len(self.chargers)], self.chargers)
            for i in range(self.NUM_ROBOTS)
        ]
        self.timestep = 0
        self.prev_batteries = [r.battery for r in self.robots]
        self.prev_positions = [tuple(r.pos) for r in self.robots]
        return self._get_obs()

    def _get_obs(self):
        return np.array([
            [r.battery / MAX_BATTERY, r.pos[0] / self.HEIGHT, r.pos[1] / self.WIDTH]
            for r in self.robots
        ], dtype=np.float32)
    
##新加入
    def _rebalance_robots_by_density(self):
        # 1. 筛选空闲、能量充足的机器人
        active_robots = [
            r for r in self.robots
            if r.state == 'idle' and r.task is None and r.battery > LOW_BATTERY_THRESHOLD
        ]
        if not active_robots:
            return

        # 2. 统计机器人密度（按网格划分区域）
        region_size = 5  # 每 5x5 个格子为一个区域，可调
        region_counts = {}

        for r in self.robots:
            x, y = r.pos
            region = (x // region_size, y // region_size)
            region_counts[region] = region_counts.get(region, 0) + 1

        # 3. 找出密度最高 & 最低区域
        if not region_counts:
            return

        max_region = max(region_counts.items(), key=lambda kv: kv[1])[0]
        min_region = min(region_counts.items(), key=lambda kv: kv[1])[0]

        if max_region == min_region:
            return  # 没有差异，不做迁移

        # 4. 找出高密度区中的机器人，向低密度区中央前进
        for r in active_robots:
            rx, ry = r.pos
            r_region = (rx // region_size, ry // region_size)
            if r_region == max_region:
                # 将机器人移向低密度区中心点
                target_x = min_region[0] * region_size + region_size // 2
                target_y = min_region[1] * region_size + region_size // 2
                path = a_star( self.grid, r.pos, (target_x, target_y),self.HEIGHT, self.WIDTH,)
                if path:
                    r.task = None
                    r.path = deque(path)
                    r.state = 'to_balance'



    def step(self, action):
        # 记录上一步完成的任务数、电量和位置，用于奖励计算
        prev_done = sum(t['served'] for t in self.tasks)
        self.prev_batteries = [r.battery for r in self.robots]
        self.prev_positions = [tuple(r.pos) for r in self.robots]

        # —— 任务可行性检查 —— 
        for r in self.robots:
            if r.task is not None:
                feasible = is_feasible(
                    r, r.task, self.grid, self.chargers,
                    self.HEIGHT, self.WIDTH
                )
                if not feasible:
                    r.task['assigned_to'] = None
                    r.task = None
                    r.return_to_charge(self.grid, self.HEIGHT, self.WIDTH)

        # —— 低电量空闲机器人主动返航 —— 
        for r in self.robots:
            if r.state == 'idle' and r.battery < LOW_BATTERY_THRESHOLD:
                if r.task is not None:
                    r.task['assigned_to'] = None
                    r.task = None
                r.return_to_charge(self.grid, self.HEIGHT, self.WIDTH)

        # —— 执行动作 —— 
        if action == len(self.tasks):
            # 策略1：最低电量机器人返航
            lowest = min(self.robots, key=lambda r: r.battery)
            if lowest.task is not None:
                lowest.task['assigned_to'] = None
                lowest.task = None
            lowest.return_to_charge(self.grid, self.HEIGHT, self.WIDTH)

        elif action == len(self.tasks) + 1:
            # 策略2：所有空闲且低电量机器人返航
            for r in self.robots:
                if r.state == 'idle' and r.battery < LOW_BATTERY_THRESHOLD:
                    if r.task is not None:
                        r.task['assigned_to'] = None
                        r.task = None
                    r.return_to_charge(self.grid, self.HEIGHT, self.WIDTH)

        elif action == len(self.tasks) + 2:
            # 策略3：空闲高密度机器人迁移到低密度区
            active = [
                r for r in self.robots
                if r.state == 'idle' and r.task is None and r.battery > LOW_BATTERY_THRESHOLD
            ]
            region_size = 5
            counts = {}
            for r in self.robots:
                rx, ry = r.pos
                region = (rx // region_size, ry // region_size)
                counts[region] = counts.get(region, 0) + 1
            if counts:
                max_reg = max(counts, key=counts.get)
                min_reg = min(counts, key=counts.get)
                if max_reg != min_reg:
                    target = (min_reg[0] * region_size + region_size // 2,
                            min_reg[1] * region_size + region_size // 2)
                    for r in active:
                        if (r.pos[0] // region_size, r.pos[1] // region_size) == max_reg:
                            path = a_star(self.grid, r.pos, target, self.HEIGHT, self.WIDTH)
                            if path:
                                r.task = None
                                r.path = deque(path)
                                r.state = 'to_balance'

        elif action < len(self.tasks):
            # 策略0：按任务列表顺序依次尝试分配
            for task in self.tasks:
                if (not task['served']
                    and not task['expired']
                    and task['arrival_time'] <= self.timestep):

                    idle = [
                        r for r in self.robots
                        if r.state == 'idle' and r.task is None
                    ]
                    if not idle:
                        continue

                    idle.sort(
                        key=lambda r: path_cost(
                            self.grid, r.pos, task['location'],
                            self.HEIGHT, self.WIDTH
                        )
                    )

                    for r in idle:
                        path = a_star(
                            self.grid, r.pos, task['location'],
                            self.HEIGHT, self.WIDTH
                        )
                        if path:
                            r.assign(task, path)
                            break

        # —— 所有机器人执行一步 —— 
        for r in self.robots:
            r.step(self.grid, self.timestep, self.HEIGHT, self.WIDTH)

        self.timestep += 1
        obs = self._get_obs()
        reward = self._compute_reward(prev_done, self.prev_batteries, self.prev_positions)
        done = (self.timestep >= MAX_TIME) or (sum(t['served'] for t in self.tasks) == self.TOTAL_TASKS)
        return obs, reward, done, {}


    def _compute_reward(self, prev_done, prev_bats, prev_pos):
        reward = 0.0
        cur_done = sum(1 for t in self.tasks if t['served'])
        delta = cur_done - prev_done
        STEP_PENALTY = 1.0
        reward = -STEP_PENALTY
        # —— 完成任务 —— 
        reward += 1200 * delta



        # —— 低电量惩罚 —— 
        for r in self.robots:
            if r.battery < MAX_BATTERY*0.1 and r.state not in ('charging','returning_idle'):
                reward -= 30
            if r.battery<=0:
                reward -= 200

        # —— 能耗惩罚 —— 
        used = sum(max(0, pb-r.battery) for pb,r in zip(prev_bats, self.robots))
        reward -= 0.05 * used
        # —— 等待任务惩罚 —— 
        waits = [self.timestep - t['arrival_time']
                for t in self.tasks
                if not t['served']
                and t['assigned_to'] is None
                and t['arrival_time'] <= self.timestep]
        if waits:
            reward -= 0.2 * (sum(waits)/len(waits))

        # —— 长时间等待额外惩罚 —— 
        MAX_WAIT = 50  # 超过 20 步就开始罚
        OVER_PENALTY = 1.0  # 每超出一步，额外扣 2 分
        for t in self.tasks:
            if not t['served'] and t['arrival_time'] <= self.timestep:
                wait = self.timestep - t['arrival_time']
                if wait > MAX_WAIT:
                    reward -= OVER_PENALTY * (wait - MAX_WAIT)

        # —— 多人抢同一任务惩罚 —— 
        counts = {}
        for r in self.robots:
            if r.task:
                counts[r.task['task_id']] = counts.get(r.task['task_id'],0)+1
        for c in counts.values():
            if c>1:
                reward -= 10*c

                # 10) 互相换位死锁惩罚
        for i, ri in enumerate(self.robots):
            for j, rj in enumerate(self.robots[i+1:], start=i+1):
                if ri.path and rj.path:
                    nxt_i = ri.path[0]
                    nxt_j = rj.path[0]
                    # 如果下一步 ri 要到 rj 的当前位置，且 rj 下一步要到 ri 的当前位置
                    if nxt_i == tuple(rj.pos) and nxt_j == tuple(ri.pos):
                        reward -= 20

        # 11) 长时间不动惩罚（防止多机器人互相堵在一起）
        #for r in self.robots:
            #if r.idle_counter > IDLE_TIMEOUT * 0.5:  # 你自己调参数,已经调整
                #reward -= 5 * r.idle_counter

        # —— 全局进度 —— 
        comp_rate = cur_done / max(1,len(self.tasks))
        reward += 1000 * (comp_rate - 0.3)

        return reward
# 创建环境实例的工厂
def make_env(scale, distribution, arrival):
    return lambda: RobotSchedulingEnv(
        scale=scale,
        parking_distribution=distribution,
        arrival_mode=arrival
    )

# 训练流程
def train(scale, total_timesteps, n_envs, distribution, arrival):
    vec_env = DummyVecEnv([make_env(scale, distribution, arrival) for _ in range(n_envs)])
    vec_env = VecMonitor(vec_env)
    # 评估环境
    eval_env = DummyVecEnv([make_env(scale, distribution, arrival)])
    tmp_env = RobotSchedulingEnv(scale=scale, parking_distribution=distribution, arrival_mode=arrival)
    threshold = tmp_env.TOTAL_TASKS * 0.8
    stop_cb = StopTrainingOnRewardThreshold(reward_threshold=threshold, verbose=1)
    eval_cb = EvalCallback(
        eval_env, callback_on_new_best=stop_cb,
        best_model_save_path="./best_model/",
        log_path="./eval_logs/", eval_freq=10_000,
        deterministic=True, render=False
    )
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        learning_rate=2e-4,      # 更小学习率，价值网络更新更平滑
        n_steps=1024,            # 每 1024 步进行一次网络更新
        batch_size=256,          # 更大 batch 提升稳定性
        n_epochs=8,              # 减少重复更新次数，防止过拟合
        gamma=0.98,              # 稍微降低折扣因子，重视近期回报
        gae_lambda=0.95,         # 更严格的优势估计，减少高方差
        clip_range=0.15,         # 缩小 PPO 剪切范围
        vf_coef=0.5,             # 价值函数损失占比（默认 0.5，可根据需要微调）
        ent_coef=0.01,           # 加一点熵奖励，保证足够探索
        tensorboard_log=LOG_DIR
    )
    model.learn(total_timesteps=total_timesteps, callback=eval_cb)
    # 保存模型与策略权重
    model_path = f"ppo_robot_{scale}_{distribution}_{arrival}.zip"
    model.save(model_path)
    print(f"[INFO] Saved SB3 model to {model_path}")
    policy_path = f"policy_{scale}_{distribution}_{arrival}.pth"
    torch.save(model.policy.state_dict(), policy_path)
    print(f"[INFO] Saved policy weights to {policy_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", choices=["small","medium","large"], default="medium")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--distribution", choices=["uniform","clustered","mixed"], default="uniform")
    parser.add_argument("--arrival", choices=["poisson","uniform","normal"], default="poisson")
    args = parser.parse_args()
    train(
        scale=args.scale,
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        distribution=args.distribution,
        arrival=args.arrival
    )
