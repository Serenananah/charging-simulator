import time
import numpy as np
import random
import copy
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import threading

# 导入环境常量和函数，确保与environment.py兼容
from environment import (
    a_star, path_cost,  # 路径规划函数
    MOVE_COST, CHARGE_TRANSFER, MAX_BATTERY,  # 能量相关常量
    LOW_BATTERY_THRESHOLD,  # 电池状态阈值
    OBSTACLE, PARKING_SPOT, CHARGING_STATION, EMPTY  # 地图元素常量
)

# --- 混合SPSO配置 ---
POPULATION_SIZE = 12  # 粒子数量
MAX_ITERATIONS = 20  # 最大迭代次数
TIME_LIMIT = 2.0  # 主计算时间限制(秒)
URGENT_TASK_THRESHOLD = 20  # 紧急任务的时间阈值
MAX_TASKS_PER_ROBOT = 2  # 每个机器人最多分配的任务数
ENABLE_THREADING = True  # 启用多线程计算
MAX_THREADS = 4  # 最大线程数
BATCH_SIZE = 3  # 每个批次处理的粒子数

# PSO参数
INERTIA_WEIGHT = 0.7  # 惯性权重
C1 = 1.5  # 个体学习因子
C2 = 1.5  # 社会学习因子
LOCAL_REFINEMENT_PROB = 0.2  # 局部优化概率

# 定义自己的常量
CRITICAL_BATTERY = 5.0  # 最低安全电量值

# --- 全局缓存 ---
# 使用线程锁保护全局缓存
path_cache_lock = threading.Lock()
global_path_cache = {}

def get_cached_path(cache_key):
    """从全局缓存获取路径"""
    with path_cache_lock:
        return global_path_cache.get(cache_key)

def set_cached_path(cache_key, path):
    """设置全局缓存中的路径"""
    with path_cache_lock:
        global_path_cache[cache_key] = path

# --- 贪心算法部分 ---
def greedy_assignment(robots, tasks, grid, chargers, height, width, current_time=0):
    """
    贪心算法为机器人分配任务
    
    Args:
        robots: 机器人列表
        tasks: 任务列表
        grid: 环境网格
        chargers: 充电站列表
        height: 网格高度
        width: 网格宽度
        current_time: 当前时间
        
    Returns:
        dict: 分配结果 {robot_id: {"task_sequence": [task_id], "end_station_id": station_id}}
    """
    assignments = {}
    
    # 筛选可用机器人和未分配任务 - 包括充电中但电量足够的
    available_robots = [r for r in robots if 
                    (r.state == "idle" and r.battery >= LOW_BATTERY_THRESHOLD) or
                    (r.state == "charging" and r.battery >= MAX_BATTERY * 0.8)]
    pending_tasks = [t for t in tasks if not t.get('served', False) and t.get('assigned_to') is None]
    
    # 计算任务紧急度（基于剩余时间）
    task_urgency = {}
    for task in pending_tasks:
        if 'departure_time' in task:
            time_left = task['departure_time'] - current_time
            # 归一化紧急度分数：0（不紧急）到 1（非常紧急）
            urgency = max(0, min(1, 1 - (time_left / 100)))  # 假设最大时间窗口为100
            task_urgency[task['task_id']] = urgency
        else:
            task_urgency[task['task_id']] = 0.5  # 默认中等紧急度
    
    # 为低电量机器人分配最近的充电站
    low_battery_robots = [r for r in robots if r.state == "idle" and r.battery < LOW_BATTERY_THRESHOLD]
    for robot in low_battery_robots:
        closest_station = find_closest_station(robot.pos, robot.battery, chargers, grid, height, width)
        if closest_station:
            assignments[robot.id] = {
                "task_sequence": [],
                "end_station_id": f"S{closest_station[0]}"
            }
    
    # 为每个紧急任务找最近的机器人
    urgent_tasks = sorted(
        [(t, task_urgency.get(t['task_id'], 0)) for t in pending_tasks],
        key=lambda x: x[1],
        reverse=True  # 最紧急的任务优先
    )
    
    assigned_tasks = set()
    
    for task, urgency in urgent_tasks:
        if task['task_id'] in assigned_tasks:
            continue
            
        # 按距离排序可用机器人
        candidates = []
        for robot in available_robots:
            if robot.id in assignments:  # 跳过已分配的机器人
                continue
                
            # 计算路径
            path = a_star(grid, robot.pos, task['location'], height, width)
            if not path:
                continue
                
            # 计算成本和可行性
            distance = len(path) - 1
            energy_to_task = distance * MOVE_COST
            energy_for_task = task['required_energy'] - task.get('initial_energy', 0)
            total_energy = energy_to_task + energy_for_task
            
            # 检查电池是否足够
            if robot.battery < total_energy + CRITICAL_BATTERY:
                continue
                
            # 检查完成任务后能否到达充电站
            can_reach_station = False
            closest_station_after_task = None
            
            for i, charger in enumerate(chargers):
                path_to_station = a_star(grid, task['location'], charger, height, width)
                if path_to_station:
                    energy_to_station = (len(path_to_station) - 1) * MOVE_COST
                    if robot.battery - total_energy >= energy_to_station + CRITICAL_BATTERY:
                        can_reach_station = True
                        closest_station_after_task = i
                        break
            
            if can_reach_station:
                # 计算综合分数（距离越短，紧急度越高，分数越高）
                score = urgency / (distance + 1)  # 加1避免除零
                candidates.append((robot, distance, score, closest_station_after_task))
        
        if candidates:
            # 选择综合分数最高的机器人
            candidates.sort(key=lambda x: x[2], reverse=True)
            best_robot, _, _, station_id = candidates[0]
            
            # 分配任务
            assignments[best_robot.id] = {
                "task_sequence": [task['task_id']],
                "end_station_id": f"S{station_id}"
            }
            assigned_tasks.add(task['task_id'])
            available_robots.remove(best_robot)  # 从可用列表中移除
    
    return assignments

def find_closest_station(position, battery, chargers, grid, height, width):
    """找到最近的可达充电站"""
    closest_station = None
    min_distance = float('inf')
    
    for i, charger in enumerate(chargers):
        # 首先检查全局缓存
        cache_key = (tuple(position), tuple(charger))
        cached_path = get_cached_path(cache_key)
        
        if cached_path:
            path = cached_path
        else:
            path = a_star(grid, position, charger, height, width)
            if path:
                set_cached_path(cache_key, path)
                
        if not path:
            continue
            
        distance = len(path) - 1
        energy_needed = distance * MOVE_COST
        
        if battery >= energy_needed + CRITICAL_BATTERY:
            if distance < min_distance:
                min_distance = distance
                closest_station = (i, charger)
    
    return closest_station

# --- 基本工具函数 ---
def normalize_prefs(prefs_dict):
    """将偏好分数归一化为概率"""
    total_pref = sum(prefs_dict.values())
    if total_pref > 0:
        return {k: v / total_pref for k, v in prefs_dict.items()}
    elif prefs_dict:  # 处理总和为零的情况 - 平均分配
        return {k: 1.0 / len(prefs_dict) for k in prefs_dict}
    else:
        return {}

# --- 粒子类 ---
class Particle:
    def __init__(self, robots, tasks, stations, grid, initialize_with_greedy=False, current_time=0, thread_id=0):
        """初始化粒子
        
        Args:
            robots: 机器人列表
            tasks: 待处理任务列表
            stations: 充电站列表
            grid: 环境网格
            initialize_with_greedy: 是否使用贪心算法初始化
            current_time: 当前时间
            thread_id: 线程ID，用于调试
        """
        # 线程ID，用于调试
        self.thread_id = thread_id
        
        # 初始化私有路径缓存 - 最先初始化
        self._path_cache = {}
        
        # 保存环境引用
        self.grid = grid
        self.height, self.width = grid.shape
        self.current_time = current_time
        
        # 预处理并规范化数据
        self.robots = self._adapt_robots(robots)
        self.tasks = self._adapt_tasks(tasks)
        self.stations = self._adapt_stations(stations)
        
        # 构建查找字典
        self.robots_dict = {r['id']: r for r in self.robots}
        self.tasks_dict = {t['id']: t for t in self.tasks}
        self.stations_dict = {s['id']: s for s in self.stations}
        
        # 粒子状态
        self.position = {}  # 当前解决方案
        self.velocity = {}  # 当前速度(偏好)
        self.pbest_position = {}  # 个体历史最优位置
        self.pbest_fitness = float('inf')  # 个体历史最优适应度
        self.current_fitness = float('inf')  # 当前适应度
        
        # 初始化方式
        if initialize_with_greedy and len(self.tasks) > 0:
            # 转换为原始格式的数据进行贪心分配
            original_robots = [r for r in robots if isinstance(r, object) and hasattr(r, 'id')]
            original_tasks = [t for t in tasks if isinstance(t, dict) and 'task_id' in t]
            original_stations = [s['location'] for s in self.stations]
            
            # 使用贪心算法初始化
            try:
                greedy_result = greedy_assignment(
                    original_robots, original_tasks, grid, original_stations, self.height, self.width, current_time
                )
                # 转换结果为粒子位置格式
                self.position = greedy_result
                # 根据贪心解初始化速度偏好
                self._initialize_velocity_from_position()
                self.current_fitness = self.calculate_fitness(self.position)
                self.pbest_position = copy.deepcopy(self.position)
                self.pbest_fitness = self.current_fitness
            except Exception as e:
                print(f"贪心初始化失败: {e}")
                # 失败时回退到随机初始化
                self._initialize_velocity()
                self.position = self._construct_solution()
                self.current_fitness = self.calculate_fitness(self.position)
                self.pbest_position = copy.deepcopy(self.position)
                self.pbest_fitness = self.current_fitness
        else:
            # 标准随机初始化
            self._initialize_velocity()
            self.position = self._construct_solution()
            self.current_fitness = self.calculate_fitness(self.position)
            self.pbest_position = copy.deepcopy(self.position)
            self.pbest_fitness = self.current_fitness
    
    def _energy_efficient_path(self, start_pos, candidates, remaining_battery):
        """选择能耗效率最高的下一个目标位置
        
        Args:
            start_pos: 起始位置
            candidates: 候选目标列表[(id, location, energy_required), ...]
            remaining_battery: 剩余电量
            
        Returns:
            tuple: (best_id, path_cost, energy_required)，如果无可行目标则返回(None, None, None)
        """
        best_score = float('-inf')
        best_id = None
        best_cost = None
        best_required = None
        
        for cand_id, location, energy_required in candidates:
            # 计算距离成本
            dist = self._get_path_cost(start_pos, location)
            if dist == float('inf'):
                continue
                
            # 计算总能耗
            total_energy = dist * MOVE_COST + energy_required
            
            # 检查是否有足够电量
            if total_energy + CRITICAL_BATTERY > remaining_battery:
                continue
                
            # 计算能效分数 (任务价值/能量消耗)
            # 紧急任务给予更高价值
            task_value = 1.0
            if cand_id in self.tasks_dict:
                task = self.tasks_dict[cand_id]
                if 'latest_departure_time' in task:
                    time_left = task['latest_departure_time'] - self.current_time
                    urgency_factor = max(1.0, 5.0 / max(1, time_left))
                    task_value = urgency_factor
            
            # 能效分数
            energy_efficiency = task_value / total_energy
            
            # 如果该选择效率更高，更新最佳选择
            if energy_efficiency > best_score:
                best_score = energy_efficiency
                best_id = cand_id
                best_cost = dist
                best_required = energy_required
        
        return best_id, best_cost, best_required
    
    def _adapt_robots(self, robots):
        """将Robot对象转换为标准字典格式"""
        adapted_robots = []
        for r in robots:
            # 对于环境中的Robot对象
            if hasattr(r, 'id') and hasattr(r, 'pos') and hasattr(r, 'battery'):
                # 如果是充电状态且电量达到一定水平，将其视为空闲状态
                state = r.state
                if state == "charging" and r.battery >= MAX_BATTERY * 0.8:
                    state = "idle"  # 将充电状态但电量足够的机器人视为空闲
                    
                robot_dict = {
                    'id': r.id,
                    'position': tuple(r.pos) if isinstance(r.pos, (list, tuple)) else r.pos,
                    'battery': r.battery,
                    'state': state
                }
                adapted_robots.append(robot_dict)
            # 对于已经是字典格式的
            elif isinstance(r, dict) and 'id' in r:
                # 确保位置是标准元组格式
                if 'position' in r:
                    r['position'] = tuple(r['position']) if isinstance(r['position'], (list, tuple)) else r['position']
                # 同样处理充电状态
                if r.get('state') == "charging" and r.get('battery', 0) >= MAX_BATTERY * 0.8:
                    r['state'] = "idle"
                adapted_robots.append(r)
        return adapted_robots
    
    def _adapt_tasks(self, tasks):
        """将Task对象转换为标准字典格式，并加入等待时间评估"""
        adapted_tasks = []
        for i, t in enumerate(tasks):
            # 处理环境中的任务
            if isinstance(t, dict):
                # 任务 ID 可能是 'task_id' 或 'id'
                task_id = t.get('task_id', None)
                if task_id is None:
                    task_id = t.get('id', f"T{i}")
                    
                # 对位置进行规范化
                location = t.get('location', None)
                if location:
                    location = tuple(location) if isinstance(location, (list, tuple)) else location
                    
                # 计算任务等待时间
                wait_time = 0
                if 'arrival_time' in t:
                    wait_time = max(0, self.current_time - t['arrival_time'])
                
                # 计算任务紧急度
                urgency = 0.5  # 默认中等紧急度
                if 'departure_time' in t:
                    time_left = t['departure_time'] - self.current_time
                    # 紧急度：时间越短越紧急
                    urgency = max(0.1, min(0.9, 1.0 - time_left / 100))
                    
                task_dict = {
                    'id': task_id,
                    'location': location,
                    'required_energy': t.get('required_energy', 0),
                    'initial_energy': t.get('initial_energy', 0),
                    'status': 'pending' if not t.get('served', False) and t.get('assigned_to') is None else 'assigned',
                    'latest_departure_time': t.get('departure_time', float('inf')),
                    'arrival_time': t.get('arrival_time', 0),
                    'wait_time': wait_time,  # 新增：等待时间
                    'urgency': urgency       # 新增：紧急度
                }
                adapted_tasks.append(task_dict)
        return adapted_tasks
    
    def _adapt_stations(self, stations):
        """将充电站转换为标准字典格式"""
        adapted_stations = []
        for i, s in enumerate(stations):
            # 如果是元组/列表格式的位置
            if isinstance(s, (list, tuple)) and len(s) == 2:
                station_dict = {
                    'id': f"S{i}",
                    'location': tuple(s)
                }
                adapted_stations.append(station_dict)
            # 如果已经是字典格式
            elif isinstance(s, dict) and 'id' in s and 'location' in s:
                s['location'] = tuple(s['location']) if isinstance(s['location'], (list, tuple)) else s['location']
                adapted_stations.append(s)
        return adapted_stations
    
    def _get_path(self, start, goal):
        """获取路径，使用缓存提高性能"""
        try:
            # 规范化位置 - 确保是元组格式
            if isinstance(start, dict) and 'position' in start:
                start_pos = tuple(start['position']) if isinstance(start['position'], (list, tuple)) else start['position']
            else:
                start_pos = tuple(start) if isinstance(start, (list, tuple)) else start
                
            if isinstance(goal, dict) and 'location' in goal:
                goal_pos = tuple(goal['location']) if isinstance(goal['location'], (list, tuple)) else goal['location']
            else:
                goal_pos = tuple(goal) if isinstance(goal, (list, tuple)) else goal
            
            # 首先检查全局缓存
            cache_key = (start_pos, goal_pos)
            cached_path = get_cached_path(cache_key)
            
            if cached_path:
                return cached_path
            
            # 其次检查私有缓存
            if hasattr(self, '_path_cache') and cache_key in self._path_cache:
                return self._path_cache[cache_key]
            
            # 确保 _path_cache 存在
            if not hasattr(self, '_path_cache'):
                self._path_cache = {}
                
            # 计算新路径
            path = a_star(self.grid, start_pos, goal_pos, self.height, self.width)
            if path:
                # 保存到私有缓存
                self._path_cache[cache_key] = path
                # 也保存到全局缓存
                set_cached_path(cache_key, path)
            return path
        except Exception as e:
            print(f"路径计算错误: {e} - start: {start}, goal: {goal}")
            return None  # 返回 None 而不是抛出异常
    
    def _get_path_cost(self, start, goal):
        """获取路径成本，使用缓存提高性能"""
        path = self._get_path(start, goal)
        if not path:
            return float('inf')
        return len(path) - 1
    
    def _initialize_velocity_from_position(self):
        """根据当前位置(贪心解)初始化速度(偏好)"""
        self.velocity = {}
        
        # 为每个机器人初始化偏好
        for robot in self.robots:
            robot_id = robot['id']
            self.velocity[robot_id] = {
                "task_prefs": {},   # 任务偏好集合
                "station_prefs": {} # 充电站偏好集合
            }
            
            # 从位置提取机器人的任务和充电站
            if robot_id in self.position:
                assigned_tasks = self.position[robot_id].get("task_sequence", [])
                assigned_station = self.position[robot_id].get("end_station_id")
                
                # 为分配的任务设置高偏好
                for task in self.tasks:
                    task_id = task['id']
                    if task_id in assigned_tasks:
                        self.velocity[robot_id]["task_prefs"][task_id] = 0.9  # 高偏好
                    else:
                        # 为未分配任务设置随机低偏好
                        self.velocity[robot_id]["task_prefs"][task_id] = random.uniform(0.1, 0.3)
                
                # 为分配的充电站设置高偏好
                for station in self.stations:
                    station_id = station['id']
                    if station_id == assigned_station:
                        self.velocity[robot_id]["station_prefs"][station_id] = 0.9  # 高偏好
                    else:
                        # 为未分配充电站设置随机低偏好
                        self.velocity[robot_id]["station_prefs"][station_id] = random.uniform(0.1, 0.3)
            else:
                # 如果位置中没有该机器人，使用标准方法初始化
                self._initialize_velocity_for_robot(robot_id)
    
    def _initialize_velocity(self):
        """初始化速度(偏好)"""
        self.velocity = {}
        
        # 为每个机器人初始化偏好
        for robot in self.robots:
            robot_id = robot['id']
            self._initialize_velocity_for_robot(robot_id)
    
    def _initialize_velocity_for_robot(self, robot_id):
        """为单个机器人初始化速度(偏好)"""
        robot = self.robots_dict[robot_id]
        
        self.velocity[robot_id] = {
            "task_prefs": {},   # 任务偏好集合
            "station_prefs": {} # 充电站偏好集合
        }
        
        # 计算任务偏好
        for task in self.tasks:
            if task['status'] != 'pending':
                continue
                
            task_id = task['id']
            task_location = task['location']
            
            # 计算到任务的距离
            dist = self._get_path_cost(robot['position'], task_location)
            
            if dist == float('inf'):
                # 不可达任务给低初始偏好
                self.velocity[robot_id]["task_prefs"][task_id] = 0.01
                continue
            
            # 1. 距离偏好(距离越近越优先)
            dist_pref = max(0.05, min(0.8, 5.0 / (dist + 1)))
            
            # 2. 时间紧急度偏好
            urgency_pref = 0.5  # 默认中等紧急度
            if 'latest_departure_time' in task:
                time_left = task['latest_departure_time'] - self.current_time
                # 紧急度：时间越短越紧急
                urgency_pref = max(0.1, min(0.9, 1.0 - time_left / 100))
            
            # 3. 能量效率偏好(能量传输效率高的任务优先)
            energy_required = task['required_energy'] - task.get('initial_energy', 0.0)
            energy_efficiency = energy_required / max(1, dist)
            efficiency_pref = min(0.8, max(0.1, energy_efficiency / 20.0))
            
            # 计算总偏好 - 紧急度权重更高
            # 计算任务紧急性加权
            urgency_weight = 0.5
            # 如果任务非常紧急，提高权重
            if 'latest_departure_time' in task:
                time_left = task['latest_departure_time'] - self.current_time
                if time_left < URGENT_TASK_THRESHOLD:  # 20时间单位内的任务视为紧急
                    urgency_weight = 0.7  # 增加紧急任务权重
                    
            # 计算总偏好 - 紧急度权重更高
            self.velocity[robot_id]["task_prefs"][task_id] = (
                0.3 * dist_pref + 
                urgency_weight * urgency_pref + 
                (1.0 - 0.3 - urgency_weight) * efficiency_pref  # 确保权重总和为1
            )
        
        # 计算充电站偏好
        for station in self.stations:
            station_id = station['id']
            station_location = station['location']
            
            # 计算到充电站的距离
            dist = self._get_path_cost(robot['position'], station_location)
            
            if dist == float('inf'):
                self.velocity[robot_id]["station_prefs"][station_id] = 0.01
            else:
                # 距离越近，初始偏好越高
                self.velocity[robot_id]["station_prefs"][station_id] = max(0.05, min(0.5, 5.0 / (dist + 1)))
    
    def _construct_solution(self):
        """根据当前速度(偏好)构建解决方案"""
        position = {}
        remaining_tasks = {t['id']: t for t in self.tasks if t['status'] == 'pending'}
        assigned_tasks = set()
        
        # 处理每个机器人
        for robot in self.robots:
            robot_id = robot['id']
            task_prefs = self.velocity.get(robot_id, {}).get("task_prefs", {})
            station_prefs = self.velocity.get(robot_id, {}).get("station_prefs", {})
            
            # 跳过无效机器人
            if robot_id not in self.velocity:
                continue
                
            current_assignment = {"task_sequence": [], "end_station_id": None}
            simulated_pos = robot['position']
            simulated_battery = robot['battery']
            
            # 尝试分配任务
            # 按偏好排序候选任务
            candidate_tasks = [(tid, task_prefs.get(tid, 0)) for tid in remaining_tasks 
                               if tid not in assigned_tasks]
            candidate_tasks.sort(key=lambda x: x[1], reverse=True)
            
            for task_id, _ in candidate_tasks:
                # 如果已达最大任务数，跳出
                if len(current_assignment["task_sequence"]) >= MAX_TASKS_PER_ROBOT:
                    break
                    
                task = remaining_tasks[task_id]
                
                # 计算到任务的路径
                path_to_task = self._get_path(simulated_pos, task['location'])
                if not path_to_task:
                    continue
                
                # 计算路径和任务能量消耗
                energy_to_task = (len(path_to_task) - 1) * MOVE_COST
                energy_for_charging = task['required_energy'] - task.get('initial_energy', 0)
                total_energy = energy_to_task + energy_for_charging
                
                # 检查电池是否足够
                if simulated_battery < total_energy + CRITICAL_BATTERY:
                    continue
                
                # 检查完成任务后能否到达充电站
                can_reach_station = False
                best_station_id = None
                
                # 按偏好排序充电站
                candidate_stations = [(sid, station_prefs.get(sid, 0)) for sid in self.stations_dict]
                candidate_stations.sort(key=lambda x: x[1], reverse=True)
                
                for station_id, _ in candidate_stations:
                    station = self.stations_dict[station_id]
                    station_location = station['location']
                    
                    # 计算任务到充电站的路径
                    path_to_station = self._get_path(task['location'], station_location)
                    if not path_to_station:
                        continue
                    
                    energy_to_station = (len(path_to_station) - 1) * MOVE_COST
                    
                    # 检查能否在保留临界电量的情况下完成任务并到达充电站
                    if simulated_battery - total_energy >= energy_to_station + CRITICAL_BATTERY:
                        can_reach_station = True
                        best_station_id = station_id
                        break
                
                if can_reach_station:
                    # 分配任务
                    current_assignment["task_sequence"].append(task_id)
                    assigned_tasks.add(task_id)
                    
                    # 更新模拟状态
                    simulated_pos = task['location']
                    simulated_battery -= total_energy
                else:
                    # 如果无法到达任何充电站，跳过该任务
                    continue
            
            # 如果没有分配任务，为机器人分配充电站
            if not current_assignment["task_sequence"]:
                # 按偏好排序充电站
                candidate_stations = [(sid, station_prefs.get(sid, 0)) for sid in self.stations_dict]
                candidate_stations.sort(key=lambda x: x[1], reverse=True)
                
                for station_id, _ in candidate_stations:
                    station = self.stations_dict[station_id]
                    station_location = station['location']
                    
                    # 计算到充电站的路径
                    path_to_station = self._get_path(simulated_pos, station_location)
                    if not path_to_station:
                        continue
                    
                    energy_to_station = (len(path_to_station) - 1) * MOVE_COST
                    
                    # 检查能否到达充电站
                    if simulated_battery >= energy_to_station + CRITICAL_BATTERY:
                        current_assignment["end_station_id"] = station_id
                        break
            # 确保有终点充电站
            if current_assignment["task_sequence"] and not current_assignment["end_station_id"]:
                # 从最后一个任务位置找最近的充电站
                last_task_id = current_assignment["task_sequence"][-1]
                last_task = remaining_tasks.get(last_task_id)
                
                if last_task:
                    for station in self.stations:
                        path = self._get_path(last_task['location'], station['location'])
                        if path:
                            current_assignment["end_station_id"] = station['id']
                            break
            
            # 保存当前机器人的分配方案
            position[robot_id] = current_assignment
        
        return position

    def update_velocity(self, gbest_position):
        """更新速度(偏好)
        
        Args:
            gbest_position: 全局最优位置
        """
        new_velocity = {}
        
        # 为每个机器人更新速度
        for robot_id in self.velocity:
            new_velocity[robot_id] = {
                "task_prefs": {},
                "station_prefs": {}
            }
            
            # 更新任务偏好
            for task_id in self.velocity[robot_id].get("task_prefs", {}):
                # 当前偏好
                v_old = self.velocity[robot_id]["task_prefs"].get(task_id, 0)
                
                # 个体最优和全局最优的影响
                pbest_influence = 0
                gbest_influence = 0
                
                # 计算个体最优影响
                if robot_id in self.pbest_position:
                    pbest_tasks = self.pbest_position[robot_id].get("task_sequence", [])
                    pbest_influence = 1.0 if task_id in pbest_tasks else -0.5
                
                # 计算全局最优影响
                if robot_id in gbest_position:
                    gbest_tasks = gbest_position[robot_id].get("task_sequence", [])
                    gbest_influence = 1.0 if task_id in gbest_tasks else -0.5
                
                # 更新公式: v = w*v + c1*r1*(pbest-x) + c2*r2*(gbest-x)
                v_new = (INERTIA_WEIGHT * v_old + 
                         C1 * random.random() * pbest_influence + 
                         C2 * random.random() * gbest_influence)
                
                # 限制速度范围
                v_new = max(0.01, min(1.0, v_new))
                new_velocity[robot_id]["task_prefs"][task_id] = v_new
            
            # 更新充电站偏好
            for station_id in self.velocity[robot_id].get("station_prefs", {}):
                # 当前偏好
                v_old = self.velocity[robot_id]["station_prefs"].get(station_id, 0)
                
                # 个体最优和全局最优的影响
                pbest_influence = 0
                gbest_influence = 0
                
                # 计算个体最优影响
                if robot_id in self.pbest_position:
                    pbest_station = self.pbest_position[robot_id].get("end_station_id")
                    pbest_influence = 1.0 if station_id == pbest_station else -0.3
                
                # 计算全局最优影响
                if robot_id in gbest_position:
                    gbest_station = gbest_position[robot_id].get("end_station_id")
                    gbest_influence = 1.0 if station_id == gbest_station else -0.3
                
                # 更新公式
                v_new = (INERTIA_WEIGHT * v_old + 
                         C1 * random.random() * pbest_influence + 
                         C2 * random.random() * gbest_influence)
                
                # 限制速度范围
                v_new = max(0.01, min(1.0, v_new))
                new_velocity[robot_id]["station_prefs"][station_id] = v_new
        
        self.velocity = new_velocity
    
    def update_position(self):
        """更新位置(解决方案)"""
        # 根据更新后的速度构建新解决方案
        self.position = self._construct_solution()
        self.current_fitness = self.calculate_fitness(self.position)
        
        # 更新个体最优
        if self.current_fitness < self.pbest_fitness:
            self.pbest_fitness = self.current_fitness
            self.pbest_position = copy.deepcopy(self.position)
    
    def calculate_fitness(self, position):
        """计算适应度函数值
        
        Args:
            position: 解决方案
            
        Returns:
            float: 适应度值(越小越好)
        """
        if not position:
            return float('inf')
        
        # 1. 计算总行驶距离
        total_distance = 0
        # 2. 任务等待时间
        wait_times = []
        # 3. 任务分配平衡性
        task_counts = []
        # 4. 未分配任务惩罚
        pending_task_ids = {t['id'] for t in self.tasks if t['status'] == 'pending'}
        assigned_task_ids = set()
        # 5. 任务截止时间违反情况
        deadline_violations = 0
        
        # 遍历每个机器人的分配
        for robot_id, assignment in position.items():
            robot = self.robots_dict.get(robot_id)
            if not robot:
                continue
                
            current_pos = robot['position']
            tasks_in_sequence = assignment.get("task_sequence", [])
            station_id = assignment.get("end_station_id")
            
            # 记录任务数量
            task_counts.append(len(tasks_in_sequence))
            
            # 添加到已分配任务集合
            assigned_task_ids.update(tasks_in_sequence)
            
            # 模拟执行时间
            sim_time = self.current_time
            
            # 计算任务序列的距离
            for task_id in tasks_in_sequence:
                task = self.tasks_dict.get(task_id)
                if not task:
                    continue
                    
                task_location = task['location']
                dist = self._get_path_cost(current_pos, task_location)
                
                if dist == float('inf'):
                    return float('inf')  # 无效路径
                    
                total_distance += dist
                
                # 模拟到达时间
                sim_time += dist  # 简化：距离 = 时间
                
                # 模拟等待时间
                if 'arrival_time' in task:
                    wait_time = max(0, sim_time - task['arrival_time'])
                    wait_times.append(wait_time)
                
                # 模拟任务执行时间
                energy_required = task['required_energy'] - task.get('initial_energy', 0)
                task_execution_time = energy_required * 0.2  # 简化：每单位能量需要0.2时间
                sim_time += task_execution_time
                
                # 检查是否违反任务截止时间
                if 'latest_departure_time' in task and sim_time > task['latest_departure_time']:
                    deadline_violations += 1
                
                current_pos = task_location
            
            # 计算到充电站的距离
            if station_id:
                station = self.stations_dict.get(station_id)
                if station:
                    station_location = station['location']
                    dist = self._get_path_cost(current_pos, station_location)
                    
                    if dist == float('inf'):
                        return float('inf')  # 无效路径
                        
                    total_distance += dist
        
        # 计算未分配任务数
        unassigned_count = len(pending_task_ids - assigned_task_ids)
        
        # 计算权重适应度
        α1 = 1.0  # 距离权重 - 降低以更重视任务完成
        α2 = 3.0  # 等待时间权重 - 增加以减少平均等待时间
        α3 = 1.5  # 平衡性权重
        α4 = 15.0  # 未分配任务惩罚权重 - 显著增加未分配任务的惩罚
        α5 = 20.0  # 截止时间违反惩罚权重 - 增加以减少超时
        α6 = 3.5  # 等待时间标准差权重 - 增加以平衡任务处理
        
        avg_wait = sum(wait_times) / len(wait_times) if wait_times else 0
        wait_std = math.sqrt(np.var(wait_times)) if len(wait_times) > 1 else 0
        balance_penalty = np.std(task_counts) if task_counts else 0
        
        # 引入等待时间标准差作为优化目标
        fitness = (
            α1 * total_distance + 
            α2 * avg_wait + 
            α3 * balance_penalty + 
            α4 * unassigned_count + 
            α5 * deadline_violations +
            α6 * wait_std  # 新增：等待时间标准差
        )
        
        return fitness

def evaluate_particle(particle, gbest_position):
    """用于并行计算的粒子评估函数"""
    # 更新粒子速度
    particle.update_velocity(gbest_position)
    
    # 更新粒子位置
    particle.update_position()
    
    # 应用局部优化(概率性)
    if random.random() < LOCAL_REFINEMENT_PROB:
        # 创建局部优化函数的引用
        local_refinement = HybridSPSODispatcher._local_refinement
        refined_position = local_refinement(HybridSPSODispatcher, particle.position, particle)
        refined_fitness = particle.calculate_fitness(refined_position)
        
        if refined_fitness < particle.current_fitness:
            particle.position = refined_position
            particle.current_fitness = refined_fitness
            
            # 更新个体最优
            if refined_fitness < particle.pbest_fitness:
                particle.pbest_fitness = refined_fitness
                particle.pbest_position = copy.deepcopy(refined_position)
    
    return particle

# --- HybridSPSODispatcher类 ---
class HybridSPSODispatcher:
    def __init__(self, robots_all, tasks_all, stations_all, grid_ref, current_time=0):
        """初始化混合SPSO调度器
        
        Args:
            robots_all: 所有机器人列表
            tasks_all: 所有任务列表
            stations_all: 所有充电站列表
            grid_ref: 环境网格
            current_time: 当前时间步
        """
        self.robots_all = robots_all
        self.tasks_all = tasks_all
        self.stations_all = stations_all
        self.grid = grid_ref
        self.height, self.width = grid_ref.shape
        self.current_time = current_time
        
        # 初始化全局最优
        self.gbest_position = {}
        self.gbest_fitness = float('inf')
        
        # 初始化粒子群
        self.population = []
        
        # 线程池
        self.executor = None if not ENABLE_THREADING else ThreadPoolExecutor(max_workers=MAX_THREADS)
    
    def __del__(self):
        """析构函数，确保线程池关闭"""
        if self.executor:
            self.executor.shutdown(wait=False)
    
    def run_optimization(self):
        """运行混合SPSO优化
        
        Returns:
            dict: 优化后的分配方案
        """
        # 记录开始时间
        start_time = time.time()
        
        # 1. 筛选数据
        # 筛选可分配任务的机器人和低电量机器人
        # 定义一个新的电量阈值，表示充电到这个程度就可以工作
        SUFFICIENT_BATTERY_LEVEL = MAX_BATTERY * 0.8  # 充到80%就足够了

        # 筛选可分配任务的机器人和低电量机器人
        assignable_robots = [r for r in self.robots_all if 
                            (r.state == "idle" and r.battery >= LOW_BATTERY_THRESHOLD) or
                            (r.state == "charging" and r.battery >= SUFFICIENT_BATTERY_LEVEL)]
        low_battery_robots = [r for r in self.robots_all if 
                            r.state == "idle" and r.battery < LOW_BATTERY_THRESHOLD]
        
        # 筛选待处理任务
        pending_tasks = [t for t in self.tasks_all if not t.get('served', False) and t.get('assigned_to') is None]
        
        # 快速检查：如果没有待处理任务或可用机器人
        if not pending_tasks or (not assignable_robots and not low_battery_robots):
            return {}
        
        print(f"运行混合SPSO: {len(assignable_robots)}个可用机器人, {len(low_battery_robots)}个低电量机器人, {len(pending_tasks)}个待处理任务")
        
        # 2. 识别紧急任务
        urgent_tasks = [t for t in pending_tasks if 
                        'departure_time' in t and 
                        t['departure_time'] - self.current_time <= URGENT_TASK_THRESHOLD]
        
        normal_tasks = [t for t in pending_tasks if 
                        t not in urgent_tasks]
        
        assignments = {}
        
        # 3. 使用贪心算法处理低电量机器人和紧急任务
        if low_battery_robots or urgent_tasks:
            # 准备用于贪心分配的机器人队列
            greedy_robots = low_battery_robots.copy()
            
            # 如果有紧急任务，也用部分正常机器人处理
            if urgent_tasks:
                # 确定用于紧急任务的机器人数量（最多使用一半的可用机器人）
                urgent_robot_count = min(len(urgent_tasks), max(1, len(assignable_robots) // 2))
                greedy_robots.extend(assignable_robots[:urgent_robot_count])
                
                # 更新可用机器人列表
                assignable_robots = assignable_robots[urgent_robot_count:]
            
            # 运行贪心算法
            if greedy_robots:
                greedy_tasks = urgent_tasks + ([] if not assignable_robots else normal_tasks[:len(greedy_robots)])
                if greedy_tasks:
                    stations_data = [s.get('location') if isinstance(s, dict) else s for s in self.stations_all]
                    greedy_assignments = greedy_assignment(
                        greedy_robots, greedy_tasks, self.grid, stations_data, 
                        self.height, self.width, self.current_time
                    )
                    
                    # 合并分配结果
                    assignments.update(greedy_assignments)
                    
                    # 更新任务列表，移除已分配的任务
                    assigned_task_ids = set()
                    for robot_id, assignment in greedy_assignments.items():
                        assigned_task_ids.update(assignment.get("task_sequence", []))
                    
                    normal_tasks = [t for t in normal_tasks if t.get('task_id') not in assigned_task_ids]
        
        # 4. 如果没有足够的机器人或任务，跳过SPSO阶段
        if not assignable_robots or not normal_tasks:
            return assignments
        
        # 5. 使用SPSO处理剩余任务
        # 创建初始粒子群
        self.population = []
        
        # 第一个粒子使用贪心算法初始化
        self.population.append(
            Particle(
                robots=assignable_robots,
                tasks=normal_tasks,
                stations=self.stations_all,
                grid=self.grid,
                initialize_with_greedy=True,
                current_time=self.current_time,
                thread_id=0
            )
        )
        
        # 其余粒子随机初始化
        for i in range(POPULATION_SIZE - 1):
            self.population.append(
                Particle(
                    robots=assignable_robots,
                    tasks=normal_tasks,
                    stations=self.stations_all,
                    grid=self.grid,
                    current_time=self.current_time,
                    thread_id=i+1
                )
            )
        
        # 评估初始粒子群并找到全局最优
        for particle in self.population:
            if particle.current_fitness < self.gbest_fitness:
                self.gbest_fitness = particle.current_fitness
                self.gbest_position = copy.deepcopy(particle.position)
        
        # 主循环：SPSO迭代
        no_improvement_counter = 0
        best_fitness_so_far = self.gbest_fitness
        
        for iteration in range(MAX_ITERATIONS):
            # 检查是否超时
            if time.time() - start_time > TIME_LIMIT:
                print(f"混合SPSO提前终止: 达到时间限制({TIME_LIMIT}秒), 已完成{iteration}次迭代")
                break
            
            # 使用多线程并行处理粒子
            if ENABLE_THREADING and self.executor:
                # 将粒子分批处理
                updated_particles = []
                
                # 提交粒子更新任务到线程池
                futures = []
                for particle in self.population:
                    future = self.executor.submit(evaluate_particle, particle, self.gbest_position)
                    futures.append(future)
                
                # 收集更新后的粒子
                for future in futures:
                    try:
                        updated_particle = future.result()
                        updated_particles.append(updated_particle)
                        
                        # 更新全局最优
                        if updated_particle.current_fitness < self.gbest_fitness:
                            self.gbest_fitness = updated_particle.current_fitness
                            self.gbest_position = copy.deepcopy(updated_particle.position)
                    except Exception as e:
                        print(f"粒子更新线程错误: {e}")
                
                # 更新粒子群
                self.population = updated_particles
            else:
                # 串行处理每个粒子
                for particle in self.population:
                    # 更新粒子速度
                    particle.update_velocity(self.gbest_position)
                    
                    # 更新粒子位置
                    particle.update_position()
                    
                    # 应用局部优化(概率性)
                    if random.random() < LOCAL_REFINEMENT_PROB:
                        refined_position = self._local_refinement(particle.position, particle)
                        refined_fitness = particle.calculate_fitness(refined_position)
                        
                        if refined_fitness < particle.current_fitness:
                            particle.position = refined_position
                            particle.current_fitness = refined_fitness
                            
                            # 更新个体最优
                            if refined_fitness < particle.pbest_fitness:
                                particle.pbest_fitness = refined_fitness
                                particle.pbest_position = copy.deepcopy(refined_position)
                    
                    # 更新全局最优
                    if particle.current_fitness < self.gbest_fitness:
                        self.gbest_fitness = particle.current_fitness
                        self.gbest_position = copy.deepcopy(particle.position)
            
            # 收敛检测
            if self.gbest_fitness < best_fitness_so_far:
                best_fitness_so_far = self.gbest_fitness
                no_improvement_counter = 0
                print(f"迭代 {iteration}: 找到更好的解, 适应度 = {self.gbest_fitness:.2f}")
            else:
                no_improvement_counter += 1
            
            # 早期终止
            if no_improvement_counter >= 6:  # 提前终止
                print(f"混合SPSO提前终止: 连续6次迭代无改进, 已完成{iteration+1}次迭代")
                break
        
        # 合并SPSO分配结果
        spso_assignments = self._verify_assignments(self.gbest_position)
        
        # 合并结果
        for robot_id, assignment in spso_assignments.items():
            # 只添加未在贪心阶段分配的机器人
            if robot_id not in assignments:
                assignments[robot_id] = assignment
        
        # 如果多线程启用，关闭线程池
        if ENABLE_THREADING and self.executor:
            self.executor.shutdown(wait=False)
            self.executor = None
        
        return assignments
    
    @staticmethod
    def _local_refinement(cls, position, particle):
        """局部优化，简化版的2-opt
        
        Args:
            position: 待优化的解决方案
            particle: 粒子对象
            
        Returns:
            dict: 优化后的解决方案
        """
        refined_position = copy.deepcopy(position)
        
        # 对每个机器人的任务序列进行优化
        for robot_id, assignment in refined_position.items():
            task_sequence = assignment.get("task_sequence", [])
            
            # 如果序列太短，跳过
            if len(task_sequence) < 2:
                continue
            
            # 尝试任务交换
            best_sequence = task_sequence.copy()
            best_distance = float('inf')
            
            robot_data = next((r for r in particle.robots if r['id'] == robot_id), None)
            if not robot_data:
                continue
                
            robot_pos = robot_data['position']
            
            # 评估当前序列的距离
            current_distance = 0
            current_pos = robot_pos
            
            for task_id in task_sequence:
                task = particle.tasks_dict.get(task_id)
                if task:
                    task_loc = task['location']
                    dist = particle._get_path_cost(current_pos, task_loc)
                    current_distance += dist
                    current_pos = task_loc
            
            # 尝试交换任务顺序
            for i in range(len(task_sequence)):
                for j in range(i+1, len(task_sequence)):
                    new_sequence = task_sequence.copy()
                    new_sequence[i], new_sequence[j] = new_sequence[j], new_sequence[i]
                    
                    # 评估新序列的距离
                    new_distance = 0
                    current_pos = robot_pos
                    
                    for task_id in new_sequence:
                        task = particle.tasks_dict.get(task_id)
                        if task:
                            task_loc = task['location']
                            dist = particle._get_path_cost(current_pos, task_loc)
                            new_distance += dist
                            current_pos = task_loc
                    
                    # 如果新序列更好，更新
                    if new_distance < best_distance:
                        best_distance = new_distance
                        best_sequence = new_sequence
            
            # 如果找到更好的序列，更新位置
            if best_distance < current_distance:
                refined_position[robot_id]["task_sequence"] = best_sequence
        
        return refined_position
    
    def _verify_assignments(self, position):
        """验证分配方案的可行性
        
        Args:
            position: 分配方案
            
        Returns:
            dict: 验证后的有效分配方案
        """
        if not position:
            return {}
        
        valid_assignments = {}
        
        # 遍历每个机器人的分配
        for robot_id, assignment in position.items():
            # 获取机器人对象
            robot = next((r for r in self.robots_all if r.id == robot_id), None)
            if not robot:
                continue
            
            # 获取任务序列
            task_sequence = assignment.get("task_sequence", [])
            
            # 验证任务序列
            valid_task_sequence = []
            invalid_sequence = False
            
            # 模拟执行任务序列
            sim_pos = robot.pos
            sim_battery = robot.battery
            
            for task_id in task_sequence:
                # 获取任务
                task = None
                # 尝试多种方式匹配任务ID
                for t in self.tasks_all:
                    t_id = t.get('task_id', None)
                    if t_id is not None and (str(t_id) == str(task_id)):
                        task = t
                        break
                
                if not task:
                    invalid_sequence = True
                    break
                
                # 跳过已完成或已分配的任务
                if task.get('served', False) or task.get('assigned_to') is not None:
                    continue
                
                # 计算到任务的路径
                # 首先检查全局缓存
                cache_key = (tuple(sim_pos), tuple(task['location']))
                cached_path = get_cached_path(cache_key)
                
                if cached_path:
                    path = cached_path
                else:
                    path = a_star(self.grid, sim_pos, task['location'], self.height, self.width)
                    if path:
                        set_cached_path(cache_key, path)
                
                if not path:
                    invalid_sequence = True
                    break
                
                # 计算能量消耗
                travel_energy = (len(path) - 1) * MOVE_COST
                charging_energy = task['required_energy'] - task.get('initial_energy', 0)
                total_energy = travel_energy + charging_energy
                
                # 检查电池是否足够
                if sim_battery < total_energy + CRITICAL_BATTERY:
                    invalid_sequence = True
                    break
                
                # 更新模拟状态
                sim_pos = task['location']
                sim_battery -= total_energy
                
                # 添加到有效序列
                valid_task_sequence.append(task_id)
            
            # 如果任务序列有效，验证终点充电站
            if not invalid_sequence:
                station_id = assignment.get("end_station_id")
                
                # 获取充电站位置
                station_index = None
                station_location = None
                
                if station_id and station_id.startswith('S'):
                    try:
                        station_index = int(station_id[1:])
                        if 0 <= station_index < len(self.stations_all):
                            if isinstance(self.stations_all[station_index], dict) and 'location' in self.stations_all[station_index]:
                                station_location = self.stations_all[station_index]['location']
                            else:
                                station_location = self.stations_all[station_index]
                    except (ValueError, IndexError):
                        station_index = None
                        station_location = None
                
                if station_location:
                    # 检查全局缓存
                    cache_key = (tuple(sim_pos), tuple(station_location))
                    cached_path = get_cached_path(cache_key)
                    
                    if cached_path:
                        path = cached_path
                    else:
                        # 计算到充电站的路径
                        path = a_star(self.grid, sim_pos, station_location, self.height, self.width)
                        if path:
                            set_cached_path(cache_key, path)
                            
                    if path:
                        energy_to_station = (len(path) - 1) * MOVE_COST
                        
                        # 检查是否有足够的电池电量
                        if sim_battery >= energy_to_station + CRITICAL_BATTERY:
                            # 有效的分配
                            valid_assignments[robot_id] = {
                                "task_sequence": valid_task_sequence,
                                "end_station_id": station_id
                            }
                        else:
                            # 电量不足，尝试找到更近的充电站
                            closest_station = self._find_closest_station(sim_pos, sim_battery)
                            if closest_station:
                                valid_assignments[robot_id] = {
                                    "task_sequence": valid_task_sequence,
                                    "end_station_id": f"S{closest_station[0]}"
                                }
                else:
                    # 找不到充电站，尝试找最近的
                    closest_station = self._find_closest_station(sim_pos, sim_battery)
                    if closest_station:
                        valid_assignments[robot_id] = {
                            "task_sequence": valid_task_sequence,
                            "end_station_id": f"S{closest_station[0]}"
                        }
        
        # 如果验证后没有有效分配，回退到简单贪心
        if not valid_assignments:
            print("警告: 没有有效的分配方案，使用贪心策略")
            return self._greedy_fallback()
        
        return valid_assignments
    
    def _find_closest_station(self, position, battery):
        """找到最近的可达充电站
        
        Args:
            position: 当前位置
            battery: 当前电量
            
        Returns:
            tuple: (station_index, station_location)，如果没有则返回None
        """
        closest_station = None
        min_distance = float('inf')
        
        for i, station in enumerate(self.stations_all):
            # 获取充电站位置
            if isinstance(station, dict) and 'location' in station:
                station_loc = station['location']
            elif isinstance(station, (list, tuple)):
                station_loc = station
            else:
                continue
            
            # 首先检查全局缓存
            cache_key = (tuple(position), tuple(station_loc))
            cached_path = get_cached_path(cache_key)
            
            if cached_path:
                path = cached_path
            else:
                # 计算路径
                path = a_star(self.grid, position, station_loc, self.height, self.width)
                if path:
                    set_cached_path(cache_key, path)
                    
            if not path:
                continue
                
            distance = len(path) - 1
            energy_needed = distance * MOVE_COST
            
            if battery >= energy_needed + CRITICAL_BATTERY:
                if distance < min_distance:
                    min_distance = distance
                    closest_station = (i, station_loc)
        
        return closest_station
    
    def _greedy_fallback(self):
        """贪心算法作为备选方案"""
        try:
            # 筛选空闲的机器人
            idle_robots = [r for r in self.robots_all if r.state == "idle"]
            
            # 筛选待处理任务
            pending_tasks = [t for t in self.tasks_all if not t.get('served', False) and t.get('assigned_to') is None]
            
            # 准备充电站数据
            stations_data = [s.get('location') if isinstance(s, dict) else s for s in self.stations_all]
            
            # 调用贪心算法
            return greedy_assignment(
                idle_robots, pending_tasks, self.grid, stations_data, 
                self.height, self.width, self.current_time
            )
        except Exception as e:
            print(f"贪心备选方案失败: {e}")
            return {}

# --- 接口函数(用于与app.py集成) ---
def assign_tasks_optimized_hybrid_spso(robots, tasks, tick, grid, chargers, height, width):
    """优化的混合SPSO调度算法的主接口函数
    
    Args:
        robots: 机器人列表
        tasks: 任务列表
        tick: 当前时间步
        grid: 环境网格
        chargers: 充电站位置列表
        height: 网格高度
        width: 网格宽度
        
    Returns:
        bool: 是否成功分配
    """
    try:
        # 准备充电站数据
        stations = [{"id": f"S{i}", "location": loc} for i, loc in enumerate(chargers)]
        
        # 创建调度器
        dispatcher = HybridSPSODispatcher(
            robots_all=robots,
            tasks_all=tasks,
            stations_all=stations,
            grid_ref=grid,
            current_time=tick
        )
        
        # 运行优化并获取分配结果
        assignments = dispatcher.run_optimization()
        
        # 没有分配结果
        if not assignments:
            return False
        
        # 应用分配结果
        assignments_applied = 0
        
        for robot_id, assignment in assignments.items():
            robot = next((r for r in robots if r.id == robot_id), None)
            # 允许充电状态且电量足够的机器人也可以被分配任务
            if not robot or (robot.state != "idle" and 
                            not (robot.state == "charging" and robot.battery >= MAX_BATTERY * 0.8)):
                continue
            
            task_sequence = assignment.get("task_sequence", [])
            if task_sequence:
                # 分配第一个任务
                task_id = task_sequence[0]
                
                # 尝试多种方式查找任务
                task = None
                for t in tasks:
                    t_id = t.get('task_id', None)
                    if t_id is not None and (str(t_id) == str(task_id)):
                        task = t
                        break
                
                if task and not task.get('served', False) and task.get('assigned_to') is None:
                    try:
                        # 首先检查全局缓存
                        cache_key = (tuple(robot.pos), tuple(task['location']))
                        cached_path = get_cached_path(cache_key)
                        
                        if cached_path:
                            path = cached_path
                        else:
                            path = a_star(grid, robot.pos, task['location'], height, width)
                            if path:
                                set_cached_path(cache_key, path)
                        
                        if path:
                            robot.assign(task, path)
                            assignments_applied += 1
                            print(f"[Tick {tick}] 优化混合SPSO: 机器人 {robot.id} 分配到任务 {task_id}")
                    except Exception as e:
                        print(f"[Tick {tick}] 优化混合SPSO: 机器人 {robot.id} 分配任务 {task_id} 失败: {e}")
            elif assignment.get("end_station_id"):
                # 如果没有任务但有充电站，返回充电
                try:
                    robot.return_to_charge(grid, height, width)
                    assignments_applied += 1
                    print(f"[Tick {tick}] 优化混合SPSO: 机器人 {robot.id} 返回充电")
                except Exception as e:
                    print(f"[Tick {tick}] 优化混合SPSO: 机器人 {robot.id} 返回充电失败: {e}")
        
        return assignments_applied > 0
    except Exception as e:
        print(f"[Tick {tick}] 优化混合SPSO调度器出错: {e}")
        import traceback
        traceback.print_exc()
        return False# optimized_hybrid_spso_algorithm.py