# environment.py （支持规模动态切换）

import numpy as np
import random
from collections import deque
import heapq

# ================= 参数设置 =================
# 部分规模切换相关的参数值需在接口文件app.py中作为参数传入，并非在此处作为全局变量
# WIDTH、HEIGHT、NUM_ROBOTS、TOTAL_TASKS、LAMBDA = TOTAL_TASKS / MAX_TIME在task函数中动态计算

# 地图元素数量设定
# NUM_PARKING_GROUPS = 5  # 停车位组数
PARKING_GROUP_SIZE = 6  # 每组停车位大小

# 任务和时间参数
MAX_TIME = 100  # 总时间步数（泊松到达基准）
INTERVAL = 500  # 调度间隔步数（备用）

# 机器人参数
ROBOT_SPEED = 1
MAX_BATTERY = 100
MOVE_COST = 1
CHARGE_RATE = 5
LOW_BATTERY_THRESHOLD = 15
CHARGE_TRANSFER = 5
IDLE_TIMEOUT = 10

# 地图元素编码
EMPTY, OBSTACLE, PARKING_SPOT, CHARGING_STATION = 0, 1, 2, 3

# 4个方向
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# ================= 地图生成 =================
def place_building(grid, top_left, size, height, width):
    """
    在指定起点放置矩形障碍物
    """
    h, w = size
    for i in range(top_left[0], min(top_left[0] + h, height)):
        for j in range(top_left[1], min(top_left[1] + w, width)):
            grid[i][j] = OBSTACLE


def generate_realistic_obstacles(grid, height, width, max_obstacles=12):
    """
    在地图上随机生成障碍物，最大数量可控
    """
    attempts = 0
    while attempts < max_obstacles:
        h, w = random.randint(3, 6), random.randint(4, 8)
        i, j = random.randint(0, height - h - 1), random.randint(0, width - w - 1)
        if np.all(grid[i:i + h, j:j + w] == EMPTY):
            place_building(grid, (i, j), (h, w), height, width)
            attempts += 1

def generate_parking_near_edges_or_buildings(grid, height, width, num_parking_groups=5):
    """
    生成停车位，最大组数可控
    """
    count, placements = 0, 0
    while placements < num_parking_groups and count < 100:
        count += 1
        orientation = random.choice(['horizontal', 'vertical'])
        length = PARKING_GROUP_SIZE
        i, j = random.randint(0, height - 1), random.randint(0, width - 1)

        if orientation == 'horizontal':
            if j + length >= width: continue
            region = [(i, j + offset) for offset in range(length)]
        else:
            if i + length >= height: continue
            region = [(i + offset, j) for offset in range(length)]

        if all(grid[x][y] == EMPTY for x, y in region):
            near_building_or_edge = False
            for x, y in region:
                for dx, dy in DIRECTIONS:
                    nx, ny = x + dx, y + dy
                    if not (0 <= nx < height and 0 <= ny < width) or grid[nx][ny] == OBSTACLE:
                        near_building_or_edge = True
            if near_building_or_edge:
                for x, y in region:
                    grid[x][y] = PARKING_SPOT
                placements += 1

def generate_map(width, height, num_chargers, num_parking_groups=5, max_obstacles=12):
    """
    动态参数的完整地图生成
    """
    grid = np.zeros((height, width), dtype=int)
    generate_realistic_obstacles(grid, height, width, max_obstacles)
    generate_parking_near_edges_or_buildings(grid, height, width, num_parking_groups)

    edge_candidates = set()
    for i in range(height):
        for j in range(width):
            if grid[i][j] == PARKING_SPOT:
                for dx, dy in DIRECTIONS:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < height and 0 <= nj < width and grid[ni][nj] == EMPTY:
                        edge_candidates.add((ni, nj))
    edge_candidates = list(edge_candidates)

    if len(edge_candidates) >= num_chargers:
        selected_positions = random.sample(edge_candidates, num_chargers)
    else:
        all_empty = [(i, j) for i in range(height) for j in range(width) if grid[i][j] == EMPTY]
        if len(all_empty) < num_chargers:
            raise ValueError(f"地图空间不足，无法放置{num_chargers}个充电桩。")
        selected_positions = random.sample(all_empty, num_chargers)

    for i, j in selected_positions:
        grid[i][j] = CHARGING_STATION

    return grid


# ================= 路径与任务生成 =================
def a_star(grid, start, goal, height, width):
    """
    使用A*搜索算法，寻找从start到goal的最短路径（曼哈顿距离启发）
    """
    def h(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open = [(h(start, goal), 0, start)]
    came, g = {}, {start: 0}

    while open:
        _, cost, curr = heapq.heappop(open)
        if curr == goal:
            path = []
            while curr in came:
                path.append(curr)
                curr = came[curr]
            return path[::-1]

        for dx, dy in DIRECTIONS:
            nxt = (curr[0] + dx, curr[1] + dy)
            if 0 <= nxt[0] < height and 0 <= nxt[1] < width and grid[nxt[0]][nxt[1]] != OBSTACLE:
                tg = g[curr] + 1
                if nxt not in g or tg < g[nxt]:
                    came[nxt] = curr
                    g[nxt] = tg
                    heapq.heappush(open, (tg + h(nxt, goal), tg, nxt))

    return []


def path_cost(grid, start, goal, height, width):
    """
    返回start到goal路径长度，无法到达返回无穷大
    """
    path = a_star(grid, start, goal, height, width)
    return len(path) if path else float('inf')


def is_feasible(robot, task, grid, chargers, height, width):
    """
    检查机器人是否有足够能量完成任务并到最近充电桩
    """
    to_task = path_cost(grid, robot.pos, task['location'], height, width)
    to_charger = min(path_cost(grid, task['location'], ch, height, width) for ch in chargers)
    total_cost = (to_task + to_charger) * MOVE_COST + (task['required_energy'] - task['initial_energy'])
    return total_cost <= robot.battery


def generate_tasks(grid, chargers, width, height, total_tasks, max_time):
    """
    生成随机到达的任务，确保每个任务可达且机器人可以完成
    """
    LAMBDA = total_tasks / max_time  # 动态计算泊松到达率
    spots = [(i, j) for i in range(height) for j in range(width) if grid[i][j] == PARKING_SPOT]

    # 动态调整最大任务量
    total_tasks = min(total_tasks, len(spots))

    used, tasks, t = set(), [], 0

    while len(tasks) < total_tasks:
        t += np.random.exponential(1 / LAMBDA)
        avail = list(set(spots) - used)
        if not avail:
            break
        loc = random.choice(avail)
        used.add(loc)

        ini, req = random.uniform(10, 50), random.uniform(80, 100)

        if all(path_cost(grid, ch, loc, height, width) + path_cost(grid, loc, min(chargers, key=lambda c: path_cost(grid, loc, c, height, width)), height, width) + (req - ini) // MOVE_COST <= MAX_BATTERY for ch in chargers):
            tasks.append({
                'task_id': len(tasks),
                'arrival_time': round(t, 2),
                'location': loc,
                'initial_energy': ini,
                'required_energy': req,
                'received_energy': 0,
                'served': False,
                'assigned_to': None,
                'start_time': None
            })
    return tasks

# ================= 机器人类 =================
class Robot:
    """
    机器人对象，管理位置、任务、电量、状态机
    """
    def __init__(self, i, pos, chargers):
        self.id = i
        self.pos = pos
        self.path = deque()
        self.task = None
        self.battery = MAX_BATTERY
        self.state = 'idle'
        self.idle_counter = 0
        self.chargers = chargers

    def assign(self, task, path):
        """
        分配任务并设定行进路径
        """
        self.task = task
        self.path = deque(path)
        self.state = 'to_task'
        task['assigned_to'] = self.id
        self.idle_counter = 0

    def return_to_charge(self, grid, height, width):
        """
        返回最近充电桩充电
        """
        charger = min(self.chargers, key=lambda c: path_cost(grid, self.pos, c, height, width))
        self.path = deque(a_star(grid, self.pos, charger, height, width))
        self.task = None
        self.state = 'returning_idle'
        self.idle_counter = 0

    def step(self, grid, t, height, width):
        """
        每个时间步执行动作（移动、充电、服务任务等）
        """
        if self.battery <= 0:
            self.state = 'dead'
            return

        if self.state == 'idle':
            self.idle_counter += 1
            if self.idle_counter >= IDLE_TIMEOUT:
                self.return_to_charge(grid, height, width)
            return

        if self.state == 'returning_idle':
            self.idle_counter += 1
            if self.path:
                self.pos = self.path.popleft()
                self.battery -= MOVE_COST
            else:
                self.state = 'charging'
            return

        self.idle_counter = 0

        if self.task and self.task['served']:
            self.task = None
            self.state = 'idle'
            return

        if self.state == 'charging':
            self.battery = min(MAX_BATTERY, self.battery + CHARGE_RATE)
            if self.battery >= MAX_BATTERY:
                self.state = 'idle'
            return

        if self.path:
            self.pos = self.path.popleft()
            self.battery -= MOVE_COST
        elif self.task and not self.task['served']:
            if self.task['start_time'] is None:
                self.task['start_time'] = t
            need = self.task['required_energy'] - self.task['initial_energy'] - self.task['received_energy']
            give = min(CHARGE_TRANSFER, need, self.battery)
            self.task['received_energy'] += give
            self.battery -= give
            if self.task['initial_energy'] + self.task['received_energy'] >= self.task['required_energy']:
                self.task['served'] = True
                self.task = None
                self.state = 'idle'
