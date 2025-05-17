# environment.py （支持规模动态切换 + 三阶段充电）
import math
import numpy as np
import random
from collections import deque
import heapq
from sklearn.datasets import make_blobs

# ================= 参数设置 =================
PARKING_GROUP_SIZE = 6
MAX_TIME = 100
INTERVAL = 500
ROBOT_SPEED = 1
MAX_BATTERY = 100
MOVE_COST = 1
CHARGE_RATE = 5
LOW_BATTERY_THRESHOLD = 15
CHARGE_TRANSFER = 5
IDLE_TIMEOUT = 3  # 等待 3tick 没有任务就去充电，缩短等待触发时间
# 但鬼打墙，gpt 建议我延长，避免机器人刚 idle 就返航

EMPTY, OBSTACLE, PARKING_SPOT, CHARGING_STATION = 0, 1, 2, 3
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# ========== 锂电池三阶段充电函数 ==========
def battery_charging_curve(current_percent, max_rate=5.0):
    if current_percent < 0.8:
        return max_rate  # 恒流
    elif current_percent < 0.98:
        slope = (max_rate - 0.5) / (0.98 - 0.8)
        return max_rate - slope * (current_percent - 0.8)  # 恒压
    else:
        return 0.1  # 滴流

# ================= 地图生成 =================
def place_building(grid, top_left, size, height, width):
    """
    在指定起点放置矩形障碍物
    """
    h, w = size
    for i in range(top_left[0], min(top_left[0] + h, height)):
        for j in range(top_left[1], min(top_left[1] + w, width)):
            grid[i][j] = OBSTACLE

def generate_realistic_obstacles(grid, height, width, max_obstacles):
    """
    在地图上随机生成若干个障碍物，模拟真实建筑物布局
    """
    attempts = 0
    while attempts < max_obstacles:
        h, w = random.randint(3, 6), random.randint(4, 8)
        i, j = random.randint(0, height - h - 1), random.randint(0, width - w - 1)
        if np.all(grid[i:i + h, j:j + w] == EMPTY):
            place_building(grid, (i, j), (h, w), height, width)
            attempts += 1

# 三种分布策略
def generate_parking_near_edges_or_buildings(grid, height, width, parking_groups, distribution='clustered'):
    """
    靠近障碍物或边界生成停车位，增加地图复杂性
    """
    candidate_positions = []
    for i in range(height):
        for j in range(width):
            if grid[i][j] == EMPTY:
                for dx, dy in DIRECTIONS:
                    ni, nj = i + dx, j + dy
                    if not (0 <= ni < height and 0 <= nj < width) or grid[ni][nj] == OBSTACLE:
                        candidate_positions.append((i, j))
                        break

    selected = []

    if distribution == 'uniform':
        attempts = 0
        while len(selected) < parking_groups * PARKING_GROUP_SIZE and attempts < 1000:
            base = random.choice(candidate_positions)
            region = set()
            queue = [base]
            while queue and len(region) < PARKING_GROUP_SIZE:
                curr = queue.pop()
                if curr in region or curr not in candidate_positions:
                    continue
                region.add(curr)
                for dx, dy in DIRECTIONS:
                    ni, nj = curr[0] + dx, curr[1] + dy
                    if 0 <= ni < height and 0 <= nj < width and (ni, nj) in candidate_positions:
                        queue.append((ni, nj))
            if len(region) == PARKING_GROUP_SIZE:
                selected.extend(region)
                for pt in region:
                    candidate_positions.remove(pt)
            attempts += 1

    elif distribution == 'clustered':
        total_spots = parking_groups * PARKING_GROUP_SIZE
        # 每簇大致一半左右，±1
        half = PARKING_GROUP_SIZE // 2
        low, high = max(1, half - 1), min(PARKING_GROUP_SIZE, half + 1)

        # 1. 全图所有空格（不再只限障碍边缘）
        empties = [(i, j) for i in range(height) for j in range(width) if grid[i][j] == EMPTY]

        # 2. Farthest‐point Sampling 挑 seed，使种子彼此最远
        seeds = [random.choice(empties)]
        while len(seeds) < parking_groups:
            # 对每个候选点，算到已有 seeds 的最小距离，取最大的那个
            next_seed = max(
                empties,
                key=lambda p: min((p[0] - s[0]) ** 2 + (p[1] - s[1]) ** 2 for s in seeds)
            )
            seeds.append(next_seed)

        selected = set()
        # 3. 围绕每个 seed 生长一个小团
        for sx, sy in seeds:
            if len(selected) >= total_spots:
                break
            size = random.randint(low, high)
            size = min(size, total_spots - len(selected))

            region = {(sx, sy)}
            attempts = 0
            while len(region) < size and attempts < 200:
                bx, by = random.choice(tuple(region))
                dx, dy = random.choice(DIRECTIONS)
                nx, ny = bx + dx, by + dy
                if (0 <= nx < height and 0 <= ny < width
                        and grid[nx][ny] == EMPTY
                        and (nx, ny) not in selected
                        and (nx, ny) not in region):
                    region.add((nx, ny))
                attempts += 1

            # 如果成功成团，就写入
            if len(region) == size:
                for x, y in region:
                    grid[x][y] = PARKING_SPOT
                    selected.add((x, y))

        # 4. 优先在已有簇的邻居里补齐剩余，避免孤点
        remaining = total_spots - len(selected)
        free = {p for p in empties if p not in selected}
        while remaining > 0 and free:
            # 找所有已占点的空格邻居
            nbrs = {
                (x + dx, y + dy)
                for (x, y) in selected
                for dx, dy in DIRECTIONS
                if (x + dx, y + dy) in free
            }
            if nbrs:
                pt = random.choice(list(nbrs))
            else:
                pt = random.choice(list(free))
            grid[pt[0]][pt[1]] = PARKING_SPOT
            selected.add(pt)
            free.remove(pt)
            remaining -= 1

        ''' 原clustered
        elif distribution == 'clustered':
        points, _ = make_blobs(n_samples=parking_groups, centers=random.randint(2, 4), cluster_std=1.5,
                               center_box=(0, width))
        points = [(int(y), int(x)) for x, y in points]
        for cx, cy in points:
            region = []
            attempts = 0
            while len(region) < PARKING_GROUP_SIZE and attempts < 100:
                dx, dy = random.choice([(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)])
                px, py = min(max(cx + dx, 0), height - 1), min(max(cy + dy, 0), width - 1)
                if grid[px][py] == EMPTY and (px, py) not in region:
                    region.append((px, py))
                attempts += 1
            if len(region) == PARKING_GROUP_SIZE:
                selected.extend(region)
        selected = []
        required_total = parking_groups * PARKING_GROUP_SIZE
        '''

        return

    elif distribution == 'mixed':
        half = parking_groups // 2
        # uniform half
        generate_parking_near_edges_or_buildings(grid, height, width, half, distribution='uniform')
        # clustered half
        generate_parking_near_edges_or_buildings(grid, height, width, parking_groups - half, distribution='clustered')
        return  # 直接递归添加，已在 grid 上完成

    else:
        raise ValueError("Invalid parking distribution")

    for x, y in selected:
        grid[x][y] = PARKING_SPOT

def generate_map(width, height, num_chargers, parking_groups, max_obstacles, parking_distribution='uniform'):
    """
    生成完整地图，包括障碍、停车位、充电桩
    """
    grid = np.zeros((height, width), dtype=int)
    generate_realistic_obstacles(grid, height, width, max_obstacles)
    generate_parking_near_edges_or_buildings(grid, height, width, parking_groups, distribution=parking_distribution)

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
        source = edge_candidates
    else:
        source = [(i, j) for i in range(height) for j in range(width) if grid[i][j] == EMPTY]
    step = max(1, len(source) // num_chargers)
    selected = [source[i] for i in range(0, len(source), step)][:num_chargers]

    for i, j in selected:
        grid[i][j] = CHARGING_STATION

    return grid

# ========== 任务到达时间生成函数 ==========
def generate_disclosure_times(num_tasks, planning_horizon, mode='poisson'):
    if mode == 'poisson':
        LAMBDA = num_tasks / planning_horizon
        t, times = 0, []
        while len(times) < num_tasks:
            t += np.random.exponential(1 / LAMBDA)
            times.append(round(t, 2))
        return times[:num_tasks]
    elif mode == 'uniform':
        return list(np.round(np.random.uniform(0, planning_horizon, size=num_tasks), 2))
    elif mode == 'normal':
        times = np.random.normal(loc=planning_horizon / 2, scale=planning_horizon / 4, size=num_tasks)
        return list(np.round(np.clip(times, 0, planning_horizon), 2))
    else:
        raise ValueError("Invalid disclosure distribution mode")

# ================= 路径与任务生成 =================
def a_star(grid, start, goal, height, width):
    """
    使用A*搜索算法，寻找从start到goal的最短路径（曼哈顿距离启发）
    """
    def h(a, b): return abs(a[0] - b[0]) + abs(a[1] - b[1])
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

def generate_tasks(grid, chargers, width, height, total_tasks, max_time, mode='poisson'):
    """
    生成随机到达的任务，确保每个任务可达且机器人可以完成
    """
    spots = [(i, j) for i in range(height) for j in range(width) if grid[i][j] == PARKING_SPOT]
    if not spots:
        print("[Debug] ⚠️ 当前地图上未成功生成任何车位")
    tasks = []
    arrival_times = generate_disclosure_times(total_tasks, max_time, mode=mode)

    task_id = 0
    for t in arrival_times:
        attempts, success = 0, False
        while not success and attempts < 50:
            attempts += 1

            # 1. 当前时刻，哪些车位已被任务占用？
            occupied = {task['location'] for task in tasks if not task['served']}

            # 2. 从空闲停车位中挑选位置
            available_spots = [s for s in spots if s not in occupied]
            if not available_spots:
                break  # 没有位置了

            loc = random.choice(available_spots)

            # 3. 生成任务电量
            ini, req = random.uniform(10, 50), random.uniform(80, 100)

            # 4. 判断这个任务是否是可行的（从任何一个充电桩出发都能完成后回桩）
            feasible = False
            for ch in chargers:
                to_task = path_cost(grid, ch, loc, height, width)
                to_charger = min(path_cost(grid, loc, c, height, width) for c in chargers)
                energy_needed = (to_task + to_charger) * MOVE_COST + (req - ini)
                if energy_needed <= MAX_BATTERY:
                    feasible = True
                    break

            # 5. 如果可行就添加任务
            if feasible:
                departure_time = t + 10 + 1.5 * (req - ini)  # 等待时间可调
                tasks.append({
                    'task_id': task_id,
                    'arrival_time': round(t, 2),
                    'location': loc,
                    'initial_energy': ini,
                    'required_energy': req,
                    'received_energy': 0,
                    'served': False,
                    'expired': False,  # ✅ 新增字段，标记是否因超时未被完成
                    'assigned_to': None,
                    'start_time': None,
                    'departure_time': round(departure_time, 2)  # ✅ 新增
                })
                task_id += 1
                success = True

        if not success:
            break  # 当前时刻没法安排任务，放弃这个任务

    return tasks

# ================= 机器人类 =================
class Robot:
    def __init__(self, i, pos, chargers):
        self.id = i
        self.pos = pos
        self.path = deque()
        self.task = None
        self.battery = MAX_BATTERY
        self.state = 'idle'
        self.idle_counter = 0
        self.chargers = chargers
        self.energy_used = 0  # 机器人累计消耗能量

    def assign(self, task, path):
        self.task = task
        self.path = deque(path)
        self.state = 'to_task'
        task['assigned_to'] = self.id
        self.idle_counter = 0


    def return_to_charge(self, grid, height, width):
        charger = min(self.chargers, key=lambda c: path_cost(grid, self.pos, c, height, width))
        self.path = deque(a_star(grid, self.pos, charger, height, width))
        self.task = None
        self.state = 'returning_idle'
        self.idle_counter = 0

    def step(self, grid, t, height, width):
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
                self.energy_used += MOVE_COST
            else:
                self.state = 'charging'
            return

        self.idle_counter = 0

        if self.task and self.task['served']:
            self.task = None
            self.state = 'idle'
            return

        if self.state == 'charging':
            charge_percent = self.battery / MAX_BATTERY
            charge_rate = battery_charging_curve(charge_percent, max_rate=CHARGE_RATE)
            self.battery = min(MAX_BATTERY, self.battery + charge_rate)
            if self.battery >= MAX_BATTERY:
                self.state = 'idle'  # ✅ 充满电后切换为可调度状态
            return

        if self.path:
            self.pos = self.path.popleft()
            self.battery -= MOVE_COST
            self.energy_used += MOVE_COST

        elif self.task and not self.task['served']:
            # 如果 PPO 策略提前安排机器人提前到达任务点，但任务还没到达
            # 1.该任务的等待时间不会被计入
            '''
            if self.task['start_time'] is None and t >= self.task['arrival_time']: # 检查当前 tick t 是否 ≥ 任务到达时间
                self.task['start_time'] = t  # 机器人此时已经抵达任务地点并开始充电
            '''
            # 2.该任务的等待时间记为0
            if self.task['start_time'] is None:
                self.task['start_time'] = max(t, self.task['arrival_time'])
            current_energy = self.task['initial_energy'] + self.task['received_energy']
            target = self.task['required_energy']
            task_percent = current_energy / target
            task_charge_rate = battery_charging_curve(task_percent, max_rate=CHARGE_TRANSFER)
            need = target - current_energy
            give = min(task_charge_rate, need, self.battery)
            self.task['received_energy'] += give
            self.battery -= give
            self.energy_used += give
            if self.task['initial_energy'] + self.task['received_energy'] >= self.task['required_energy']:
                self.task['served'] = True
                self.task = None
                self.state = 'idle'
