import numpy as np
from scipy.optimize import linear_sum_assignment
from environment import a_star, path_cost, is_feasible
from environment import LOW_BATTERY_THRESHOLD, CHARGE_TRANSFER

# ================= 匈牙利调度器 =================

'''
使用匈牙利算法为当前空闲机器人分配未完成的任务。

参数：
    robots: 机器人列表
    tasks: 任务列表
    time: 当前时间步
    grid: 当前地图网格
    chargers: 充电桩位置列表
    height: 地图高度
    width: 地图宽度

流程：
    - 找出所有等待中的任务（到达且未被服务且未分配）。
    - 找出所有处于空闲或返回空闲状态且电量充足的机器人。
    - 构建机器人与任务之间的代价矩阵（路径成本 + 能量成本 + 等待时间惩罚）。
    - 使用线性求解器（匈牙利算法）最优匹配机器人与任务。
    - 给机器人分配任务并规划前往路径。
'''
def assign_tasks_hungarian(robots, tasks, time, grid, chargers, height, width):
    waiting_tasks = [t for t in tasks if not t['served'] and t['arrival_time'] <= time and t['assigned_to'] is None]
    idle_robots = [r for r in robots if r.state in ('idle', 'returning_idle') and r.battery >= LOW_BATTERY_THRESHOLD]

    if not waiting_tasks or not idle_robots:
        return

    cost_matrix = np.full((len(idle_robots), len(waiting_tasks)), np.inf)

    for i, robot in enumerate(idle_robots):
        for j, task in enumerate(waiting_tasks):
            if is_feasible(robot, task, grid, chargers, height, width):
                wait_time = max(0, time - task['arrival_time'])
                lam = 10  # 可调参数，控制等待权重,单调递增，但上限为 -λ,调度系统“逐步”提升等待任务的优先级，但不希望它压制所有其他新任务。
                wait_penalty = -lam * (wait_time / (1 + wait_time))
                move_cost = path_cost(grid, robot.pos, task['location'], height, width)
                energy_cost = (task['required_energy'] - task['initial_energy']) #/ CHARGE_TRANSFER
                total_cost = move_cost + energy_cost + wait_penalty
                cost_matrix[i][j] = total_cost

    if not np.isfinite(cost_matrix).any():
        return

    try:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
    except ValueError:
        return

    for i, j in zip(row_ind, col_ind):
        if np.isinf(cost_matrix[i][j]):
            continue
        robot = idle_robots[i]
        task = waiting_tasks[j]
        path = a_star(grid, robot.pos, task['location'], height, width)
        if path:
            robot.assign(task, path)


