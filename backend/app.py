from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import random

# 导入核心模块
from hungarian import assign_tasks_hungarian
from environment import *
from ppo import PPOAgent

app = Flask(__name__)
CORS(app)

# ========== 全局状态管理 ==========
state = {
    "tick": 0,               # 当前时间步
    "grid": None,            # 地图网格矩阵
    "chargers": [],          # 所有充电桩位置列表
    "tasks": [],             # 所有任务列表
    "robots": [],            # 所有机器人对象列表
    "strategy": "hungarian", # 当前调度策略
    "scale": "medium",       # 当前地图规模
    "WIDTH": 0,              # 地图宽度
    "HEIGHT": 0,             # 地图高度
    "distribution": "uniform", # 停车位分布策略
    "arrival": "poisson"       # 任务到达策略
}

# ========== 初始化地图与系统 ==========
@app.route("/api/init_map")
def init_map():
    try:
        # 解析前端请求参数
        strategy = request.args.get("strategy", "hungarian")
        scale = request.args.get("scale", "medium")
        distribution = request.args.get("distribution", "uniform")
        arrival = request.args.get("arrival", "poisson")

        # 更新状态参数
        state.update({
            "strategy": strategy,
            "scale": scale,
            "distribution": distribution,
            "arrival": arrival
        })

        # 根据规模选择地图和任务参数
        if scale == "small":
            w, h, robots, tasks, groups, obstacles = 15, 15, 5, 10, 2, 5
        elif scale == "large":
            w, h, robots, tasks, groups, obstacles = 35, 35, 12, 30, 7, 15
        else:
            w, h, robots, tasks, groups, obstacles = 25, 25, 8, 20, 5, 10

        chargers_estimate = max(robots, int(robots * 0.7))

        # 初始化地图、充电桩、任务、机器人
        grid = generate_map(w, h, chargers_estimate, groups, obstacles, parking_distribution=distribution)
        chargers = [(i, j) for i in range(h) for j in range(w) if grid[i][j] == CHARGING_STATION]

        if len(chargers) < robots:
            raise ValueError(f"生成充电桩数不足：仅有{len(chargers)}个，要求至少{robots}个。")

        state.update({
            "grid": grid,
            "chargers": chargers,
            "tasks": generate_tasks(grid, chargers, w, h, tasks, MAX_TIME, mode=arrival),
            "robots": [Robot(i, pos, chargers) for i, pos in enumerate(random.sample(chargers, robots))],
            "tick": 0,
            "WIDTH": w,
            "HEIGHT": h,
            "NUM_ROBOTS": robots,
            "TOTAL_TASKS": tasks
        })

        state.pop("ppo_agent", None)
        return jsonify({"message": "初始化完成", "tick": 0})

    except Exception as e:
        print(f"初始化失败: {e}")
        return jsonify({"error": str(e)}), 500

# 判断是否有空闲机器人且电量足够
def should_dispatch(robots):
    return any(r.state in ('idle', 'returning_idle') and r.battery >= LOW_BATTERY_THRESHOLD for r in robots)

@app.route("/api/next_step")
def next_step():
    total_tasks = len(state["tasks"])
    completed_tasks = sum(1 for t in state["tasks"] if t["served"])

    if completed_tasks == total_tasks and total_tasks > 0:
        return jsonify({
            "message": "全部任务已完成",
            "tick": state["tick"],
            "done": True
        })

    state["tick"] += 1

    if state["strategy"] == "hungarian":
        #  仅在有空闲且电量充足的机器人时触发调度器
        if should_dispatch(state["robots"]):
            assign_tasks_hungarian(
                state["robots"], state["tasks"], state["tick"],
                state["grid"], state["chargers"],
                state["HEIGHT"], state["WIDTH"]
            )
            print(f"[Tick {state['tick']}] 调用匈牙利调度器 ")

    '''
    elif state["strategy"] == "ppo":
        if "ppo_agent" not in state:
            state["ppo_agent"] = PPOAgent(actor_path="ppo_actor.h5", critic_path="ppo_critic.h5",
                                           width=state["WIDTH"], height=state["HEIGHT"], num_robots=len(state["robots"]))

        active_tasks = [t for t in state["tasks"] if t['arrival_time'] <= state["tick"] and not t['served']]

        env_state = {
            'grid': state["grid"],
            'robots': np.array([
                [r.pos[0]/state["HEIGHT"], r.pos[1]/state["WIDTH"], r.battery/MAX_BATTERY,
                 1 if r.task else 0, 1 if r.state == 'charging' else 0, r.idle_counter/IDLE_TIMEOUT] for r in state["robots"]
            ], dtype=np.float32),
            'tasks': np.zeros((10, 5), dtype=np.float32),
            'global': np.array([
                state["tick"]/MAX_TIME,
                sum(1 for t in active_tasks)/state["TOTAL_TASKS"],
                sum(1 for t in state["tasks"] if t['served'])/state["TOTAL_TASKS"]
            ], dtype=np.float32)
        }

        for i, task in enumerate(active_tasks[:10]):
            env_state['tasks'][i] = np.array([
                task['location'][0]/state["HEIGHT"],
                task['location'][1]/state["WIDTH"],
                task['initial_energy']/100,
                task['required_energy']/100,
                (state["tick"] - task['arrival_time'])/MAX_TIME
            ])

        actions = state["ppo_agent"].choose_action(env_state)

        for i, action in enumerate(actions):
            r = state["robots"][i]
            if r.task or r.state == 'charging':
                continue
            if action == 10:
                r.return_to_charge(state["grid"], state["HEIGHT"], state["WIDTH"])
            elif action == 11:
                continue
            elif action < len(active_tasks):
                task = active_tasks[action]
                path = a_star(state["grid"], r.pos, task['location'], state["HEIGHT"], state["WIDTH"])
                if path:
                    r.assign(task, path)
        '''

    for robot in state["robots"]:
        robot.step(state["grid"], state["tick"], state["HEIGHT"], state["WIDTH"])

    return jsonify({"message": "调度推进一帧", "tick": state["tick"], "done": False})

# ========== 获取当前状态快照 ==========
@app.route("/api/get_state")
def get_state():
    energy = sum(r.energy_used for r in state["robots"])
    done = [t for t in state["tasks"] if t["served"]]
    delay = sum(t["start_time"] - t["arrival_time"] for t in done if t["start_time"] is not None) / len(done) if done else 0
    # 所有已经出现的任务
    appeared = [t for t in state["tasks"] if t["arrival_time"] <= state["tick"]]
    # 当前已出现任务中，成功完成（未过期）的比例
    completed = [t for t in appeared if t["served"] and not t.get("expired")]
    # 完成率（去除过期任务）
    completionRate = 100 * len(completed) / len(appeared) if appeared else 0

    return jsonify({
        "tick": state["tick"],
        "grid": state["grid"].tolist() if state["grid"] is not None else [],
        "robots": [{
            "id": r.id,
            "pos": r.pos,
            "battery": r.battery,
            "state": r.state,
            "task": r.task["task_id"] if r.task else None
        } for r in state["robots"]],
        "tasks": state["tasks"],
        "strategy": state["strategy"],
        "scale": state["scale"],
        "distribution": state["distribution"],
        "arrival": state["arrival"],
        "metrics": {
            "energy": round(energy, 2),
            "delay": round(delay, 2), # 从任务生成到首次被机器人服务的平均时间
            "completed": len(done),
            "completionRate": round(completionRate, 1)
        }
    })

if __name__ == '__main__':
    app.run(debug=True, port=5050)
