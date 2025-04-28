# backend/app.py

from flask import Flask, jsonify, request
from flask_cors import CORS

# 导入匈牙利调度器和 PPO推理模块
from hungarian import assign_tasks_hungarian
from environment import *  # 地图、机器人、任务生成器
from ppo import PPOAgent

app = Flask(__name__)
CORS(app)

# ================= 后端状态管理与接口 =================
state = {
    "tick": 0,
    "grid": None,
    "chargers": [],
    "tasks": [],
    "robots": [],
    "strategy": "hungarian",  # 当前策略
    "scale": "medium",          # 当前规模
    "WIDTH": 0,
    "HEIGHT":0

}

@app.route("/api/init_map")
def init_map():
    try:
        # 接收前端传参：策略与规模
        strategy = request.args.get("strategy", "hungarian")
        scale = request.args.get("scale", "medium")

        # 保存到state
        state["strategy"] = strategy
        state["scale"] = scale

        # 根据scale动态调整参数
        if scale == "small":
            w, h, robots, tasks = 15, 15, 5, 10
            parking_groups = 2   # 小地图停车位组数减少
            max_obstacles = 5    # 小地图障碍物数量减少
        elif scale == "medium":
            w, h, robots, tasks = 25, 25, 8, 20
            parking_groups = 5   # 中等地图正常停车位
            max_obstacles = 10   # 中等地图正常障碍物
        elif scale == "large":
            w, h, robots, tasks = 35, 35, 12, 30
            parking_groups = 7   # 大地图停车位增加
            max_obstacles = 15   # 大地图障碍物更多
        else:
            w, h, robots, tasks = 25, 25, 8, 20
            parking_groups = 5
            max_obstacles = 10

        # 预估充电桩数量（但最终以实际生成为准）
        chargers_estimate = max(robots, robots * 7 // 10)

        # 初始化地图与充电桩
        grid = generate_map(w, h, chargers_estimate, parking_groups, max_obstacles)
        chargers = [(i, j) for i in range(h) for j in range(w) if grid[i][j] == CHARGING_STATION]

        # 校验充电桩数量是否足够
        if len(chargers) < robots:
            raise ValueError(f"生成充电桩数不足：仅有{len(chargers)}个，要求至少{robots}个。")

        # 保存地图和充电桩到state
        state["grid"] = grid
        state["chargers"] = chargers

        # 初始化任务
        state["tasks"] = generate_tasks(grid, chargers, w, h, tasks, MAX_TIME)

        # 随机在充电桩位置抽取机器人初始位置
        state["robots"] = [Robot(i, pos, chargers) for i, pos in enumerate(random.sample(chargers, robots))]

        # 初始化tick
        state["tick"] = 0

        # 清空旧PPO代理（如果有）
        state.pop("ppo_agent", None)

        # 保存基本参数供后续推理
        state["WIDTH"] = w
        state["HEIGHT"] = h
        state["NUM_ROBOTS"] = robots
        state["TOTAL_TASKS"] = tasks

        return jsonify({"message": "初始化完成", "tick": state["tick"]})

    except Exception as e:
        print(f"初始化失败: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/next_step")
def next_step():
    total_tasks = len(state["tasks"])
    completed_tasks = sum(1 for t in state["tasks"] if t["served"])

    # ✅ 在推进tick之前，先判断任务是否全部完成
    if completed_tasks == total_tasks and total_tasks > 0:
        return jsonify({
            "message": "全部任务已完成",
            "tick": state["tick"],
            "done": True
        })

    # 如果还有未完成任务，正常推进一帧
    state["tick"] += 1

    if state["strategy"] == "hungarian":
        # 匈牙利调度策略
        assign_tasks_hungarian(state["robots"], state["tasks"], state["tick"], state["grid"], state["chargers"], state["WIDTH"], state["HEIGHT"])

    elif state["strategy"] == "ppo":
        # PPO推理策略
        if "ppo_agent" not in state:
            state["ppo_agent"] = PPOAgent(actor_path="ppo_actor.h5", critic_path="ppo_critic.h5",
                                           width=state["WIDTH"], height=state["HEIGHT"], num_robots=len(state["robots"]))

        # 构建PPO输入状态
        env_state = {
            'grid': state["grid"],
            'robots': np.array([
                [r.pos[0]/state["HEIGHT"], r.pos[1]/state["WIDTH"], r.battery/MAX_BATTERY,
                 1 if r.task else 0, 1 if r.state == 'charging' else 0, r.idle_counter/IDLE_TIMEOUT] for r in state["robots"]
            ], dtype=np.float32),
            'tasks': np.zeros((10, 5), dtype=np.float32),
            'global': np.array([
                state["tick"]/MAX_TIME,
                sum(1 for t in state["tasks"] if t['arrival_time'] <= state["tick"] and not t['served'])/state["TOTAL_TASKS"],
                sum(1 for t in state["tasks"] if t['served'])/state["TOTAL_TASKS"]
            ], dtype=np.float32)
        }

        active_tasks = [t for t in state["tasks"] if t['arrival_time'] <= state["tick"] and not t['served']]
        for i, task in enumerate(active_tasks[:10]):
            env_state['tasks'][i] = np.array([
                task['location'][0]/state["HEIGHT"],
                task['location'][1]/state["WIDTH"],
                task['initial_energy']/100,
                task['required_energy']/100,
                (state["tick"] - task['arrival_time'])/MAX_TIME
            ])

        actions = state["ppo_agent"].choose_action(env_state)

        for robot_id, action in enumerate(actions):
            robot = state["robots"][robot_id]
            if robot.task or robot.state == 'charging':
                continue
            if action == 10:
                robot.return_to_charge(state["grid"], state["HEIGHT"], state["WIDTH"])
            elif action == 11:
                pass
            elif action < len(active_tasks):
                task = active_tasks[action]
                path = a_star(state["grid"], robot.pos, task['location'], state["HEIGHT"], state["WIDTH"])
                if path:
                    robot.assign(task, path)

    for robot in state["robots"]:
        robot.step(state["grid"], state["tick"], state["HEIGHT"], state["WIDTH"])

    return jsonify({
        "message": "调度推进一帧",
        "tick": state["tick"],
        "done": False
    })

# 获取当前状态（快照）
@app.route("/api/get_state")
def get_state():
    total_energy_used = sum(MAX_BATTERY - r.battery for r in state["robots"])

    completed_tasks = [t for t in state["tasks"] if t["served"]]
    if completed_tasks:
        avg_delay = sum(t["start_time"] - t["arrival_time"] for t in completed_tasks if t["start_time"] is not None) / len(completed_tasks)
    else:
        avg_delay = 0

    completed_count = len(completed_tasks)

    return jsonify({
        "tick": state["tick"],
        "grid": state["grid"].tolist() if state["grid"] is not None else [],
        "robots": [
            {
                "id": r.id,
                "pos": r.pos,
                "battery": r.battery,
                "state": r.state,
                "task": r.task["task_id"] if r.task else None
            } for r in state["robots"]
        ],
        "tasks": state["tasks"],
        "strategy": state["strategy"],
        "scale": state["scale"],
        "metrics": {
            "energy": round(total_energy_used, 2),
            "delay": round(avg_delay, 2),
            "completed": completed_count
        }
    })

if __name__ == '__main__':
    app.run(debug=True, port=5050)