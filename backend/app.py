from flask import Flask, jsonify, request
from flask_cors import CORS
from statistics import variance
import math
import numpy as np
import random

# 导入核心模块
from hungarian import assign_tasks_hungarian
from environment import *
from ppo_custom import PPOAgent
from spso_algorithm import assign_tasks_optimized_hybrid_spso

app = Flask(__name__)
CORS(app)

# ========== 全局状态管理 ==========
state = {
    "tick": 0,  # 当前时间步
    "grid": None,  # 地图网格矩阵
    "chargers": [],  # 所有充电桩位置列表
    "tasks": [],  # 所有任务列表
    "robots": [],  # 所有机器人对象列表
    "strategy": "hungarian",  # 当前调度策略
    "scale": "medium",  # 当前地图规模
    "WIDTH": 0,  # 地图宽度
    "HEIGHT": 0,  # 地图高度
    "distribution": "uniform",  # 停车位分布策略
    "arrival": "poisson"  # 任务到达策略
}


# ========== 初始化地图与系统 ==========
@app.route("/api/init_map")
def init_map():
    try:
        # 解析前端请求参数
        strategy = request.args.get("strategy", "hungarian")
        scale = request.args.get("scale", "medium")
        distribution = request.args.get("distribution", "uniform")
        arrival = request.args.get("arrival_mode", "poisson")

        # 策略名称标准化
        if strategy in ["s_pso_d", "spso_d", "spso"]:
            strategy = "hspso"  # 统一为 spso

        # 更新状态参数
        state.update({
            "strategy": strategy,
            "scale": scale,
            "distribution": distribution,
            "arrival": arrival
        })

        # 根据规模选择地图和任务参数
        if scale == "small":
            w, h = 25, 25
            task_density = 0.07  # 44
            robot_density = 0.020  # 13
            group_density = 0.012  # 8
        elif scale == "medium":
            w, h = 35, 35
            task_density = 0.06  # 74
            robot_density = 0.017  # 21
            group_density = 0.011  # 13
        elif scale == "large":
            w, h = 40, 40
            task_density = 0.05  # 80
            robot_density = 0.015  # 24
            group_density = 0.010  # 16
        else:
            raise ValueError("Invalid scale (use 'small', 'medium', 'large')")

        area = w * h

        # 参数计算（四舍五入，确保整数）
        tasks = round(area * task_density)
        robots = round(area * robot_density)
        groups = round(area * group_density)
        obstacles = round(area * 0.15 / 9)  # 每块障碍大约3x3 = 9格

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
    return any(
        (r.state in ('idle', 'returning_idle') or
         (r.state == 'charging' and r.battery >= 0.80 * MAX_BATTERY))
        and r.battery >= LOW_BATTERY_THRESHOLD
        for r in robots
    )


@app.route("/api/next_step")
def next_step():
    total_tasks = len(state["tasks"])
    completed_tasks = sum(1 for t in state["tasks"] if t["served"])

    MAX_DELAY = 20  # 任务分配后最长等待时间（tick）
    # ====== 过期机器人 ======
    for t in state["tasks"]:
        if (not t["served"]
                # and t.get("assigned_to") is None
                and state["tick"] > t.get("departure_time", MAX_TIME)
                and not t.get("expired", False)):
            # 如果任务从未分配，或已分配但长时间没人执行也过期
            assigned = t.get("assigned_to")
            assigned_robot = next((r for r in state["robots"] if r.id == assigned), None)

            if assigned is None or (
                    assigned_robot and assigned_robot.task != t
                    and (state["tick"] - t["arrival_time"] > MAX_DELAY)
            ):
                t["expired"] = True
                t["served"] = True  # ✅ 表示生命周期终止

    # ====== 模拟终止条件：所有任务完成 ======
    if completed_tasks == total_tasks and total_tasks > 0:
        # === 输出最终统计指标（以控制台方式展示） ===
        energy = sum(r.energy_used for r in state["robots"])
        appeared = [t for t in state["tasks"] if t["arrival_time"] <= state["tick"]]
        served = [t for t in appeared if t["served"] and not t.get("expired")]
        timeout = [t for t in appeared if t.get("expired")]
        wait_times = [t["start_time"] - t["arrival_time"] for t in served if t["start_time"] is not None]

        avg_wait = sum(wait_times) / len(wait_times) if wait_times else 0
        wait_std = math.sqrt(variance(wait_times)) if len(wait_times) > 1 else 0
        completion_rate = 100 * len(served) / len(appeared) if appeared else 0
        timeout_ratio = 100 * len(timeout) / len(appeared) if appeared else 0

        print("=" * 50)
        print(f"[Tick {state['tick']}] 仿真已完成，最终统计指标如下：")
        print(f"总能耗 Energy Used: {energy:.2f} 单位")
        print(f"平均等待时间 Avg Wait: {avg_wait:.2f} tick")
        print(f"等待时间标准差 Std Dev: {wait_std:.2f} tick")
        print(f"完成率 Completion Rate: {completion_rate:.1f}%")
        print(f"超时率 Timeout Ratio: {timeout_ratio:.1f}%")
        print("=" * 50)

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

    elif state["strategy"] == "ppo":  # <-- PPO 分支
        # 1. 首次创建自定义 PPOAgent
        if "ppo_agent" not in state:
            obs_dim = state["NUM_ROBOTS"] * 3
            action_dim = len(state["tasks"]) + 1  # 记得加上"返桩"动作

            # 用当前 scale/distribution/arrival 拼文件名
            # fname = f"policy_{state['scale']}_{state['distribution']}_{state['arrival']}.pth"
            fname = f"ppo_model/{state['scale']}/policy_{state['scale']}_{state['distribution']}_{state['arrival']}.pth"
            state["ppo_agent"] = PPOAgent(
                obs_dim=obs_dim,
                action_dim=action_dim,
                policy_path=fname,  # ← 动态加载对应文件
                device="cpu"
            )
        # 2. 整理所有未完成的任务（不再考虑 arrival_time）
        active_tasks = [
            t for t in state["tasks"]
            if (not t["served"]) and (t.get("assigned_to") is None)
        ]

        # 3. 构造 obs，并 flatten
        obs = np.array([
            [
                r.battery / MAX_BATTERY,
                r.pos[0] / state["HEIGHT"],
                r.pos[1] / state["WIDTH"]
            ]
            for r in state["robots"]
        ], dtype=np.float32)  # shape = (NUM_ROBOTS, 3)
        flat_obs = obs.flatten()  # shape = (NUM_ROBOTS*3,)

        # 4. 用自定义 PPOAgent 预测
        idx = state["ppo_agent"].choose_action(flat_obs)
        # 4.1 如果选中了 "返回充电" 动作,让电量最低的去充电
        if idx == len(active_tasks):
            # 选一个电量最低的机器人去充电
            to_charge = min(state["robots"], key=lambda r: r.battery)
            # 解锁它的任务（如果有的话）
            if to_charge.task is not None:
                to_charge.task["assigned_to"] = None
                to_charge.task = None
            # 立即返最近的充电桩
            to_charge.return_to_charge(
                state["grid"],
                state["HEIGHT"],
                state["WIDTH"]
            )
        # 4.2 如果选中了某个任务
        # 筛出所有空闲且手头无活的机器人
        for task in active_tasks:
            # 筛出所有空闲且手头无活的机器人
            idle_robots = [
                r for r in state["robots"]
                if r.state == "idle" and r.task is None
            ]
            if not idle_robots:
                continue
            # 按到任务点的路径成本升序排序
            idle_robots.sort(
                key=lambda r: path_cost(
                    state["grid"],
                    r.pos,
                    task["location"],
                    state["HEIGHT"],
                    state["WIDTH"]
                )
            )
            # 依次尝试，找到第一个既可行又有路可走的最近机器人
            for r in idle_robots:
                # 可行性检查
                if not is_feasible(
                        r,
                        task,
                        state["grid"],
                        state["chargers"],
                        state["HEIGHT"],
                        state["WIDTH"]
                ):
                    continue
                # 计算 A* 路径
                path = a_star(
                    state["grid"],
                    r.pos,
                    task["location"],
                    state["HEIGHT"],
                    state["WIDTH"]
                )
                if path:
                    r.assign(task, path)
                    break

    elif state["strategy"] == "hspso" or state["strategy"] == "hs_pso_d":  # <-- SPSO 分支
        #  仅在有空闲且电量充足的机器人时触发调度器
        if should_dispatch(state["robots"]):
            # 使用优化版本的混合SPSO
            assign_tasks_optimized_hybrid_spso(
                state["robots"], state["tasks"], state["tick"],
                state["grid"], state["chargers"],
                state["HEIGHT"], state["WIDTH"]
            )
            print(f"[Tick {state['tick']}] 调用优化混合SPSO调度器")

    for robot in state["robots"]:
        robot.step(state["grid"], state["tick"], state["HEIGHT"], state["WIDTH"])

    return jsonify({"message": "调度推进一帧", "tick": state["tick"], "done": False})


# ========== 获取当前状态快照 ==========
@app.route("/api/get_state")
def get_state():
    # 地图面积 area = state["WIDTH"] * state["HEIGHT"]
    energy = sum(r.energy_used for r in state["robots"])
    # 已出现任务（可被分配的+包括过期的）
    appeared = [t for t in state["tasks"] if t["arrival_time"] <= state["tick"]]

    # 已服务的未过期的任务
    served = [t for t in appeared if t["served"] and not t.get("expired")]

    # 已完成任务（包括过期和未过期的）
    completed = [t for t in state["tasks"] if t["served"]]

    # 超时任务（expired）
    timeout = [t for t in appeared if t.get("expired")]

    # 等待时间列表（只取已成功被服务的任务）
    wait_times = [t["start_time"] - t["arrival_time"] for t in served if t["start_time"] is not None]

    # 计算指标
    # 成功被服务任务的从任务生成到首次被机器人服务的平均时间
    avg_wait = sum(wait_times) / len(wait_times) if wait_times else 0
    wait_std = math.sqrt(variance(wait_times)) if len(wait_times) > 1 else 0
    # 在已出现的任务中成功服务且未过期的
    # 与progress进度条辨析：progress是仿真"整体进程"，即所有任务中完成了多少（包括过期的）
    completion_rate = 100 * len(served) / len(appeared) if appeared else 0
    timeout_ratio = 100 * len(timeout) / len(appeared) if appeared else 0

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
        # Overview页面展示
        "metrics": {
            "efficiency": {
                "energy": round(energy, 2),
                "avg_wait": round(avg_wait, 2),
                "wait_standard": round(wait_std, 2)
            },
            "task_stats": {
                "completion_rate": round(completion_rate, 1),
                "timeout_ratio": round(timeout_ratio, 1)
            }
        },
        # 其他可能会用到的数据指标
        "counts": {
            "appeared": len(appeared),
            "served": len(served),
            "completed": len(completed),
            "timeout": len(timeout)
        }
    })


# ========== 设置调度策略 ==========
@app.route("/api/set_strategy")
def set_strategy():
    strategy = request.args.get("strategy", "hungarian")

    # 策略名称标准化
    if strategy in ["s_pso_d", "spso_d", "spso"]:
        strategy = "hspso"  # 统一为 hspso

    if strategy in ["hungarian", "ppo", "hspso"]:
        state["strategy"] = strategy
        return jsonify({"message": f"调度策略已设置为: {strategy}", "success": True})
    else:
        return jsonify({"message": f"无效的策略: {strategy}", "success": False})


if __name__ == '__main__':
    app.run(debug=True, port=5050)