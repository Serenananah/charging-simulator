// 地图渲染组件
import React from 'react';
import '../styles/map.css';

// 地图编码定义
const EMPTY = 0;
const OBSTACLE = 1;
const PARKING_SPOT = 2;
const CHARGING_STATION = 3;

// 生成五角星路径
export function generateStarPoints(centerX, centerY, innerRadius, outerRadius, points) {
  const angle = Math.PI / points;
  let path = "";
  for (let i = 0; i < 2 * points; i++) {
    const r = (i % 2 === 0) ? outerRadius : innerRadius;
    const currX = centerX + r * Math.sin(i * angle);
    const currY = centerY - r * Math.cos(i * angle);
    path += `${i === 0 ? "" : " "} ${currX},${currY}`;
  }
  return path.trim();
}

function MapView({ state }) {
  const grid = state.grid || [];
  const robots = state.robots || [];
  const tasks = state.tasks || [];
  const tick = state.tick || 0; // 当前时间步

  const GRID_SIZE = grid.length;  // 动态格子数量
  const TOTAL_SIZE = 625;          // 固定总尺寸，比如625px
  const CELL_SIZE = GRID_SIZE > 0 ? TOTAL_SIZE / GRID_SIZE : 25;  // 自动适配每格大小

  return (
    <div className="map-container">
      <svg
          className="map-grid"
          width={TOTAL_SIZE}
          height={TOTAL_SIZE}
          // style={{padding: '10px'}}          // 加内边距
      >
        {/* 绘制网格背景格子 */}
        {[...Array(GRID_SIZE)].map((_, i) => (
            <line key={"h" + i} x1={0} y1={i * CELL_SIZE} x2={GRID_SIZE * CELL_SIZE} y2={i * CELL_SIZE}
                  className="grid-line"/>
        ))}
        {[...Array(GRID_SIZE)].map((_, i) => (
            <line key={"v" + i} y1={0} x1={i * CELL_SIZE} y2={GRID_SIZE * CELL_SIZE} x2={i * CELL_SIZE}
                  className="grid-line"/>
        ))}

        {/* 绘制地图内容（障碍物 / 停车位 / 充电桩） */}
        {grid.map((row, y) =>
            row.map((cell, x) => {
              if (cell === OBSTACLE) {
                return (
                    <rect
                        key={`obstacle-${x}-${y}`}
                        x={x * CELL_SIZE}
                        y={y * CELL_SIZE}
                        width={CELL_SIZE}
                        height={CELL_SIZE}
                        className="obstacle"
                    />
                )
              }
              if (cell === CHARGING_STATION) {
                return (
                    <circle
                        key={`station-${x}-${y}`}
                        cx={x * CELL_SIZE + CELL_SIZE / 2}
                        cy={y * CELL_SIZE + CELL_SIZE / 2}
                        // 控制大小
                        r={CELL_SIZE * 0.6}
                        className="station"
                    />
                )
              }
              if (cell === PARKING_SPOT) {
                return (
                    <circle
                        key={`parking-${x}-${y}`}
                        cx={x * CELL_SIZE + CELL_SIZE / 2}
                        cy={y * CELL_SIZE + CELL_SIZE / 2}
                        // 控制大小
                        r={CELL_SIZE * 0.5}
                        className="parking-spot"
                    />
                )
              }
              return null;
            })
        )}

        {/* 绘制任务（未完成任务：#0984e3蓝色五角星/#f5b971黄色五角星/#f5a623鲜亮橙黄，完成任务：灰色半透明五角星） */}
        {tasks.map((task) => {
            if (task.arrival_time > tick) return null; // 不显示未来任务（对齐后端动画）
            // 星星中心点横纵坐标
            const centerX = task.location[1] * CELL_SIZE + CELL_SIZE / 2;
            const centerY = task.location[0] * CELL_SIZE + CELL_SIZE / 2;
            // 星星外半径，决定星星整体大小
            const outerRadius = CELL_SIZE * 0.6;
            // 星星内半径，决定凹陷程度（越小越尖）
            const innerRadius = outerRadius * 0.55;


            return (
              <polygon
                  key={`task-${task.task_id}`}
                  points={generateStarPoints(centerX, centerY, innerRadius, outerRadius, 5)}
                  fill={task.served ? "#bdc3c7" : "#f5a623"}
                  // 完成任务半透明
                  opacity={task.served ? 0.6 : 1}
                  stroke={task.served ? "#95a5a6" : "#d35400"} // 未完成任务橙色描边，完成任务灰描边
                  strokeWidth="1" // 细轮廓，提升清晰度
              />
            );
        })}

        {/* 绘制机器人（根据电量动态变色） */}
        {robots.map((r) => (
            <g key={`robot-${r.id}`}>
              {/* 绘制机器人本体 */}
              <rect
                  // 控制中心
                  x={r.pos[1] * CELL_SIZE + CELL_SIZE * 0.1}
                  y={r.pos[0] * CELL_SIZE + CELL_SIZE * 0.1}
                  // 控制大小
                  width={CELL_SIZE * 0.8}
                  height={CELL_SIZE * 0.8}
                  className={r.battery < 20 ? "robot-low" : "robot"}
              />
              {/* 在机器人上方绘制电量百分比 */}
              <text
                  x={r.pos[1] * CELL_SIZE + CELL_SIZE / 2}
                  y={r.pos[0] * CELL_SIZE + 1}  // 比机器人稍微高一点
                  textAnchor="middle"
                  // fontSize="12px"
                  fontSize={Math.max(CELL_SIZE * 0.6, 11)}  // 动态字号，最小值防止太小
                  fill="#000000"  // #333深灰色
                  fontWeight="bold"  // 加粗字体，让字体更深更清晰
                  strokeWidth="0.8px"  // 描边宽度
              >
                {`${Math.floor(r.battery)}%`}
              </text>
            </g>
        ))}
      </svg>
    </div>
  );
}

// 默认导出 MapView 组件
export default MapView;
