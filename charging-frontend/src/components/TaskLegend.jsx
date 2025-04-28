// src/components/TaskLegend.jsx
import React from 'react';
import { motion } from 'framer-motion'; // ✅ 引入 framer-motion 库用于添加动效
import '../styles/panel.css';
import '../styles/map.css';
import { Bot, Car, BatteryCharging, ShieldOff, Star } from 'lucide-react';
import { generateStarPoints } from './MapView.jsx'; // 保留，继续使用原有的generateStarPoints函数

// 定义星星形状（中心点11,11；内圈半径5；外圈半径10；五个角）
const starPoints = generateStarPoints(11, 11, 5, 10, 5);

function TaskLegend() {
  return (
    <div className="task-legend">
      <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        {/* Icon绘制 */}
        <ShieldOff size={20} /> 图例说明
      </h3>

      <ul style={{ listStyle: 'none', padding: 0, marginTop: '1rem' }}>

        {/* Robot Normal */}
        <li style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
          <Bot size={18} />
          {/* 使用motion.div增加轻微放大hover动效 */}
          <motion.div whileHover={{ scale: 1.2 }} style={{ width: '16px', height: '16px', backgroundColor: '#0984e3', border: '1px solid #34495e', borderRadius: '2px' }} />
          Robot（正常）
        </li>

        {/* Robot Low Battery */}
        <li style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
          <div style={{ width: '18px', height: '18px' }} /> {/* 保持左侧空位对齐 */}
          <motion.div whileHover={{ scale: 1.2 }} style={{ width: '16px', height: '16px', backgroundColor: '#f4a261', border: '1px solid #e98b3b', borderRadius: '2px' }} />
          Robot（低电量）
        </li>

        {/* Task Unfinished */}
        <li style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
          <Star size={18} />
          {/* 使用motion.svg使星星hover时轻微放大 */}
          <motion.svg whileHover={{ scale: 1.2 }} width="22" height="22" viewBox="0 0 22 22">
            <polygon points={starPoints} fill="#f5a623" />
          </motion.svg>
          Task（待充车辆）
        </li>

        {/* Task Finished */}
        <li style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
          <div style={{ width: '18px', height: '18px' }} />
          <motion.svg whileHover={{ scale: 1.2 }} width="22" height="22" viewBox="0 0 22 22">
            <polygon points={starPoints} fill="#d5d8dc" opacity="0.6" />
          </motion.svg>
          Task（充电完成）
        </li>

        {/* Charging Station */}
        <li style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
          <BatteryCharging size={18} />
          <motion.div whileHover={{ scale: 1.2 }} style={{ width: '16px', height: '16px', backgroundColor: '#6fcf97', border: '1px solid #45b37e', borderRadius: '50%' }} />
          Charging Station
        </li>

        {/* Parking Spot */}
        <li style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
          <Car size={18} />
          <motion.div whileHover={{ scale: 1.2 }} style={{ width: '16px', height: '16px', backgroundColor: '#bdc3c7', border: '1px solid #95a5a6', borderRadius: '50%' }} />
          Parking Spot
        </li>

        {/* Obstacle */}
        <li style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
          <ShieldOff size={18} />
          <motion.div whileHover={{ scale: 1.2 }} style={{ width: '16px', height: '16px', backgroundColor: '#4b4b4b', border: '1px solid #333', borderRadius: '2px' }} />
          Obstacle
        </li>
      </ul>
    </div>
  );
}

export default TaskLegend;