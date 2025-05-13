import React from 'react';
import { motion } from 'framer-motion'; // 用于添加动效
import '../styles/panel.css';
import '../styles/map.css';
import { Bot, Car, BatteryCharging, ShieldOff, Star } from 'lucide-react';
import { generateStarPoints } from './MapView.jsx';

const starPoints = generateStarPoints(11, 11, 5, 10, 5);

function TaskLegend() {
  return (
    <div className="task-legend" style={{
      marginTop: '-0.05rem',             // 整体卡片上移
      padding: '1rem',                          // 设置内边距使内容不紧贴边缘
      borderRadius: '1rem',                    // 圆角卡片样式，提升柔和感
      backgroundColor: '#ffffff',              // 白色背景，与InfoPanel统一
      boxShadow: '0 4px 12px rgba(0,0,0,0.08)', // 阴影增强层次感
      fontSize: '0.85rem'                      // 字体略小，避免显得拥挤
    }}>
      <h4 style={{
        display: 'flex',                       // 图标和文字并排
        alignItems: 'center',                  // 垂直居中对齐
        gap: '0.5rem',                          // 图标与文字间距
        fontSize: '1.1rem',                     // 稍小一级标题
        fontWeight: '600',                      // 加粗提升辨识度
        color: '#1e293b',                       // 深蓝灰色，统一文本色
        borderBottom: '1px solid #e2e8f0',      // 下边框分隔
        paddingBottom: '0.5rem',                // 标题下间距
        marginBottom: '0.75rem',                 // 与内容分离
        marginTop: '-0.025rem',                  // 标题整体上移
      }}>
        <ShieldOff size={18} /> 图例说明
      </h4>

      <ul style={{
        listStyle: 'none',                     // 移除默认圆点
        padding: 0,
        margin: 0,
        display: 'grid',                        // 使用grid控制间距一致
        rowGap: '0.75rem'                       // 每项间距
      }}>
        {/* Robot Normal */}
        <li style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <Bot size={16} />
          <motion.div whileHover={{ scale: 1.2 }} style={{
            width: '16px', height: '16px',
            backgroundColor: '#0984e3',         // 正常电量蓝
            border: '1px solid #34495e',
            borderRadius: '2px'
          }} />
          Robot（正常）
        </li>

        {/* Robot Low Battery */}
        <li style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <div style={{ width: '16px' }} />
          <motion.div whileHover={{ scale: 1.2 }} style={{
            width: '16px', height: '16px',
            backgroundColor: '#f4a261',         // 低电量橙
            border: '1px solid #e98b3b',
            borderRadius: '2px'
          }} />
          Robot（低电量）
        </li>

        {/* Task Unfinished */}
        <li style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <Star size={16} />
          <motion.svg whileHover={{ scale: 1.2 }} width="22" height="22" viewBox="0 0 22 22">
            <polygon points={starPoints} fill="#f5a623" />
          </motion.svg>
          Task（待充车辆）
        </li>

        {/* Task Finished */}
        <li style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <div style={{ width: '16px' }} />
          <motion.svg whileHover={{ scale: 1.2 }} width="22" height="22" viewBox="0 0 22 22">
            <polygon points={starPoints} fill="#d5d8dc" opacity="0.6" />
          </motion.svg>
          Task（充电完成）
        </li>

        {/* Charging Station */}
        <li style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <BatteryCharging size={16} />
          <motion.div whileHover={{ scale: 1.2 }} style={{
            width: '16px', height: '16px',
            backgroundColor: '#6fcf97',
            border: '1px solid #45b37e',
            borderRadius: '50%'
          }} />
          Charging Station
        </li>

        {/* Parking Spot */}
        <li style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <Car size={16} />
          <motion.div whileHover={{ scale: 1.2 }} style={{
            width: '16px', height: '16px',
            backgroundColor: '#bdc3c7',
            border: '1px solid #95a5a6',
            borderRadius: '50%'
          }} />
          Parking Spot
        </li>

        {/* Obstacle */}
        <li style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <ShieldOff size={16} />
          <motion.div whileHover={{ scale: 1.2 }} style={{
            width: '16px', height: '16px',
            backgroundColor: '#4b4b4b',
            border: '1px solid #333',
            borderRadius: '2px'
          }} />
          Obstacle
        </li>
      </ul>
    </div>
  );
}

export default TaskLegend;
