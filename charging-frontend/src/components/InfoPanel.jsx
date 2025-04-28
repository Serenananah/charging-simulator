// src/components/InfoPanel.jsx
import React from 'react';
import '../styles/panel.css';
import { SlidersHorizontal, Map, BatteryCharging, Clock, CheckCircle } from 'lucide-react';

/**
 * InfoPanel 组件：展示系统当前 tick 的关键指标
 * Props:
 *  - metrics: 包含 energy, delay, completed
 *  - strategy: 当前调度策略名（可选）
 *  - scale: 当前地图规模名（可选）
 */
function InfoPanel({ metrics = {}, strategy = '', scale = '' }) {
  const { energy = 0, delay = 0, completed = 0 } = metrics;

  return (
    <div className="info-panel">
      <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <BatteryCharging size={16} /> 当前指标
      </h3>
      <ul>
        {/* 上面有了策略和地图规模选择
        <li style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <SlidersHorizontal size={16} /> 调度策略：<strong>{strategy || '未指定'}</strong>
        </li>
        <li style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Map size={16} /> 地图规模：<strong>{scale || '未指定'}</strong>
        </li>
        */}
        <li style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <BatteryCharging size={16} /> 总能耗：<strong>{energy.toFixed(2)}</strong>
        </li>
        <li style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Clock size={16} /> 平均任务延迟：<strong>{delay.toFixed(2)}</strong>
        </li>
        <li style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <CheckCircle size={16} /> 已完成任务数：<strong>{completed}</strong>
        </li>
      </ul>
    </div>
  );
}

export default InfoPanel;