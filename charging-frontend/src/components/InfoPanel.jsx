import React from 'react';
import '../styles/panel.css';
import { SlidersHorizontal, Map, BatteryCharging, Clock, CheckCircle, Percent} from 'lucide-react';

/**
 * InfoPanel 组件：展示系统当前 tick 的关键指标
 * Props:
 *  *  - metrics: 包含 energy（总能耗）, delay（平均延迟）, completed（已完成任务数）, completionRate（任务完成率）等
 *  - strategy: 当前调度策略名（可选）
 *  - scale: 当前地图规模名（可选）
 */
function InfoPanel({ metrics = {}, strategy = '', scale = '' }) {
  const { energy = 0, delay = 0, completed = 0, completionRate = 0} = metrics;

  return (
    <div className="info-panel" style={{
        //marginTop: '-0.05rem',             // 整体卡片上移
        padding: '1rem',                          // 设置内边距使内容不紧贴边缘
        borderRadius: '1rem',                    // 圆角卡片样式，提升柔和感
        backgroundColor: '#ffffff',              // 白色背景，与InfoPanel统一
        boxShadow: '0 4px 12px rgba(0,0,0,0.08)', // 阴影增强层次感
    }}>
      <h4 style={{
        display: 'flex',
        alignItems: 'center',
        gap: '0.5rem',
        fontSize: '1.1rem',
        fontWeight: '600',
        color: '#1e293b',
        borderBottom: '1px solid #e2e8f0',
        paddingBottom: '0.5rem',               // 标题下内边距减小
        marginBottom: '0.75rem',                 // 标题下外边距减小
        marginTop: '-0.025rem',                  // 标题整体上移
      }}>
        <BatteryCharging size={18} /> 当前指标
      </h4>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', rowGap: '0.75rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#475569' }}>
            <BatteryCharging size={16} /> 总能耗
          </span>
          <strong style={{ color: '#0f172a' }}>{energy.toFixed(2)}</strong>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#475569' }}>
            <Clock size={16} /> 平均任务延迟
          </span>
          <strong style={{ color: '#0f172a' }}>{delay.toFixed(2)}</strong>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#475569' }}>
            <CheckCircle size={16} /> 已完成任务数
          </span>
          <strong style={{ color: '#0f172a' }}>{completed}</strong>
        </div>
        {/*  任务完成率（去除过期任务后真正完成的比例） */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#475569' }}>
            <Percent size={16} /> 完成率
          </span>
          <strong style={{ color: '#0f172a' }}>{completionRate.toFixed(1)}%</strong>
        </div>
      </div>
    </div>
  );
}

export default InfoPanel;
