import React from 'react';
import '../styles/panel.css';
import {
  BatteryCharging,
  Clock,
  SlidersHorizontal,
  Percent,
  TimerReset,
  BarChart
} from 'lucide-react';

/**
 * InfoPanel 组件：展示系统当前 tick 的关键指标（按功能分类分组）
 * Props:
 *  - metrics: 结构包含两个子项：efficiency（效率类）与 task_stats（任务类）
 *  - strategy: 当前调度策略名（可选）
 *  - scale: 当前地图规模名（可选）
 */
function InfoPanel({ metrics = {}, strategy = '', scale = '' }) {
  const {
    efficiency = {},
    task_stats = {}
  } = metrics;

  const { energy = 0, avg_wait = 0, wait_standard = 0 } = efficiency;
  const { completion_rate = 0, timeout_ratio = 0 } = task_stats;

  // 单行指标显示组件
  const MetricRow = ({ icon: Icon, label, value, suffix = '' }) => (
    <div style={{
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      fontSize: '0.925rem',
      color: '#334155'
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
        <Icon size={16} strokeWidth={1.75} color="#334155" />
        <span>{label}</span>
      </div>
      <strong style={{ fontWeight: 600, color: '#0f172a' }}>{value}{suffix}</strong>
    </div>
  );

  return (
    <div className="info-panel" style={{
      marginTop: '-0.05rem',             // 整体卡片上移
      padding: '1rem',
      borderRadius: '1rem',
      backgroundColor: '#ffffff',
      boxShadow: '0 4px 12px rgba(0,0,0,0.08)',
      fontFamily: 'system-ui'
    }}>
      {/* 标题 */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '0.5rem',
        fontSize: '1.05rem',
        fontWeight: '600',
        color: '#1e293b',
        borderBottom: '1px solid #e2e8f0',
        paddingBottom: '0.4rem',  // 标题下内边距减小
        marginBottom: '0.85rem',    // 标题下外边距减小
        marginTop: '-0.025rem',   // 标题整体上移
      }}>
        <BarChart size={18} strokeWidth={2} />
        当前指标
      </div>

      {/* 系统效率指标 */}
      <div style={{ fontSize: '0.85rem', fontWeight: 500, color: '#64748b', marginBottom: '0.5rem' }}>
        系统效率指标
      </div>
      <div style={{ display: 'grid', rowGap: '0.5rem', marginBottom: '1rem' }}>
        <MetricRow icon={BatteryCharging} label="总能耗" value={energy.toFixed(2)} />
        <MetricRow icon={Clock} label="平均等待时间" value={avg_wait.toFixed(2)} />
        <MetricRow icon={SlidersHorizontal} label="等待时间标准差" value={wait_standard.toFixed(2)} />
      </div>

      {/* 任务完成指标 */}
      <div style={{ fontSize: '0.85rem', fontWeight: 500, color: '#64748b', marginBottom: '0.5rem' }}>
        任务完成指标
      </div>
      <div style={{ display: 'grid', rowGap: '0.5rem' }}>
        <MetricRow icon={Percent} label="完成率" value={completion_rate.toFixed(1)} suffix="%" />
        <MetricRow icon={TimerReset} label="超时比例" value={timeout_ratio.toFixed(1)} suffix="%" />
      </div>
    </div>
  );
}

export default InfoPanel;