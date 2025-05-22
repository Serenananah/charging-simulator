import React from 'react';
import '../styles/metrics.css';

const scenarios = [
  {
    id: 'E1',
    label: '理想低负载',
    tooltip: `任务数：20\n分布：clustered + poisson\n地图：25×25\n说明：性能上限基准场景`,
  },
  {
    id: 'E2',
    label: '任务密集高负载',
    tooltip: `任务数：60\n分布：clustered + poisson\n地图：25×25\n说明：任务密度高时系统抗压表现`,
  },
  {
    id: 'E3',
    label: '峰值冲击',
    tooltip: `任务数：44\n分布：clustered + normal\n地图：25×25\n说明：模拟集中高峰时间任务突发时系统响应能力`,
  },
  {
    id: 'E4',
    label: '路径冲突',
    tooltip: `任务数：44\n分布：uniform + poisson\n地图：25×25\n说明：车位稀疏分布时的路径竞争`,
  },
  {
    id: 'E5',
    label: '大图路径代价',
    tooltip: `任务数：44\n分布：clustered + poisson\n地图：40×40\n说明：测试地图放大是否带来路径成本增加`,
  },
  {
    id: 'E6',
    label: '多重压力',
    tooltip: `任务数：80\n分布：uniform + normal\n地图：40×40\n说明：系统复杂压力极限测试`,
  },
];

export default function ScenarioHeaderStrip() {
  return (
    <div className="scenario-strip">
      {scenarios.map((s, i) => (
        <div key={i} className={`scenario-tag color-${i + 1}`}>
          <span className="scenario-id">{s.id}</span>
          <span className="scenario-label">{s.label}</span>
          <div className="tooltip bottom">{s.tooltip}</div>
        </div>
      ))}
    </div>
  );
}
