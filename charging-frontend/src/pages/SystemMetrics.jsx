import React from 'react';
import Barchart from '../components/Barchart';
import BarStack from '../components/BarStack';
import Boxplot from '../components/Boxplot';
import ThresholdChart from '../components/ThresholdChart.js';
import ScenarioHeaderStrip from '../components/Scenar.jsx'

import '../styles/metrics.css';

export default function SystemMetrics() {
  const chartWidth = 500;
  const chartHeight = 300;

  return (
    <div className="metrics-container">
      <ScenarioHeaderStrip />
      <div className="grid-container">
        <div className="chart-box">
          <h3>Total Energy Consumption</h3>
          <Barchart width={chartWidth} height={chartHeight}/>
        </div>
        <div className="chart-box">
          <h3>Completion & Timeout Rate</h3>
          <BarStack width={chartWidth} height={chartHeight}/>
        </div>
        <div className="chart-box">
          <h3>Wait Time Variance</h3>
          <Boxplot width={chartWidth} height={chartHeight}/>
        </div>
        <div className="chart-box">
          <h3>Average Wait Time</h3>
          <ThresholdChart width={chartWidth} height={chartHeight}/>
        </div>
      </div>
    </div>
  );
}
