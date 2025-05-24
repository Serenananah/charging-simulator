import React from 'react';
import { Group } from '@visx/group';
import { curveBasis } from '@visx/curve';
import { LinePath } from '@visx/shape';
import { Threshold } from '@visx/threshold';
import { scaleLinear, scalePoint } from '@visx/scale';
import { AxisLeft, AxisBottom } from '@visx/axis';
import { GridRows, GridColumns } from '@visx/grid';
import {
  useTooltip,
  useTooltipInPortal,
  TooltipWithBounds,
  defaultStyles as defaultTooltipStyles,
} from '@visx/tooltip';

export const background = '#f3f3f3';

const data = [
  { x: 'E1', PPO: 8.01, Hungarian: 25.26, HSPSO: 28.026 },
  { x: 'E2', PPO: 20.90, Hungarian: 26.84, HSPSO: 24.769 },
  { x: 'E3', PPO: 15.21, Hungarian: 26.81, HSPSO: 33.259 },
  { x: 'E4', PPO: 23.48, Hungarian: 26.48, HSPSO: 27.853 },
  { x: 'E5', PPO: 12.85, Hungarian: 28.94, HSPSO: 25.976 },
  { x: 'E6', PPO: 18.34, Hungarian: 28.68, HSPSO: 32.862 },
];

const xScale = scalePoint({
  domain: data.map(d => d.x),
  range: [0, 400],
});
const yScale = scaleLinear({
  domain: [0, 35],
  nice: true,
  range: [300, 0],
});

export type ThresholdProps = {
  width: number;
  height: number;
  margin?: { top: number; right: number; bottom: number; left: number };
};

const defaultMargin = { top: 40, right: 30, bottom: 50, left: 40 };

export default function ThresholdChart({ width, height, margin = defaultMargin }: ThresholdProps) {
  if (width < 10) return null;

  const xMax = width - margin.left - margin.right;
  const yMax = height - margin.top - margin.bottom;

  xScale.range([0, xMax]);
  yScale.range([yMax, 0]);

  const {
    tooltipData,
    tooltipLeft,
    tooltipTop,
    tooltipOpen,
    showTooltip,
    hideTooltip,
  } = useTooltip<any>();

  const { containerRef } = useTooltipInPortal({ detectBounds: true, scroll: true });

  return (
    <div ref={containerRef}>
      <svg width={width} height={height}>
        <rect x={0} y={0} width={width} height={height} fill={background} rx={14} />
        <Group left={margin.left} top={margin.top}>
          <GridRows scale={yScale} width={xMax} height={yMax} stroke="#e0e0e0" />
          <GridColumns scale={xScale} width={xMax} height={yMax} stroke="#e0e0e0" />
          <AxisBottom top={yMax} scale={xScale} />
          <AxisLeft scale={yScale} />

          {/* PPO vs Hungarian (purple) */}
          <Threshold
            id="ppo-threshold"
            data={data}
            x={d => xScale(d.x) ?? 0}
            y0={d => yScale(d.PPO) ?? 0}
            y1={d => yScale(d.Hungarian) ?? 0}
            clipAboveTo={0}
            clipBelowTo={yMax}
            curve={curveBasis}
            belowAreaProps={{ fill: 'violet', fillOpacity: 0.4 }}
            aboveAreaProps={{ fill: 'violet', fillOpacity: 0.2 }}
          />

          {/* HSPSO vs Hungarian (green) */}
          <Threshold
            id="hspso-threshold"
            data={data}
            x={d => xScale(d.x) ?? 0}
            y0={d => yScale(d.HSPSO) ?? 0}
            y1={d => yScale(d.Hungarian) ?? 0}
            clipAboveTo={0}
            clipBelowTo={yMax}
            curve={curveBasis}
            belowAreaProps={{ fill: 'green', fillOpacity: 0.3 }}
            aboveAreaProps={{ fill: 'green', fillOpacity: 0.15 }}
          />

          {/* Hungarian */}
          <LinePath
            data={data}
            curve={curveBasis}
            x={d => xScale(d.x) ?? 0}
            y={d => yScale(d.Hungarian) ?? 0}
            stroke="#1f77b4"
            strokeWidth={2}
            onMouseMove={(event) => {
              const { x, y } = event.nativeEvent;
              const closest = data.find(d => Math.abs(xScale(d.x)! - (x - margin.left)) < 20);
              if (closest) {
                showTooltip({
                  tooltipData: { label: 'Hungarian', x: closest.x, y: closest.Hungarian },
                  tooltipLeft: x,
                  tooltipTop: y,
                });
              }
            }}
            onMouseLeave={hideTooltip}
          />

          {/* PPO */}
          <LinePath
            data={data}
            curve={curveBasis}
            x={d => xScale(d.x) ?? 0}
            y={d => yScale(d.PPO) ?? 0}
            stroke="#ff7f0e"
            strokeWidth={2}
            strokeDasharray="4,2"
            onMouseMove={(event) => {
              const { x, y } = event.nativeEvent;
              const closest = data.find(d => Math.abs(xScale(d.x)! - (x - margin.left)) < 20);
              if (closest) {
                showTooltip({
                  tooltipData: { label: 'PPO', x: closest.x, y: closest.PPO },
                  tooltipLeft: x,
                  tooltipTop: y,
                });
              }
            }}
            onMouseLeave={hideTooltip}
          />

          {/* HSPSO */}
          <LinePath
            data={data}
            curve={curveBasis}
            x={d => xScale(d.x) ?? 0}
            y={d => yScale(d.HSPSO) ?? 0}
            stroke="#2ca02c"
            strokeWidth={2}
            strokeDasharray="6,2"
            onMouseMove={(event) => {
              const { x, y } = event.nativeEvent;
              const closest = data.find(d => Math.abs(xScale(d.x)! - (x - margin.left)) < 20);
              if (closest) {
                showTooltip({
                  tooltipData: { label: 'SPSO', x: closest.x, y: closest.HSPSO },
                  tooltipLeft: x,
                  tooltipTop: y,
                });
              }
            }}
            onMouseLeave={hideTooltip}
          />


          {/* PPO tooltip points */}
          {data.map((d, i) => (
            <circle
              key={`ppo-point-${i}`}
              cx={xScale(d.x) ?? 0}
              cy={yScale(d.PPO) ?? 0}
              r={10}
              fill="transparent"
              pointerEvents="all"
              onMouseMove={(event) => {
                const { clientX, clientY } = event;
                showTooltip({
                  tooltipData: { label: 'PPO', x: d.x, y: d.PPO },
                  tooltipLeft: clientX,
                  tooltipTop: clientY,
                });
              }}
              onMouseLeave={hideTooltip}
            />
          ))}

          {/*添加透明交互点来触发tooltip相应*/}
          {/* HSPSO tooltip points */}
          {data.map((d, i) => (
            <circle
              key={`spso-point-${i}`}
              cx={xScale(d.x) ?? 0}
              cy={yScale(d.HSPSO) ?? 0}
              r={10}
              fill="transparent"
              pointerEvents="all"
              onMouseMove={(event) => {
                const { clientX, clientY } = event;
                showTooltip({
                  tooltipData: { label: 'HSPSO', x: d.x, y: d.HSPSO },
                  tooltipLeft: clientX,
                  tooltipTop: clientY,
                });
              }}
              onMouseLeave={hideTooltip}
            />
          ))}

          {/* Hungarian tooltip points */}
          {data.map((d, i) => (
            <circle
              key={`hungarian-point-${i}`}
              cx={xScale(d.x) ?? 0}
              cy={yScale(d.Hungarian) ?? 0}
              r={10}
              fill="transparent"
              pointerEvents="all"
              onMouseMove={(event) => {
                const { clientX, clientY } = event;
                showTooltip({
                  tooltipData: { label: 'Hungarian', x: d.x, y: d.Hungarian },
                  tooltipLeft: clientX,
                  tooltipTop: clientY,
                });
              }}
              onMouseLeave={hideTooltip}
            />
          ))}


          {/* Labels */}
          <text x={xScale('E6')!} y={yScale(data[5].Hungarian)} fontSize={10} fontWeight="bold" fill="#1f77b4">
            Hungarian
          </text>
          <text x={xScale('E6')!} y={yScale(data[5].PPO) - 10} fontSize={10} fontWeight="bold" fill="#ff7f0e">
            PPO
          </text>
          <text x={xScale('E6')!} y={yScale(data[5].HSPSO) - 10} fontSize={10} fontWeight="bold" fill="#2ca02c">
            SPSO
          </text>
        </Group>
      </svg>

      {tooltipOpen && tooltipData && (
        <TooltipWithBounds
          top={tooltipTop}
          left={tooltipLeft}
          style={{
            ...defaultTooltipStyles,
            backgroundColor: '#333',
            color: 'white',
            fontSize: 12,
          }}
        >
          <div><strong>{tooltipData.label}</strong></div>
          <div>Scene: {tooltipData.x}</div>
          <div>Value: {tooltipData.y.toFixed(2)}</div>
        </TooltipWithBounds>
      )}
    </div>
  );
}
