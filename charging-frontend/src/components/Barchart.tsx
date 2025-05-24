import React from 'react';
import { Group } from '@visx/group';
import { BarGroup } from '@visx/shape';
import { AxisBottom } from '@visx/axis';
import { scaleBand, scaleLinear, scaleOrdinal } from '@visx/scale';
import {
  useTooltip,
  useTooltipInPortal,
  TooltipWithBounds,
  defaultStyles as defaultTooltipStyles,
} from '@visx/tooltip';

export type BarGroupProps = {
  width: number;
  height: number;
  margin?: { top: number; right: number; bottom: number; left: number };
  events?: boolean;
};

const blue = '#aeeef8';
export const green = '#e5fd3d';
const purple = '#9caff6';
export const background = '#612efb';

// Custom Data for Total Energy Consumption
const data = [
  { scenario: 'E1', PPO: 1493.999, Hungarian: 1578.437, HSPSO: 2885.163 },
  { scenario: 'E2', PPO: 2508.525, Hungarian: 2085.380, HSPSO: 5670.076 },
  { scenario: 'E3', PPO: 2366.844, Hungarian: 1967.353, HSPSO: 4853.224 },
  { scenario: 'E4', PPO: 2769.007, Hungarian: 2176.286, HSPSO: 4982.555 },
  { scenario: 'E5', PPO: 3481.242, Hungarian: 3350.114, HSPSO: 4937.552 },
  { scenario: 'E6', PPO: 4609.501, Hungarian: 3714.336, HSPSO: 5516.105 }
];

const keys = ['Hungarian', 'PPO', 'HSPSO'];
const defaultMargin = { top: 40, right: 0, bottom: 40, left: 30 };

// accessors
const getScenario = (d: any) => d.scenario;

// scales
const dateScale = scaleBand<string>({
  domain: data.map(getScenario),
  padding: 0.2,
});
const cityScale = scaleBand<string>({
  domain: keys,
  padding: 0.1,
});
const tempScale = scaleLinear<number>({
  domain: [0, Math.max(...data.flatMap((d) => keys.map((key) => d[key] ?? 0)))],
});
const colorScale = scaleOrdinal<string, string>({
  domain: keys,
  range: [blue, green, purple],
});

export default function Barchart({
  width,
  height,
  events = false,
  margin = defaultMargin,
}: BarGroupProps) {
  // bounds
  const xMax = width - margin.left - margin.right;
  const yMax = height - margin.top - margin.bottom;

  // update scale output dimensions
  dateScale.rangeRound([0, xMax]);
  cityScale.rangeRound([0, dateScale.bandwidth()]);
  tempScale.range([yMax, 0]);

  // tooltip hook
  const {
    tooltipData,
    tooltipLeft,
    tooltipTop,
    tooltipOpen,
    showTooltip,
    hideTooltip,
  } = useTooltip<any>();

  const { containerRef } = useTooltipInPortal({ detectBounds: true, scroll: true });

  return width < 10 ? null : (
    <div ref={containerRef}>
      <svg width={width} height={height}>
        <rect x={0} y={0} width={width} height={height} fill={background} rx={14} />
        <Group top={margin.top} left={margin.left}>
          <BarGroup
            data={data}
            keys={keys}
            height={yMax}
            x0={getScenario}
            x0Scale={dateScale}
            x1Scale={cityScale}
            yScale={tempScale}
            color={colorScale}
          >
            {(barGroups) =>
              barGroups.map((barGroup) => (
                <Group key={`bar-group-${barGroup.index}-${barGroup.x0}`} left={barGroup.x0}>
                  {barGroup.bars.map((bar) =>
                    bar.height > 0 && (
                      <rect
                        key={`bar-group-bar-${barGroup.index}-${bar.index}-${bar.value}-${bar.key}`}
                        x={bar.x}
                        y={bar.y}
                        width={bar.width}
                        height={bar.height}
                        fill={bar.color}
                        rx={4}
                        onMouseMove={(event) => {
                          const { key, value } = bar;
                          const coords = event.currentTarget.getBoundingClientRect();
                          showTooltip({
                            tooltipData: { key, value, scenario: barGroup.x0 },
                            tooltipLeft: coords.x + bar.width / 2,
                            tooltipTop: coords.y - 10,
                          });
                        }}
                        onMouseLeave={hideTooltip}
                      />
                    )
                  )}
                </Group>
              ))
            }
          </BarGroup>
        </Group>
        <AxisBottom
          top={yMax + margin.top}
          scale={dateScale}
          stroke={green}
          tickStroke={green}
          hideAxisLine
          tickLabelProps={{
            fill: green,
            fontSize: 11,
            textAnchor: 'middle',
          }}
        />
        <g transform={`translate(${margin.left + 10}, ${margin.top - 25})`}>
          {keys.map((key, i) => (
            <g key={key} transform={`translate(${i * 80}, 0)`}>
              <rect width={10} height={10} fill={colorScale(key)} rx={2} />
              <text x={14} y={10} fontSize={10} fill="#ffffff">
                {key}
              </text>
            </g>
          ))}
        </g>
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
          <div><strong>{tooltipData.key}</strong></div>
          {/*<div>Scenario: {tooltipData.scenario}</div>*/}
          <div>Value: {tooltipData.value.toFixed(2)}</div>
        </TooltipWithBounds>
      )}
    </div>
  );
}
