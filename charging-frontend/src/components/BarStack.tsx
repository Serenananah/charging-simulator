import React, { useState } from 'react';
import { BarStack } from '@visx/shape';
import { Group } from '@visx/group';
import { Grid } from '@visx/grid';
import { AxisBottom } from '@visx/axis';
import { scaleBand, scaleLinear, scaleOrdinal } from '@visx/scale';
import { useTooltip, useTooltipInPortal, defaultStyles } from '@visx/tooltip';
import { LegendOrdinal } from '@visx/legend';
import { localPoint } from '@visx/event';

const purple = '#a44afe';
const orange = '#f6a623';
const red = '#f44242';
export const background = '#eaedff';
const defaultMargin = { top: 60, right: 20, bottom: 50, left: 30 };
const tooltipStyles = {
  ...defaultStyles,
  minWidth: 60,
  backgroundColor: 'rgba(0,0,0,0.9)',
  color: 'white',
};

const strategies = ['Hungarian', 'PPO', 'HSPSO'];
const colors = ['#6c5efb', '#c998ff']; // Completion / Timeout

const allData = {
  Hungarian: [
    { scenario: 'E1', Completion: 97.00, Timeout: 3.00 },
    { scenario: 'E2', Completion: 62.91, Timeout: 37.09 },
    { scenario: 'E3', Completion: 62.05, Timeout: 37.95 },
    { scenario: 'E4', Completion: 70.22, Timeout: 29.78 },
    { scenario: 'E5', Completion: 88.87, Timeout: 11.13 },
    { scenario: 'E6', Completion: 63.87, Timeout: 36.13 },
  ],
  PPO: [
    { scenario: 'E1', Completion: 100.00, Timeout: 0.00 },
    { scenario: 'E2', Completion: 65.42, Timeout: 34.58 },
    { scenario: 'E3', Completion: 67.75, Timeout: 32.25 },
    { scenario: 'E4', Completion: 81.83, Timeout: 18.17 },
    { scenario: 'E5', Completion: 99.77, Timeout: 0.23 },
    { scenario: 'E6', Completion: 71.64, Timeout: 28.36 },
  ],
  HSPSO: [
    { scenario: 'E1', Completion: 90.149, Timeout: 9.78 },
    { scenario: 'E2', Completion: 90.39, Timeout: 9.61 },
    { scenario: 'E3', Completion: 89.39, Timeout: 10.68 },
    { scenario: 'E4', Completion: 90.65, Timeout: 9.35 },
    { scenario: 'E5', Completion: 88.22, Timeout: 11.81 },
    { scenario: 'E6', Completion: 89.13, Timeout: 10.87 },
  ],
};

export default function CompletionBarStack({ width, height }: { width: number; height: number }) {
  const [selected, setSelected] = useState('Hungarian');
  const data = allData[selected];
  const keys = ['Completion', 'Timeout'];

  const { tooltipOpen, tooltipLeft, tooltipTop, tooltipData, hideTooltip, showTooltip } =
    useTooltip<any>();
  const { containerRef, TooltipInPortal } = useTooltipInPortal({ scroll: true });

  const xMax = width - defaultMargin.left - defaultMargin.right;
  const yMax = height - defaultMargin.top - defaultMargin.bottom;

  const xScale = scaleBand({
    domain: data.map((d) => d.scenario),
    padding: 0.2,
    range: [0, xMax],
  });
  const yScale = scaleLinear({
    domain: [0, 110],
    nice: true,
    range: [yMax, 0],
  });
  const colorScale = scaleOrdinal({ domain: keys, range: colors });

  return (
      <div>
        <svg ref={containerRef} width={width} height={height}>
          <rect x={0} y={0} width={width} height={height} fill={background} rx={14}/>
          <Group top={defaultMargin.top} left={defaultMargin.left}>
            <Grid
                xScale={xScale}
                yScale={yScale}
                width={xMax}
                height={yMax}
                stroke="black"
                strokeOpacity={0.1}
            />
            <BarStack data={data} keys={keys} x={(d) => d.scenario} xScale={xScale} yScale={yScale} color={colorScale}>
              {(barStacks) =>
                  barStacks.map((barStack) =>
                      barStack.bars.map((bar) => (
                          <rect
                              key={`bar-${bar.key}-${bar.index}`}
                              x={bar.x}
                              y={bar.y}
                              height={bar.height}
                              width={bar.width}
                              fill={bar.color}
                              onMouseMove={(event) => {
                                const coords = localPoint(event);
                                showTooltip({
                                  tooltipData: bar,
                                  tooltipTop: coords?.y,
                                  tooltipLeft: coords?.x,
                                });
                              }}
                              onMouseLeave={hideTooltip}
                          />
                      ))
                  )
              }
            </BarStack>
            <AxisBottom
                top={yMax}
                scale={xScale}
                stroke={purple}
                tickStroke={purple}
                tickLabelProps={{
                  fill: purple,
                  fontSize: 11,
                  textAnchor: 'middle',
                }}
            />
          </Group>
          <Group top={20} left={30}>
            {keys.map((key, i) => (
                <g key={key} transform={`translate(0, ${i * 20})`}>
                  <rect width={12} height={12} fill={colorScale(key)}/>
                  <text x={18} y={12} fontSize={11} fill="#333">
                    {key}
                  </text>
                </g>
            ))}
          </Group>
        </svg>
        <div style={{textAlign: 'center', marginBottom: '8px'}}>
          {strategies.map((s) => (
              <button
                  key={s}
                  onClick={() => setSelected(s)}
                  style={{
                    margin: '0 4px',
                    padding: '2px 6px',
                    fontSize: '12px',
                    backgroundColor: s === selected ? '#a44afe' : '#f0f0f0',
                    color: s === selected ? 'white' : 'black',
                    border: 'none',
                    borderRadius: 3,
                    cursor: 'pointer',
                  }}
              >
                {s}
              </button>
          ))}
        </div>
        {tooltipOpen && tooltipData && (
            <TooltipInPortal top={tooltipTop} left={tooltipLeft} style={tooltipStyles}>
              <div style={{color: tooltipData.color}}>
                <strong>{tooltipData.key}</strong>
              </div>
              <div>{tooltipData.bar.data[tooltipData.key]}%</div>
              <div>
                <small>{tooltipData.bar.data.scenario}</small>
              </div>
            </TooltipInPortal>
        )}
      </div>
  );
}
