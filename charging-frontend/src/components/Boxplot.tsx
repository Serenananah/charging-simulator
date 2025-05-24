import React, { useState } from 'react';
import { Group } from '@visx/group';
import { ViolinPlot, BoxPlot } from '@visx/stats';
import { LinearGradient } from '@visx/gradient';
import { scaleBand, scaleLinear } from '@visx/scale';
import { withTooltip, Tooltip, defaultStyles as defaultTooltipStyles } from '@visx/tooltip';
import { WithTooltipProvidedProps } from '@visx/tooltip/lib/enhancers/withTooltip';
import { PatternLines } from '@visx/pattern';
import { violinBoxData } from './processed_violin_boxplot';

const strategies = ['Hungarian', 'PPO', 'HSPSO'];
const scenes = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6'];
const dataKeyMap = {
  PPO: 'PPO',
  Hungarian: 'HUNGARIAN',
  HSPSO: 'HSPSO',
};


const customTooltipStyles = {
  ...defaultTooltipStyles,
  backgroundColor: '#283238',
  color: 'white',
};

export default withTooltip(({ width, height, tooltipOpen, tooltipLeft, tooltipTop, tooltipData, showTooltip, hideTooltip, }: WithTooltipProvidedProps<any> & { width: number; height: number; }) => {
  const [selected, setSelected] = useState('Hungarian');

  const data = violinBoxData[dataKeyMap[selected]];

  const xMax = width;
  const yMax = height - 120;

  const xScale = scaleBand<string>({
    range: [0, xMax],
    round: true,
    domain: scenes,
    padding: 0.4,
  });

  const allValues = data.flatMap(d => [d.min, d.q1, d.median, d.q3, d.max].filter(Number.isFinite)) as number[];
  const yScale = scaleLinear<number>({
    range: [yMax, 0],
    round: true,
    domain: [Math.min(...allValues), Math.max(...allValues)],
  });

  const boxWidth = xScale.bandwidth();
  const constrainedWidth = Math.min(40, boxWidth);

  return width < 10 ? null : (
    <div style={{ position: 'relative' }}>
      <svg width={width} height={height}>
        <LinearGradient id="statsplot" to="#8b6ce7" from="#87f2d4" />
        <rect x={0} y={0} width={width} height={height} fill="url(#statsplot)" rx={14} />
        <PatternLines id="hViolinLines" height={3} width={3} stroke="#ced4da" strokeWidth={1} orientation={['horizontal']} />
        <Group top={40}>
          {data.map((d: any, i: number) => d && d.median && (
            <g key={i}>
              <ViolinPlot
                data={d.binData}
                stroke="#dee2e6"
                left={xScale(scenes[i])!}
                width={constrainedWidth}
                valueScale={yScale}
                fill="url(#hViolinLines)"
              />
              <BoxPlot
                min={d.min}
                max={d.max}
                left={xScale(scenes[i])! + 0.3 * constrainedWidth}
                firstQuartile={d.q1}
                thirdQuartile={d.q3}
                median={d.median}
                boxWidth={constrainedWidth * 0.4}
                fill="#FFFFFF"
                fillOpacity={0.3}
                stroke="#FFFFFF"
                strokeWidth={2}
                valueScale={yScale}
                outliers={d.outliers || []}
                boxProps={{
                  onMouseOver: () => showTooltip({
                    tooltipTop: yScale(d.median) + 40,
                    tooltipLeft: xScale(scenes[i])! + 50,
                    tooltipData: { ...d, name: scenes[i] },
                  }),
                  onMouseLeave: hideTooltip,
                }}
                medianProps={{
                  style: { stroke: 'white' },
                  onMouseOver: () => showTooltip({
                    tooltipTop: yScale(d.median) + 40,
                    tooltipLeft: xScale(scenes[i])! + 50,
                    tooltipData: { median: d.median, name: scenes[i] },
                  }),
                  onMouseLeave: hideTooltip,
                }}
              />
            </g>
          ))}
        </Group>
      </svg>
      <div style={{ textAlign: 'center', marginBottom: '8px' }}>
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
            }}>
            {s}
          </button>
        ))}
      </div>
      {tooltipOpen && tooltipData && (
        <Tooltip top={tooltipTop} left={tooltipLeft} style={customTooltipStyles}>
          <div><strong>{tooltipData.name}</strong></div>
          <div style={{ marginTop: '5px', fontSize: '12px' }}>
            {tooltipData.max && <div>max: {tooltipData.max}</div>}
            {tooltipData.q3 && <div>third quartile: {tooltipData.q3}</div>}
            {tooltipData.median && <div>median: {tooltipData.median}</div>}
            {tooltipData.q1 && <div>first quartile: {tooltipData.q1}</div>}
            {tooltipData.min && <div>min: {tooltipData.min}</div>}
          </div>
        </Tooltip>
      )}
    </div>
  );
});
