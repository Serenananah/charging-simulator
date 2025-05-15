// src/components/TickPlayer.jsx
import React, { useEffect, useRef, useState } from 'react';
import { FastForward, Play, PauseCircle } from 'lucide-react';
import { motion } from 'framer-motion'; // ✅ 引入 framer-motion 用于动效
import '../styles/panel.css';

/**
 * TickPlayer 组件（动态 API 版本）
 * 包含：播放/暂停按钮、后一帧控制、当前帧显示、自动播放逻辑，
 * 不再使用 tick 编号控制，而是通过 props.onAdvance() 每次推进一帧状态
 * Props:
 *  - tick: 当前帧编号（只用于展示）
 *  - onAdvance: 每次推进一帧的回调函数（由 App 传入，内部调用后端 /api/next_step）
 *  - finished: 是否全部任务已完成（新增，控制按钮状态）
 *  - progress: 完成任务百分比（新增，用于进度条）
 */
function TickPlayer({ tick, onAdvance, finished, progress }) {
  const [isPlaying, setIsPlaying] = useState(false); // 是否自动播放
  const intervalRef = useRef(null);                 // 自动播放定时器

  // 自动播放逻辑：每 300ms 触发一次 onAdvance()
  useEffect(() => {
    if (isPlaying && !finished) { // ✅ 新增：如果已完成，则停止播放
      intervalRef.current = setInterval(() => {
        onAdvance();
      }, 300);
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    }
    return () => clearInterval(intervalRef.current);
  }, [isPlaying, onAdvance, finished]);

  return (
    // 播放/暂停，下一帧，显示进度条
    <div
      className="tick-player"
      style={{
        marginTop: '-0.05rem',             // 整体卡片上移
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        width: '100%',
        padding: '0.75rem 1rem',
        borderRadius: '1rem',
        backgroundColor: '#f8fafc',
        boxShadow: '0 4px 8px rgba(0,0,0,0.08)',
        flexWrap: 'wrap',
        gap: '0.5rem'
      }}
    >
      {/* 播放按钮优先显示 */}
      <motion.button
        whileHover={!finished ? { scale: 1.05 } : {}} // ✅ hover微缩放
        onClick={() => { if (!finished) setIsPlaying((prev) => !prev); }} // ✅ 完成后禁止切换播放
        disabled={finished}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem',
          backgroundColor: finished ? '#e2e8f0' : (isPlaying ? '#fecaca' : '#d1fae5'),
          color: '#111827',
          border: 'none',
          borderRadius: '0.75rem',
          padding: '0.6rem 1.2rem',
          fontWeight: '600',
          cursor: finished ? 'not-allowed' : 'pointer',
          opacity: finished ? 0.6 : 1,
          boxShadow: finished ? 'none' : '0 2px 6px rgba(0,0,0,0.12)'
        }}
        title={finished ? 'Simulation Finished' : (isPlaying ? 'Pause Auto Play' : 'Start Auto Play')}
      >
        {isPlaying ? <PauseCircle size={20} /> : <Play size={20} />}
        {finished ? 'Finished' : (isPlaying ? 'Pause' : 'Play')}
      </motion.button>

      {/* 下一帧按钮 */}
      <motion.button
        whileHover={!isPlaying && !finished ? { scale: 1.05 } : {}} // ✅ hover微缩放
        onClick={() => onAdvance()}
        disabled={isPlaying || finished}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem',
          backgroundColor: (isPlaying || finished) ? '#f1f5f9' : '#e0f2fe',
          color: (isPlaying || finished) ? '#9ca3af' : '#1e3a8a',
          border: 'none',
          borderRadius: '0.75rem',
          padding: '0.6rem 1.2rem',
          fontWeight: '600',
          cursor: (isPlaying || finished) ? 'not-allowed' : 'pointer',
          opacity: (isPlaying || finished) ? 0.6 : 1,
          boxShadow: (isPlaying || finished) ? 'none' : '0 2px 6px rgba(0,0,0,0.12)'
        }}
        title={isPlaying ? "Playing..." : (finished ? "Simulation Finished" : "Advance one step")}
      >
        <FastForward size={20} />
        Next
      </motion.button>

      {/* 当前帧显示 */}
      <span style={{ fontWeight: 600, color: '#334155' }}>
        Tick: <strong>{tick}</strong>
      </span>

      {/* 进度条区域 */}
      <div style={{ width: '100%', marginTop: '0.5rem' }}>
        <div style={{
          backgroundColor: '#e2e8f0',
          borderRadius: '0.5rem',
          overflow: 'hidden',
          height: '8px'
        }}>
          <div style={{
            width: `${progress || 0}%`, // ✅ 动态宽度
            height: '8px',
            backgroundColor: '#38bdf8', // 浅蓝色进度条
            transition: 'width 0.3s ease'
          }} />
        </div>
        <div style={{ textAlign: 'right', marginTop: '0.25rem', fontSize: '0.75rem', color: '#64748b' }}>
          {progress ? `${progress.toFixed(1)}% Completed` : '0% Completed'}
        </div>
      </div>
    </div>
  );
}

export default TickPlayer;