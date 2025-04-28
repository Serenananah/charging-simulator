// src/pages/Overview.jsx
import React, { useEffect, useState } from 'react';
import MapView from '../components/MapView';
import TickPlayer from '../components/TickPlayer';
import InfoPanel from '../components/InfoPanel';
import TaskLegend from '../components/TaskLegend';
import { Zap, RefreshCcw } from 'lucide-react';
import { ToastContainer, toast } from 'react-toastify';
import { Dialog } from '@headlessui/react';
import { motion } from 'framer-motion';
import Skeleton from 'react-loading-skeleton';
import 'react-toastify/dist/ReactToastify.css';
import 'react-loading-skeleton/dist/skeleton.css';
import '../styles/map.css';
import '../styles/panel.css';

/**
 * Overview 页面：主调度页面
 * - 显示地图、播放控制、信息面板和图例
 * - 页面结构为三栏布局：左侧 Sidebar，中间地图，右侧 InfoPanel + TaskLegend 垂直排列
 */
export default function Overview() {
  const [currentState, setCurrentState] = useState(null);
  const [loading, setLoading] = useState(false);
  const [simulationFinished, setSimulationFinished] = useState(false);
  const [isDialogOpen, setIsDialogOpen] = useState(false);

  const [strategy, setStrategy] = useState('hungarian');
  const [scale, setScale] = useState('medium');

  // ✅ 初始化系统（传入当前选择的策略和规模）
  const initSystem = () => {
    setLoading(true);
    // 将strategy和scale动态拼接到URL
    fetch(`http://localhost:5050/api/init_map?strategy=${strategy}&scale=${scale}`)
      .then(res => res.json())
      .then(() => {
        setTimeout(fetchState, 100);
        setLoading(false);
        setSimulationFinished(false); // ✅ 初始化后，重置完成状态
      })
      .catch(err => {
        console.error("初始化系统失败:", err);
        toast.error("初始化系统失败，请检查后端日志！");
        setLoading(false);
      });
  };

  // 推进一帧
  const advanceStep = () => {
    fetch('http://localhost:5050/api/next_step')
      .then(res => res.json())
      .then((data) => {
        if (data.done) {
          setIsDialogOpen(true);
          setSimulationFinished(true);
          return;
        }
        fetchState();
      })
      .catch(err => console.error("推进失败:", err));
  };

  // 拉取当前状态
  const fetchState = () => {
    fetch('http://localhost:5050/api/get_state')
      .then(res => res.json())
      .then(data => {
        setCurrentState(data);
      })
      .catch(err => console.error("获取状态失败:", err));
  };

  useEffect(() => {
    initSystem();
  }, []); // 页面加载时初始化一次

  // ✅ 监听策略或规模变化，重新初始化系统
    {/*
  useEffect(() => {
    initSystem();
  }, [strategy, scale]);
    */}
  return (
    <div style={{
      minHeight: '100vh',
      backgroundColor: '#f8fafc',
      padding: '1.5rem',
      display: 'flex',
      flexDirection: 'column',
      gap: '1.5rem'
    }}>

      {/* 弹窗提示 */}
      <Dialog open={isDialogOpen} onClose={() => setIsDialogOpen(false)}>
        <div className="fixed inset-0 flex items-center justify-center bg-black/30">
          <Dialog.Panel className="bg-white p-6 rounded-xl shadow-xl">
            <Dialog.Title className="text-xl font-bold mb-4">🎉 任务全部完成！</Dialog.Title>
            <button
              onClick={() => setIsDialogOpen(false)}
              className="mt-4 bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
            >关闭</button>
          </Dialog.Panel>
        </div>
      </Dialog>

      {/* 顶部标题，居中单独一行 */}
      <h1 style={{
        textAlign: 'center',
        fontSize: '2rem',
        fontWeight: '800',
        color: '#1e293b'
      }}>
        <Zap size={28} style={{ marginBottom: '-4px' }} /> Autonomous Charging Robot Simulator
      </h1>

      {/* 中间主内容区域：地图+信息栏两栏式布局 */}
      <div style={{
        display: 'flex',
        flex: 1,
        gap: '2rem',
        alignItems: 'flex-start',
        maxWidth: '1440px',
        margin: '0 auto',
        width: '100%'
      }}>

        {/* 左侧地图区域，包含上方控制栏 */}
        <div style={{ flex: 3, display: 'flex', flexDirection: 'column', gap: '1rem' }}>

          {/* 上方控制栏：Start按钮+策略选择等控件 */}
          <div style={{
              background: '#ffffff',
              borderRadius: '0.75rem',
              padding: '1rem',
              boxShadow: '0 4px 12px rgba(0,0,0,0.08)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: '1rem'
          }}>
              <div style={{display: 'flex', alignItems: 'center', gap: '0.5rem'}}>
                  {/* 策略和规模选择框 */}
                  <label style={{ display: 'flex', alignItems: 'center', fontWeight: '500', color: '#334155' }}>
                      Strategy:&nbsp;
                      <select
                          value={strategy}
                          onChange={(e) => setStrategy(e.target.value)}
                          style={{ borderRadius: '0.5rem', padding: '0.25rem 0.5rem' }}
                      >
                      <option value="hungarian">Hungarian</option>
                      <option value="ppo">PPO (Reinforcement Learning)</option>
                      </select>
                  </label>
                  <label style={{ display: 'flex', alignItems: 'center', fontWeight: '500', color: '#334155' }}>
                      Scale:&nbsp;
                      <select
                          value={scale}
                          onChange={(e) => setScale(e.target.value)}
                          style={{ borderRadius: '0.5rem', padding: '0.25rem 0.5rem' }}
                      >
                      <option value="small">Small</option>
                      <option value="medium">Medium</option>
                      <option value="large">Large</option>
                      </select>
                  </label>
              </div>

              <button
                  onClick={() => {
                      // initSystem();
                      setTimeout(initSystem, 0);
                  }}
                  disabled={loading}
                  style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.5rem',
                      backgroundColor: '#3b82f6',
                      color: '#ffffff',
                      padding: '0.6rem 1.2rem',
                      borderRadius: '0.75rem',
                      fontWeight: '600',
                      fontSize: '1rem',
                      border: 'none',
                      cursor: 'pointer',
                      transition: 'background-color 0.3s ease'
                  }}
                  onMouseEnter={e => e.target.style.backgroundColor = '#2563eb'}
                  onMouseLeave={e => e.target.style.backgroundColor = '#3b82f6'}
              >
                  <RefreshCcw size={16}/> Start New Task
              </button>
          </div>

          {/* 地图区域主体 */}
          <div style={{
              background: '#ffffff',
              borderRadius: '1rem',
              padding: '1rem',
              boxShadow: '0 4px 12px rgba(0,0,0,0.08)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              minHeight: '600px'
          }}>
              {currentState ? <MapView state={currentState}/> : <Skeleton height={600}/>}
          </div>
        </div>

        {/* 右侧信息栏：统一卡片风格 */}
        <div style={{flex: 1.2, display: 'flex', flexDirection: 'column', gap: '1rem'}}>
          <motion.div initial={{opacity: 0}} animate={{opacity: 1}} transition={{duration: 0.5}} style={{
              background: '#ffffff',
              borderRadius: '1rem',
              padding: '1rem',
              boxShadow: '0 4px 12px rgba(0,0,0,0.08)'
          }}>
            <TickPlayer
              onAdvance={advanceStep}
              tick={currentState?.tick || 0}
              finished={simulationFinished}
              progress={currentState?.metrics ? (currentState.metrics.completed / currentState.tasks.length) * 100 : 0}
            />
          </motion.div>

          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.5, delay: 0.2 }} style={{ background: '#ffffff', borderRadius: '1rem', padding: '1rem', boxShadow: '0 4px 12px rgba(0,0,0,0.08)' }}>
              <InfoPanel metrics={currentState?.metrics || {}} strategy={currentState?.strategy || "hungarian"} scale={currentState?.scale || "medium"} />
          </motion.div>

          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.5, delay: 0.4 }} style={{ background: '#ffffff', borderRadius: '1rem', padding: '1rem', boxShadow: '0 4px 12px rgba(0,0,0,0.08)' }}>
            <TaskLegend />
          </motion.div>
        </div>
      </div>

      {/* Toast 容器 */}
      <ToastContainer />
    </div>
  );
}
