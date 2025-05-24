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

export default function Overview() {
  const [currentState, setCurrentState] = useState(null);
  const [loading, setLoading] = useState(false);
  const [simulationFinished, setSimulationFinished] = useState(false);
  const [isDialogOpen, setIsDialogOpen] = useState(false);

  const [strategy, setStrategy] = useState('hungarian');
  const [scale, setScale] = useState('medium');
  const [distribution, setDistribution] = useState('clustered'); // ⭐ 新增参数
  const [arrivalMode, setArrivalMode] = useState('poisson');     // ⭐ 新增参数

  const initSystem = () => {
    setLoading(true);
    fetch(`http://localhost:5050/api/init_map?strategy=${strategy}&scale=${scale}&distribution=${distribution}&arrival_mode=${arrivalMode}`)
      .then(res => res.json())
      .then(() => {
        setTimeout(fetchState, 100);
        setLoading(false);
        setSimulationFinished(false);
      })
      .catch(err => {
        console.error("初始化系统失败:", err);
        toast.error("初始化系统失败，请检查后端日志！");
        setLoading(false);
      });
  };

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
  }, []);

  return (
    <div style={{ minHeight: '100vh', backgroundColor: '#f8fafc', padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>

      <Dialog open={isDialogOpen} onClose={() => setIsDialogOpen(false)}>
        <div className="fixed inset-0 flex items-center justify-center bg-black/30">
          <Dialog.Panel className="bg-white p-6 rounded-xl shadow-xl">
            <Dialog.Title className="text-xl font-bold mb-4"> 任务全部完成！</Dialog.Title>
            <button onClick={() => setIsDialogOpen(false)} className="mt-4 bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600">关闭</button>
          </Dialog.Panel>
        </div>
      </Dialog>

      <h1 style={{ textAlign: 'center', fontSize: '2rem', fontWeight: '800', color: '#1e293b' }}>
        <Zap size={28} style={{ marginBottom: '-4px' }} /> Autonomous Charging Robot Simulator
      </h1>

      <div style={{ display: 'flex', flex: 1, gap: '2rem', alignItems: 'flex-start', maxWidth: '1440px', margin: '0 auto', width: '100%' }}>

        <div style={{flex: 3, display: 'flex', flexDirection: 'column', gap: '1rem'}}>
          <div style={{ background: '#ffffff', borderRadius: '0.75rem', padding: '1rem', boxShadow: '0 4px 12px rgba(0,0,0,0.08)', display: 'flex', flexWrap: 'wrap', gap: '1rem' }}>
            {[{ label: 'Strategy', value: strategy, setter: setStrategy, options: ['hungarian', 'ppo', 'hspso'] },
              { label: 'Scale', value: scale, setter: setScale, options: ['small', 'medium', 'large'] },
              { label: 'Distribution', value: distribution, setter: setDistribution, options: ['uniform', 'clustered', 'mixed'] },
              { label: 'Arrival Mode', value: arrivalMode, setter: setArrivalMode, options: ['poisson', 'uniform', 'normal'] }].map(({label, value, setter, options}) => (
              // ... rest of the code
              <div key={label} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flex: '1 1 0', minWidth: '180px' }}>
                <label style={{ fontWeight: '500', color: '#334155' }}>{label}:</label>
                <select value={value} onChange={(e) => setter(e.target.value)} style={{ borderRadius: '0.5rem', padding: '0.4rem 0.6rem', flex: 1 }}>
                  {options.map(opt => <option key={opt} value={opt}>{opt}</option>)}
                </select>
              </div>
            ))}

            <button onClick={() => setTimeout(initSystem, 0)} disabled={loading} style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem', backgroundColor: '#3b82f6', color: '#ffffff', padding: '0.6rem 1.2rem', borderRadius: '0.75rem', fontWeight: '600', fontSize: '1rem', border: 'none', cursor: 'pointer', flex: '1 1 0', minWidth: '180px', whiteSpace: 'nowrap', transition: 'background-color 0.3s ease' }} onMouseEnter={(e) => (e.target.style.backgroundColor = '#2563eb')} onMouseLeave={(e) => (e.target.style.backgroundColor = '#3b82f6')}>
              <RefreshCcw size={16} /> Start New Task
            </button>
          </div>

          <div style={{ background: '#ffffff', borderRadius: '1rem', padding: '1rem', boxShadow: '0 4px 12px rgba(0,0,0,0.08)', display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '600px' }}>
            {currentState ? <MapView state={currentState}/> : <Skeleton height={600}/>}
          </div>
        </div>

        <div style={{flex: 1.2, display: 'flex', flexDirection: 'column', gap: '1rem'}}>
          <motion.div initial={{opacity: 0}} animate={{opacity: 1}} transition={{duration: 0.5}} style={{ background: '#ffffff', borderRadius: '1rem', padding: '1rem', boxShadow: '0 4px 12px rgba(0,0,0,0.08)' }}>
            <TickPlayer onAdvance={advanceStep} tick={currentState?.tick || 0} finished={simulationFinished} progress={currentState?.metrics ? (currentState.counts.completed / currentState.tasks.length) * 100 : 0} />
          </motion.div>

          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.5, delay: 0.2 }} style={{ background: '#ffffff', borderRadius: '1rem', padding: '1rem', boxShadow: '0 4px 12px rgba(0,0,0,0.08)' }}>
            <InfoPanel metrics={currentState?.metrics || {}} strategy={currentState?.strategy || "hungarian"} scale={currentState?.scale || "medium"} />
          </motion.div>

          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.5, delay: 0.4 }} style={{ background: '#ffffff', borderRadius: '1rem', padding: '1rem', boxShadow: '0 4px 12px rgba(0,0,0,0.08)' }}>
            <TaskLegend />
          </motion.div>
        </div>
      </div>

      <ToastContainer />
    </div>
  );
}