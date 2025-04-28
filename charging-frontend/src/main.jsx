// src/main.jsx
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.jsx';
import './styles/map.css';
import './styles/panel.css';

/**
 * main.jsx 是整个前端项目的入口文件。
 * 作用：
 * - 将 <App /> 组件挂载到 HTML 页面中的 #root 元素上。
 * - 可在此配置全局样式、状态管理、主题切换等初始化内容。
 */
ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

