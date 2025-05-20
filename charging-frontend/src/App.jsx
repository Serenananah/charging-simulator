// src/App.jsx

// 引入 React Router 的核心组件
import { BrowserRouter as Router, Routes, Route, Navigate} from "react-router-dom"

// 引入自定义的侧边栏组件（shadcn 结构）
import { AppSidebar } from "./components/AppSidebar"

// 引入你创建的各个页面（当前只有 Overview 主页面）
import Overview from "./pages/Overview"

// ⚠️ 后续你也会添加新的页面，比如：
import Metrics from "./pages/SystemMetrics"
// import Robots from "./pages/Robots"
// import Settings from "./pages/Settings"

/**
 * App.jsx: 项目总入口
 * - 控制全局布局（左侧 Sidebar + 右侧内容）
 * - 配置页面路由（每个路径展示不同页面组件）
 */
function App() {
  return (
    // Router 组件：提供整个项目的前端路由能力
    <Router>
      {/* 使用 flex 布局，左侧为 Sidebar，右侧为动态内容区域 */}
      <div style={{ display: 'flex', height: '100vh' }}>

        {/* 左侧固定导航栏 */}
        <AppSidebar />

        {/* 右侧区域：根据路由展示内容 */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '2rem' }}>
          <Routes>
            {/* 默认访问 / 自动跳转到 /dashboard */}
            <Route path="/" element={<Navigate to="/dashboard" />} />

            {/* 概览主页面 */}
            <Route path="/dashboard" element={<Overview />} />

            {/* 🚧 示例：你未来可以在这里添加更多页面，比如： */}
            {/* 页面路径 "/metrics" 显示 Metrics 页面 */}
            <Route path="/metrics" element={<Metrics />} />

            {/* 页面路径 "/robots" 显示 Robots 页面 */}
            {/* <Route path="/robots" element={<Robots />} /> */}

            {/* 页面路径 "/settings" 显示 Settings 设置页面 */}
            {/* <Route path="/settings" element={<Settings />} /> */}
          </Routes>
        </div>
      </div>
    </Router>
  )
}

export default App