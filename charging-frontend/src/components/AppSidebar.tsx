import * as React from "react"
import { Link, useLocation } from "react-router-dom"
import { useState } from "react"
import {
  LayoutDashboard,
  BarChart2,
  Bot,
  Car,
  SlidersHorizontal,
  Settings2,
  Menu,
  ArrowLeft,
} from "lucide-react"

// 模拟 shadcn 的 Sidebar 结构，使用原生 HTML + CSS 模拟样式结构
// 替代原 <a href="">，使用 React Router 的 <Link to="">，实现页面不刷新的路由跳转

export function AppSidebar() {
  const [collapsed, setCollapsed] = useState(false) // 折叠状态
  const location = useLocation() // 当前路径，用于高亮当前项

  // 菜单项定义（含图标）
  const menuItems = [
    { label: "Overview", to: "/dashboard", icon: LayoutDashboard },
    { label: "System Metrics", to: "/metrics", icon: BarChart2 },
    { label: "Robots", to: "/robots", icon: Bot },
    { label: "Vehicles", to: "/vehicles", icon: Car },
    { label: "Strategy Compare", to: "/strategies", icon: SlidersHorizontal },
    { label: "Settings", to: "/settings", icon: Settings2 },
  ]

  return (
    <div
      style={{
        width: collapsed ? "60px" : "240px",
        height: "100vh",
        backgroundColor: "#f8f9fa",
        borderRight: "1px solid #dee2e6",
        padding: collapsed ? "0.5rem" : "1rem",
        boxSizing: "border-box",
        transition: "width 0.3s ease",
        display: "flex",
        flexDirection: "column",
        alignItems: collapsed ? "center" : "flex-start",
      }}
    >
      {/* 折叠按钮区域 */}
      <div style={{ width: "100%", marginBottom: "1rem", textAlign: collapsed ? "center" : "right" }}>
        <button
          onClick={() => setCollapsed(!collapsed)}
          style={{
            backgroundColor: "transparent",
            border: "none",
            cursor: "pointer",
            fontSize: "1.25rem",
            padding: "0.25rem",
          }}
          title={collapsed ? "Expand" : "Collapse"}
        >
          {collapsed ? <Menu size={20} /> : <ArrowLeft size={20} />}
        </button>
      </div>

      {/* Header 区域 */}
      {!collapsed && (
        <div style={{ marginBottom: "1rem" }}>
          <h2 style={{ fontSize: "1.25rem", fontWeight: "bold", display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <LayoutDashboard size={20} /> Robot Simulation
          </h2>
        </div>
      )}

      {/* 导航组标签 */}
      {!collapsed && (
        <div style={{ fontSize: "0.75rem", color: "#6c757d", textTransform: "uppercase", marginBottom: "0.5rem" }}>
          Guidance
        </div>
      )}

      {/* 菜单列表：使用 Link 替代 a，实现内部路由跳转 */}
      <ul style={{ listStyle: "none", padding: 0, margin: 0, width: "100%" }}>
        {menuItems.map(({ label, to, icon: Icon }) => (
          <li key={to} style={{ marginBottom: "0.5rem", width: "100%" }}>
            <Link
              to={to}
              style={{
                display: "flex",
                alignItems: "center",
                gap: "0.75rem",
                padding: collapsed ? "0.5rem" : "0.5rem 0.75rem",
                borderRadius: "6px",
                color: location.pathname === to ? "#0d6efd" : "#212529",
                backgroundColor: location.pathname === to ? "#e9f2ff" : "transparent",
                textDecoration: "none",
                fontSize: collapsed ? "0.9rem" : "0.95rem",
                fontWeight: location.pathname === to ? 600 : 500,
                textAlign: collapsed ? "center" : "left",
                justifyContent: collapsed ? "center" : "flex-start",
              }}
              onMouseOver={(e) => (e.currentTarget.style.backgroundColor = "#e2e6ea")}
              onMouseOut={(e) => {
                if (location.pathname !== to) {
                  e.currentTarget.style.backgroundColor = "transparent"
                }
              }}
            >
              <Icon size={18} />
              {!collapsed && label}
            </Link>
          </li>
        ))}
      </ul>
    </div>
  )
}
