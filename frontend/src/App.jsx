import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";
import Dashboard from "./pages/Dashboard";
import TryForecast from "./pages/TryForecast";
import ModelHistory from "./pages/ModelHistory";
import SystemStatus from "./pages/SystemStatus";
import MonitoringHistory from "./pages/MonitoringHistory";
import HistoricalDemand from "./pages/HistoricalDemand";
import FeatureImportance from "./pages/FeatureImportance";
import About from "./pages/About";
import "./App.css";

function Navbar() {
  const linkClass = ({ isActive }) =>
    `px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
      isActive
        ? "bg-blue-600 text-white"
        : "text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800"
    }`;

  return (
    <nav className="bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-800 px-4 py-3">
      <div className="max-w-5xl mx-auto flex items-center gap-2">
        <span className="font-bold text-gray-900 dark:text-gray-100 mr-4">⚡ GridPredict</span>
        <NavLink to="/" end className={linkClass}>Dashboard</NavLink>
        <NavLink to="/forecast" className={linkClass}>Try a Forecast</NavLink>
        <NavLink to="/history" className={linkClass}>Model History</NavLink>
        <NavLink to="/status" className={linkClass}>System Status</NavLink>
        <NavLink to="/monitoring" className={linkClass}>Monitoring History</NavLink>
        <NavLink to="/demand" className={linkClass}>Historical Demand</NavLink>
        <NavLink to="/importance" className={linkClass}>Feature Importance</NavLink>
        <NavLink to="/about" className={linkClass}>About</NavLink>
      </div>
    </nav>
  );
}

function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-50 dark:bg-gray-950">
        <Navbar />
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/forecast" element={<TryForecast />} />
          <Route path="/history" element={<ModelHistory />} />
          <Route path="/status" element={<SystemStatus />} />
          <Route path="/monitoring" element={<MonitoringHistory />} />
          <Route path="/demand" element={<HistoricalDemand />} />
          <Route path="/importance" element={<FeatureImportance />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}

export default App;
