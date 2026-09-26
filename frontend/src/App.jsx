import { BrowserRouter, Routes, Route, Link } from "react-router-dom";
import Dashboard from "./pages/Dashboard";
import TryForecast from "./pages/TryForecast";
import ModelHistory from "./pages/ModelHistory";
import "./App.css";

function App() {
  return (
    <BrowserRouter>
      <nav>
        <Link to="/">Dashboard</Link> | <Link to="/forecast">Try a Forecast</Link> | <Link to="/history">Model History</Link>
      </nav>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/forecast" element={<TryForecast />} />
        <Route path="/history" element={<ModelHistory />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
