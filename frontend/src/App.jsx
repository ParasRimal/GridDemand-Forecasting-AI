import { BrowserRouter, Routes, Route, Link } from "react-router-dom";
import Dashboard from "./pages/Dashboard";
import TryForecast from "./pages/TryForecast";
import "./App.css";

function App() {
  return (
    <BrowserRouter>
      <nav>
        <Link to="/">Dashboard</Link> | <Link to="/forecast">Try a Forecast</Link>
      </nav>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/forecast" element={<TryForecast />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
