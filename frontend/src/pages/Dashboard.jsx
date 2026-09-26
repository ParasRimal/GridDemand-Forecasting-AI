import { useEffect, useState } from "react";
import { getForecasts, getModelVersions, getMonitoring } from "../api/endpoints";

export default function Dashboard() {
  const [forecasts, setForecasts] = useState(null);
  const [modelVersions, setModelVersions] = useState(null);
  const [monitoring, setMonitoring] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([getForecasts(), getModelVersions(), getMonitoring()])
      .then(([f, mv, m]) => {
        setForecasts(f);
        setModelVersions(mv);
        setMonitoring(m);
      })
      .catch((err) => setError(err.message));
  }, []);

  if (error) {
    return <div className="dashboard-error">Failed to load dashboard: {error}</div>;
  }

  const latestCheck = monitoring && monitoring.length > 0 ? monitoring[0] : null;
  const currentProduction = modelVersions
    ? modelVersions.find((mv) => mv.is_current_production)
    : null;

  return (
    <div className="dashboard">
      <h1>GridPredict Dashboard</h1>

      <section>
        <h2>Production Model</h2>
        {currentProduction ? (
          <p>
            <strong>{currentProduction.model_name}</strong> ({currentProduction.model_type}) —
            MAE: {currentProduction.mae ?? "n/a"} MW
          </p>
        ) : (
          <p>No production model recorded yet in the database.</p>
        )}
      </section>

      <section>
        <h2>Model Comparison History</h2>
        {modelVersions && modelVersions.length > 0 ? (
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Type</th>
                <th>Status</th>
                <th>MAE</th>
              </tr>
            </thead>
            <tbody>
              {modelVersions.map((mv) => (
                <tr key={mv.id}>
                  <td>{mv.model_name}</td>
                  <td>{mv.model_type}</td>
                  <td>{mv.status}</td>
                  <td>{mv.mae ?? "n/a"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p>No model version history recorded yet.</p>
        )}
      </section>

      <section>
        <h2>Latest Drift / Retraining Check</h2>
        {latestCheck ? (
          <div>
            <p>Checked at: {new Date(latestCheck.checked_at).toLocaleString()}</p>
            <p>Drift detected: {latestCheck.drift_detected ? "Yes" : "No"} ({(latestCheck.drifted_column_share * 100).toFixed(1)}% of columns)</p>
            <p>Performance degraded: {latestCheck.performance_degraded ? "Yes" : "No"}</p>
            <p>Retrain recommended: {latestCheck.retrain_recommended ? "Yes" : "No"}</p>
            <p>{latestCheck.reason}</p>
          </div>
        ) : (
          <p>No monitoring checks recorded yet.</p>
        )}
      </section>

      <section>
        <h2>Recent Forecasts</h2>
        {forecasts && forecasts.length > 0 ? (
          <p>{forecasts.length} forecast(s) on record.</p>
        ) : (
          <p>No forecasts recorded yet.</p>
        )}
      </section>
    </div>
  );
}
