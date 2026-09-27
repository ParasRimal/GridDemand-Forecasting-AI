import { useEffect, useState } from "react";
import { getForecasts, getModelVersions, getMonitoring } from "../api/endpoints";

function Card({ title, children }) {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
      <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">{title}</h2>
      {children}
    </div>
  );
}

function Badge({ children, color = "gray" }) {
  const colors = {
    green: "bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-300",
    red: "bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300",
    gray: "bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300",
    yellow: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/40 dark:text-yellow-300",
  };
  return (
    <span className={`inline-block px-2.5 py-0.5 rounded-full text-xs font-medium ${colors[color]}`}>
      {children}
    </span>
  );
}

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
    return (
      <div className="max-w-2xl mx-auto mt-12 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg text-red-700 dark:text-red-300">
        Failed to load dashboard: {error}
      </div>
    );
  }

  const latestCheck = monitoring && monitoring.length > 0 ? monitoring[0] : null;
  const currentProduction = modelVersions
    ? modelVersions.find((mv) => mv.is_current_production)
    : null;

  return (
    <div className="max-w-5xl mx-auto px-4 py-10">
      <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100 mb-8">
        GridPredict Dashboard
      </h1>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card title="Production Model">
          {currentProduction ? (
            <div className="space-y-2">
              <p className="text-2xl font-semibold text-gray-900 dark:text-gray-100">
                {currentProduction.model_name}
              </p>
              <Badge color={currentProduction.model_type === "baseline" ? "gray" : "green"}>
                {currentProduction.model_type}
              </Badge>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                MAE: <span className="font-medium text-gray-700 dark:text-gray-300">{currentProduction.mae ?? "n/a"} MW</span>
              </p>
            </div>
          ) : (
            <p className="text-gray-500 dark:text-gray-400">No production model recorded yet.</p>
          )}
        </Card>

        <Card title="Latest Drift / Retraining Check">
          {latestCheck ? (
            <div className="space-y-2 text-sm">
              <p className="text-gray-500 dark:text-gray-400">
                Checked: {new Date(latestCheck.checked_at).toLocaleString()}
              </p>
              <div className="flex flex-wrap gap-2">
                <Badge color={latestCheck.drift_detected ? "red" : "green"}>
                  Drift: {latestCheck.drift_detected ? "Yes" : "No"} ({(latestCheck.drifted_column_share * 100).toFixed(1)}%)
                </Badge>
                <Badge color={latestCheck.performance_degraded ? "red" : "green"}>
                  Degraded: {latestCheck.performance_degraded ? "Yes" : "No"}
                </Badge>
                <Badge color={latestCheck.retrain_recommended ? "yellow" : "green"}>
                  Retrain: {latestCheck.retrain_recommended ? "Yes" : "No"}
                </Badge>
              </div>
              <p className="text-gray-600 dark:text-gray-300 mt-2">{latestCheck.reason}</p>
            </div>
          ) : (
            <p className="text-gray-500 dark:text-gray-400">No monitoring checks recorded yet.</p>
          )}
        </Card>
      </div>

      <div className="mt-6">
        <Card title="Model Comparison History">
          {modelVersions && modelVersions.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full text-sm text-left">
                <thead>
                  <tr className="border-b border-gray-200 dark:border-gray-700 text-gray-500 dark:text-gray-400">
                    <th className="py-2 pr-4 font-medium">Model</th>
                    <th className="py-2 pr-4 font-medium">Type</th>
                    <th className="py-2 pr-4 font-medium">Status</th>
                    <th className="py-2 pr-4 font-medium">MAE</th>
                  </tr>
                </thead>
                <tbody>
                  {modelVersions.map((mv) => (
                    <tr key={mv.id} className="border-b border-gray-100 dark:border-gray-800">
                      <td className="py-2 pr-4 text-gray-900 dark:text-gray-100">{mv.model_name}</td>
                      <td className="py-2 pr-4 text-gray-600 dark:text-gray-400">{mv.model_type}</td>
                      <td className="py-2 pr-4">
                        <Badge color={mv.status === "production" ? "green" : "red"}>{mv.status}</Badge>
                      </td>
                      <td className="py-2 pr-4 text-gray-600 dark:text-gray-400">{mv.mae ?? "n/a"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="text-gray-500 dark:text-gray-400">No model version history recorded yet.</p>
          )}
        </Card>
      </div>

      <div className="mt-6">
        <Card title="Recent Forecasts">
          {forecasts && forecasts.length > 0 ? (
            <p className="text-gray-600 dark:text-gray-300">{forecasts.length} forecast(s) on record.</p>
          ) : (
            <p className="text-gray-500 dark:text-gray-400">No forecasts recorded yet.</p>
          )}
        </Card>
      </div>
    </div>
  );
}
