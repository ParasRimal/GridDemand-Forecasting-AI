import { useEffect, useState } from "react";
import { getMonitoring } from "../api/endpoints";

function Badge({ children, color = "gray" }) {
  const colors = {
    green: "bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-300",
    red: "bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300",
    yellow: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/40 dark:text-yellow-300",
  };
  return (
    <span className={`inline-block px-2.5 py-0.5 rounded-full text-xs font-medium ${colors[color]}`}>
      {children}
    </span>
  );
}

export default function MonitoringHistory() {
  const [checks, setChecks] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    getMonitoring()
      .then(setChecks)
      .catch((err) => setError(err.message));
  }, []);

  if (error) {
    return (
      <div className="max-w-2xl mx-auto mt-12 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg text-red-700 dark:text-red-300">
        Failed to load monitoring history: {error}
      </div>
    );
  }
  if (!checks) return <p className="text-center mt-12 text-gray-500">Loading...</p>;

  return (
    <div className="max-w-4xl mx-auto px-4 py-10">
      <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100 mb-2">Monitoring History</h1>
      <p className="text-sm text-gray-500 dark:text-gray-400 mb-8">
        Every drift/performance check run by the scheduled Celery task (daily by default;
        see backend/monitoring/tasks.py). {checks.length} check(s) on record.
      </p>

      {checks.length === 0 ? (
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6 text-center text-gray-500 dark:text-gray-400">
          No monitoring checks have run yet. The Celery Beat scheduler runs this
          daily -- start a worker and beat process to see checks accumulate here.
        </div>
      ) : (
        <div className="space-y-4">
          {checks.map((c) => (
            <div
              key={c.id}
              className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6"
            >
              <div className="flex items-center justify-between mb-3">
                <span className="text-sm text-gray-500 dark:text-gray-400">
                  {new Date(c.checked_at).toLocaleString()}
                </span>
                <Badge color={c.retrain_recommended ? "yellow" : "green"}>
                  {c.retrain_recommended ? "Retrain recommended" : "No action needed"}
                </Badge>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-3">
                <div>
                  <p className="text-xs text-gray-400 dark:text-gray-500 uppercase tracking-wide">Drift</p>
                  <p className="font-medium text-gray-900 dark:text-gray-100">
                    {c.drift_detected ? "Detected" : "None"} ({(c.drifted_column_share * 100).toFixed(1)}%)
                  </p>
                  <p className="text-xs text-gray-400 dark:text-gray-500">
                    {c.drifted_column_count} / {c.n_columns} columns
                  </p>
                </div>
                <div>
                  <p className="text-xs text-gray-400 dark:text-gray-500 uppercase tracking-wide">Performance</p>
                  <p className="font-medium text-gray-900 dark:text-gray-100">
                    {c.performance_degraded ? "Degraded" : "Stable"}
                  </p>
                  {c.recent_mae !== null && (
                    <p className="text-xs text-gray-400 dark:text-gray-500">
                      MAE {c.recent_mae} vs ref {c.reference_mae}
                    </p>
                  )}
                </div>
                <div>
                  <p className="text-xs text-gray-400 dark:text-gray-500 uppercase tracking-wide">Decision</p>
                  <p className="font-medium text-gray-900 dark:text-gray-100">
                    {c.retrain_recommended ? "Retrain" : "Keep production model"}
                  </p>
                </div>
              </div>

              <p className="text-sm text-gray-600 dark:text-gray-300 border-t border-gray-100 dark:border-gray-800 pt-3">
                {c.reason}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
