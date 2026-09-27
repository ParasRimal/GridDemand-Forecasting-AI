import { useEffect, useState } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { getModelVersions } from "../api/endpoints";

function Badge({ children, color = "gray" }) {
  const colors = {
    green: "bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-300",
    red: "bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300",
  };
  return (
    <span className={`inline-block px-2.5 py-0.5 rounded-full text-xs font-medium ${colors[color]}`}>
      {children}
    </span>
  );
}

export default function ModelHistory() {
  const [versions, setVersions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    getModelVersions()
      .then(setVersions)
      .catch((err) => setError(err.message));
  }, []);

  if (error) {
    return (
      <div className="max-w-2xl mx-auto mt-12 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg text-red-700 dark:text-red-300">
        Failed to load model history: {error}
      </div>
    );
  }
  if (!versions) return <p className="text-center mt-12 text-gray-500">Loading...</p>;

  const chartData = versions
    .filter((v) => v.mae !== null)
    .map((v) => ({ name: v.model_name, mae: v.mae, isProduction: v.is_current_production }));

  return (
    <div className="max-w-4xl mx-auto px-4 py-10">
      <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100 mb-2">Model History</h1>
      <p className="text-sm text-gray-500 dark:text-gray-400 mb-8">
        GridPredict trained three candidate models and selected the best on validation, but the
        naive baseline still won on the held-out test set. See docs/MODELING_FINDINGS.md for the full story.
      </p>

      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6 mb-6">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
          MAE Comparison (lower is better)
        </h2>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis dataKey="name" tick={{ fontSize: 12 }} />
            <YAxis label={{ value: "MAE (MW)", angle: -90, position: "insideLeft" }} />
            <Tooltip />
            <Bar dataKey="mae" name="MAE (MW)" radius={[4, 4, 0, 0]}>
              {chartData.map((entry, index) => (
                <Cell key={index} fill={entry.isProduction ? "#22c55e" : "#ef4444"} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">Decision Log</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left">
            <thead>
              <tr className="border-b border-gray-200 dark:border-gray-700 text-gray-500 dark:text-gray-400">
                <th className="py-2 pr-4 font-medium">Model</th>
                <th className="py-2 pr-4 font-medium">Status</th>
                <th className="py-2 pr-4 font-medium">MAE</th>
                <th className="py-2 font-medium">Reason</th>
              </tr>
            </thead>
            <tbody>
              {versions.map((v) => (
                <tr key={v.id} className="border-b border-gray-100 dark:border-gray-800">
                  <td className="py-3 pr-4 font-medium text-gray-900 dark:text-gray-100">{v.model_name}</td>
                  <td className="py-3 pr-4">
                    <Badge color={v.is_current_production ? "green" : "red"}>{v.status}</Badge>
                  </td>
                  <td className="py-3 pr-4 text-gray-600 dark:text-gray-400">{v.mae ?? "n/a"}</td>
                  <td className="py-3 text-gray-500 dark:text-gray-500 text-xs">{v.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
