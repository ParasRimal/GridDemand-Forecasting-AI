import { useEffect, useState } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { getFeatureImportance } from "../api/endpoints";

export default function FeatureImportance() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    getFeatureImportance()
      .then(setData)
      .catch((err) => setError(err.response?.data?.error || err.message));
  }, []);

  if (error) {
    return (
      <div className="max-w-2xl mx-auto mt-12 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg text-red-700 dark:text-red-300">
        Failed to load feature importance: {error}
      </div>
    );
  }
  if (!data) return <p className="text-center mt-12 text-gray-500">Loading...</p>;

  const chartData = [...data.importances].reverse(); // so the bar chart reads top-to-bottom, biggest first

  return (
    <div className="max-w-4xl mx-auto px-4 py-10">
      <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100 mb-2">Feature Importance</h1>
      <p className="text-sm text-gray-500 dark:text-gray-400 mb-8">
        Real importances from the saved <span className="font-mono text-xs bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">{data.model_name}</span> candidate
        model ({data.trained_on}). Higher means the model relies on that feature more when making predictions.
      </p>

      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <ResponsiveContainer width="100%" height={Math.max(400, chartData.length * 32)}>
          <BarChart data={chartData} layout="vertical" margin={{ left: 40 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis type="number" domain={[0, "dataMax"]} />
            <YAxis type="category" dataKey="feature" width={140} tick={{ fontSize: 12 }} />
            <Tooltip />
            <Bar dataKey="importance" fill="#3b82f6" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <p className="text-xs text-gray-400 dark:text-gray-500 mt-4">
        Note: lag_24 dominating confirms what the model selection process found in
        docs/MODELING_FINDINGS.md -- recent load level matters most, which is also why the naive
        "same hour yesterday" baseline is a surprisingly strong competitor.
      </p>
    </div>
  );
}
