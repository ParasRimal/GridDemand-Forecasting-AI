import { useEffect, useState } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { getHistoricalDemand } from "../api/endpoints";

export default function HistoricalDemand() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    getHistoricalDemand()
      .then(setData)
      .catch((err) => setError(err.response?.data?.error || err.message));
  }, []);

  if (error) {
    return (
      <div className="max-w-2xl mx-auto mt-12 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg text-red-700 dark:text-red-300">
        Failed to load historical demand: {error}
      </div>
    );
  }
  if (!data) return <p className="text-center mt-12 text-gray-500">Loading {data ? "" : "(this reads the full dataset, may take a moment)"}...</p>;

  // Thin the x-axis labels so they don't overlap on a ~350-day chart.
  const tickInterval = Math.floor(data.length / 12);

  return (
    <div className="max-w-5xl mx-auto px-4 py-10">
      <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100 mb-2">Historical Demand</h1>
      <p className="text-sm text-gray-500 dark:text-gray-400 mb-8">
        Real daily electricity load (Ahmedabad, Nov 2021 - Nov 2022) from the dataset GridPredict's
        models were trained on. Notice the seasonal cycle: lowest in Nov-Jan, highest in Apr-Jun and Sep
        (see docs/MODELING_FINDINGS.md, section 1).
      </p>

      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis dataKey="date" interval={tickInterval} tick={{ fontSize: 11 }} />
            <YAxis label={{ value: "Load (MW)", angle: -90, position: "insideLeft" }} />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="max_load_mw" name="Daily max" stroke="#ef4444" dot={false} strokeWidth={1} />
            <Line type="monotone" dataKey="avg_load_mw" name="Daily avg" stroke="#3b82f6" dot={false} strokeWidth={2} />
            <Line type="monotone" dataKey="min_load_mw" name="Daily min" stroke="#22c55e" dot={false} strokeWidth={1} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <p className="text-xs text-gray-400 dark:text-gray-500 mt-4">
        {data.length} days of data shown, aggregated from hourly readings.
      </p>
    </div>
  );
}
