import { useEffect, useState } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from "recharts";
import { getModelVersions } from "../api/endpoints";

export default function ModelHistory() {
  const [versions, setVersions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    getModelVersions()
      .then(setVersions)
      .catch((err) => setError(err.message));
  }, []);

  if (error) return <div className="dashboard-error">Failed to load model history: {error}</div>;
  if (!versions) return <p>Loading...</p>;

  const chartData = versions
    .filter((v) => v.mae !== null)
    .map((v) => ({
      name: v.model_name,
      mae: v.mae,
      isProduction: v.is_current_production,
    }));

  return (
    <div className="model-history">
      <h1>Model History</h1>
      <p>
        GridPredict trained three candidate models (Random Forest, XGBoost, LightGBM) and
        selected the best on validation, but the naive baseline still won on the held-out
        test set. See docs/MODELING_FINDINGS.md for the full story.
      </p>

      <h2>MAE Comparison (lower is better)</h2>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" />
          <YAxis label={{ value: "MAE (MW)", angle: -90, position: "insideLeft" }} />
          <Tooltip />
          <Legend />
          <Bar dataKey="mae" name="MAE (MW)">
            {chartData.map((entry, index) => (
              <Cell key={index} fill={entry.isProduction ? "#2ecc71" : "#e74c3c"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      <h2>Decision Log</h2>
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Type</th>
            <th>Status</th>
            <th>MAE (MW)</th>
            <th>Reason</th>
          </tr>
        </thead>
        <tbody>
          {versions.map((v) => (
            <tr key={v.id} style={v.is_current_production ? { fontWeight: "bold" } : {}}>
              <td>{v.model_name}</td>
              <td>{v.model_type}</td>
              <td>{v.status}</td>
              <td>{v.mae ?? "n/a"}</td>
              <td>{v.reason}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
