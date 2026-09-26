import { useEffect, useState } from "react";
import { getModelInfo, predict } from "../api/mlClient";

const ALL_FIELDS = [
  "hour", "day_of_week", "is_weekend", "month", "quarter", "week_of_year", "is_holiday",
  "irradiance", "temperature", "dewpoint", "specific humidity", "wind speed",
  "lag_24", "lag_48", "lag_168",
  "rolling_mean_24", "rolling_std_24", "rolling_mean_168", "rolling_std_168",
];

export default function TryForecast() {
  const [modelInfo, setModelInfo] = useState(null);
  const [timestamp, setTimestamp] = useState("2022-09-15T14:00");
  const [values, setValues] = useState({});
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    getModelInfo()
      .then(setModelInfo)
      .catch((err) => setError(`Could not reach the ML service: ${err.message}`));
  }, []);

  const requiredFields = modelInfo ? modelInfo.required_features : [];

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setResult(null);
    setLoading(true);
    try {
      const payload = { timestamp: timestamp.replace("T", "T") + ":00" };
      for (const field of requiredFields) {
        if (values[field] === undefined || values[field] === "") {
          setError(`Please fill in required field: ${field}`);
          setLoading(false);
          return;
        }
        payload[field] = parseFloat(values[field]);
      }
      const data = await predict(payload);
      setResult(data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  };

  if (error && !modelInfo) {
    return <div className="dashboard-error">{error}</div>;
  }

  return (
    <div className="try-forecast">
      <h1>Try a Forecast</h1>
      {modelInfo && (
        <p>
          Currently serving: <strong>{modelInfo.name}</strong> ({modelInfo.type}).
          Required fields: {requiredFields.join(", ") || "none"}
        </p>
      )}

      <form onSubmit={handleSubmit}>
        <label>
          Target hour (timestamp):
          <input
            type="datetime-local"
            value={timestamp}
            onChange={(e) => setTimestamp(e.target.value)}
          />
        </label>

        {requiredFields.map((field) => (
          <label key={field}>
            {field} (required):
            <input
              type="number"
              step="any"
              value={values[field] || ""}
              onChange={(e) => setValues({ ...values, [field]: e.target.value })}
            />
          </label>
        ))}

        <button type="submit" disabled={loading || !modelInfo}>
          {loading ? "Predicting..." : "Get Forecast"}
        </button>
      </form>

      {error && <p className="error">{error}</p>}
      {result && (
        <div className="result">
          <h2>Prediction</h2>
          <p>
            <strong>{result.predicted_load_mw.toFixed(2)} MW</strong> at {result.timestamp}
          </p>
          <p>Served by: {result.model_name} ({result.model_type})</p>
        </div>
      )}
    </div>
  );
}
