import { useEffect, useState } from "react";
import { getModelInfo, predict } from "../api/mlClient";

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
      const payload = { timestamp: timestamp + ":00" };
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
    return (
      <div className="max-w-2xl mx-auto mt-12 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg text-red-700 dark:text-red-300">
        {error}
      </div>
    );
  }

  return (
    <div className="max-w-2xl mx-auto px-4 py-10">
      <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100 mb-2">Try a Forecast</h1>

      {modelInfo && (
        <p className="text-sm text-gray-500 dark:text-gray-400 mb-6">
          Currently serving <span className="font-medium text-gray-700 dark:text-gray-300">{modelInfo.name}</span> ({modelInfo.type}).
          Required fields: <span className="font-mono text-xs bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">{requiredFields.join(", ") || "none"}</span>
        </p>
      )}

      <form onSubmit={handleSubmit} className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6 space-y-4">
        <label className="block">
          <span className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Target hour</span>
          <input
            type="datetime-local"
            value={timestamp}
            onChange={(e) => setTimestamp(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 dark:bg-gray-900 dark:text-gray-100 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </label>

        {requiredFields.map((field) => (
          <label key={field} className="block">
            <span className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              {field} <span className="text-red-500">*</span>
            </span>
            <input
              type="number"
              step="any"
              value={values[field] || ""}
              onChange={(e) => setValues({ ...values, [field]: e.target.value })}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 dark:bg-gray-900 dark:text-gray-100 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </label>
        ))}

        <button
          type="submit"
          disabled={loading || !modelInfo}
          className="w-full py-2.5 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed text-white font-medium rounded-lg transition-colors"
        >
          {loading ? "Predicting..." : "Get Forecast"}
        </button>
      </form>

      {error && (
        <p className="mt-4 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg text-red-700 dark:text-red-300 text-sm">
          {error}
        </p>
      )}

      {result && (
        <div className="mt-6 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-xl p-6 text-center">
          <p className="text-3xl font-bold text-green-700 dark:text-green-300">
            {result.predicted_load_mw.toFixed(2)} MW
          </p>
          <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">at {result.timestamp}</p>
          <p className="text-xs text-gray-500 dark:text-gray-500 mt-2">
            Served by {result.model_name} ({result.model_type})
          </p>
        </div>
      )}
    </div>
  );
}
