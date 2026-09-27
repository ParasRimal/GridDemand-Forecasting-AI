import { useEffect, useState } from "react";
import { getDjangoRootHealth, getRedisHealth } from "../api/endpoints";
import { getMlHealth } from "../api/mlClient";

function StatusRow({ name, status, detail }) {
  const isUp = status === "up";
  return (
    <div className="flex items-center justify-between py-3 border-b border-gray-100 dark:border-gray-800 last:border-0">
      <div className="flex items-center gap-3">
        <span
          className={`w-3 h-3 rounded-full ${
            isUp ? "bg-green-500" : status === "checking" ? "bg-gray-300 animate-pulse" : "bg-red-500"
          }`}
        />
        <span className="font-medium text-gray-900 dark:text-gray-100">{name}</span>
      </div>
      <div className="text-right">
        <span
          className={`text-sm font-medium ${
            isUp ? "text-green-600 dark:text-green-400" : status === "checking" ? "text-gray-400" : "text-red-600 dark:text-red-400"
          }`}
        >
          {status === "checking" ? "Checking..." : isUp ? "Up" : "Down"}
        </span>
        {detail && <p className="text-xs text-gray-400 dark:text-gray-500 mt-0.5">{detail}</p>}
      </div>
    </div>
  );
}

export default function SystemStatus() {
  const [services, setServices] = useState({
    django: { status: "checking" },
    fastapi: { status: "checking" },
    redis: { status: "checking" },
  });

  const checkAll = () => {
    setServices({
      django: { status: "checking" },
      fastapi: { status: "checking" },
      redis: { status: "checking" },
    });

    getDjangoRootHealth()
      .then(() => setServices((s) => ({ ...s, django: { status: "up" } })))
      .catch((err) => setServices((s) => ({ ...s, django: { status: "down", detail: err.message } })));

    getMlHealth()
      .then((data) =>
        setServices((s) => ({
          ...s,
          fastapi: { status: "up", detail: `Serving ${data.production_model?.name ?? "unknown"}` },
        }))
      )
      .catch((err) => setServices((s) => ({ ...s, fastapi: { status: "down", detail: err.message } })));

    getRedisHealth()
      .then(() => setServices((s) => ({ ...s, redis: { status: "up" } })))
      .catch((err) => setServices((s) => ({ ...s, redis: { status: "down", detail: err.message } })));
  };

  useEffect(() => {
    checkAll();
  }, []);

  return (
    <div className="max-w-2xl mx-auto px-4 py-10">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">System Status</h1>
        <button
          onClick={checkAll}
          className="px-4 py-2 text-sm font-medium bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
        >
          Refresh
        </button>
      </div>

      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <StatusRow name="Django Backend" status={services.django.status} detail={services.django.detail} />
        <StatusRow name="FastAPI ML Service" status={services.fastapi.status} detail={services.fastapi.detail} />
        <StatusRow name="Redis Cache" status={services.redis.status} detail={services.redis.detail} />
      </div>

      <p className="text-xs text-gray-400 dark:text-gray-500 mt-4">
        Note: Celery worker status cannot be checked directly from the browser (no HTTP interface);
        its activity is reflected indirectly via the Monitoring History page.
      </p>
    </div>
  );
}
