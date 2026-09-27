import { apiClient, djangoRootClient } from "./client";

export const getForecasts = () => apiClient.get("/forecasts/").then((res) => res.data);
export const getModelVersions = () => apiClient.get("/model-versions/").then((res) => res.data);
export const getMonitoring = () => apiClient.get("/monitoring/").then((res) => res.data);

export const getDjangoRootHealth = () => djangoRootClient.get("/").then((res) => res.data);
export const getRedisHealth = () => djangoRootClient.get("/health/redis/").then((res) => res.data);
