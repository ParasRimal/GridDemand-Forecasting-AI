import axios from "axios";

// Using 127.0.0.1 explicitly, not 'localhost' -- this environment has shown
// 'localhost' can fail to resolve reliably even when 127.0.0.1 works fine
// (see ml_service/app/config.py's REDIS_HOST comment for the same lesson).
const DJANGO_API_BASE = "http://127.0.0.1:8001/api";

export const apiClient = axios.create({
  baseURL: DJANGO_API_BASE,
  timeout: 5000,
});

// Separate client for non-/api/ endpoints (root status, health checks).
export const djangoRootClient = axios.create({
  baseURL: "http://127.0.0.1:8001",
  timeout: 5000,
});
