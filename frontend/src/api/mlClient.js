import axios from "axios";

// FastAPI ML service. 127.0.0.1, not 'localhost' -- see client.js for why.
const ML_API_BASE = "http://127.0.0.1:8000";

export const mlClient = axios.create({
  baseURL: ML_API_BASE,
  timeout: 5000,
});

export const getModelInfo = () => mlClient.get("/model").then((res) => res.data);
export const predict = (payload) => mlClient.post("/predict", payload).then((res) => res.data);

export const getMlHealth = () => mlClient.get("/health").then((res) => res.data);
