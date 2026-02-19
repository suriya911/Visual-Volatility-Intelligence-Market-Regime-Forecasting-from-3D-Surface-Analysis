const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function runPipeline() {
  const res = await fetch(`${API}/pipeline/run`, { method: "POST" });
  if (!res.ok) throw new Error("Pipeline run failed");
  return res.json();
}

export async function fetchMetrics() {
  const res = await fetch(`${API}/metrics`);
  if (!res.ok) throw new Error("Metrics unavailable");
  return res.json();
}

export async function fetchPredictions() {
  const res = await fetch(`${API}/predictions?limit=120`);
  if (!res.ok) throw new Error("Predictions unavailable");
  return res.json();
}

export async function fetchSurfaces() {
  const res = await fetch(`${API}/surfaces?limit=30`);
  if (!res.ok) throw new Error("Surfaces unavailable");
  return res.json();
}

export function surfaceUrl(name) {
  return `${API}/surfaces/${name.replace(".png", "")}`;
}
