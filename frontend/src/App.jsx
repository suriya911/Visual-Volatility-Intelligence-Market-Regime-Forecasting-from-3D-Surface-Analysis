import { useEffect, useMemo, useState } from "react";
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { fetchMetrics, fetchPredictions, fetchSurfaces, runPipeline, surfaceUrl } from "./api";

function Card({ title, value }) {
  return (
    <div className="card">
      <span>{title}</span>
      <strong>{value}</strong>
    </div>
  );
}

export default function App() {
  const [metrics, setMetrics] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [surfaces, setSurfaces] = useState([]);
  const [selectedSurface, setSelectedSurface] = useState("");
  const [loading, setLoading] = useState(false);

  async function loadAll() {
    const [m, p, s] = await Promise.all([fetchMetrics(), fetchPredictions(), fetchSurfaces()]);
    setMetrics(m);
    setPredictions(p);
    setSurfaces(s);
    if (s.length) setSelectedSurface((prev) => prev || s[s.length - 1]);
  }

  useEffect(() => {
    loadAll().catch(() => {});
  }, []);

  const chartData = useMemo(
    () =>
      predictions.map((row) => ({
        date: row.date?.slice(0, 10),
        meanIV: Number(row.mean_iv),
        regime: Number(row.pred_regime)
      })),
    [predictions]
  );

  async function onRunPipeline() {
    setLoading(true);
    try {
      await runPipeline();
      await loadAll();
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <section className="hero">
        <h1>Visual Volatility Intelligence</h1>
        <p>Regime forecasting from implied volatility surfaces</p>
        <button onClick={onRunPipeline} disabled={loading}>{loading ? "Running..." : "Run Pipeline"}</button>
      </section>

      <section className="metrics">
        <Card title="CNN Accuracy" value={metrics ? `${(metrics.cnn_accuracy * 100).toFixed(1)}%` : "-"} />
        <Card title="Baseline Accuracy" value={metrics ? `${(metrics.baseline_accuracy * 100).toFixed(1)}%` : "-"} />
        <Card title="Strategy Sharpe" value={metrics ? Number(metrics.strategy_sharpe).toFixed(2) : "-"} />
        <Card title="False Signal Rate" value={metrics ? `${(metrics.false_signal_rate * 100).toFixed(1)}%` : "-"} />
      </section>

      <section className="panel two-col">
        <div className="chart-box">
          <h2>Historical Mean IV and Predicted Regime</h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData}>
              <XAxis dataKey="date" hide />
              <YAxis yAxisId="iv" />
              <YAxis yAxisId="regime" orientation="right" domain={[0, 3]} />
              <Tooltip />
              <Line yAxisId="iv" type="monotone" dataKey="meanIV" stroke="#f95f62" dot={false} strokeWidth={2} />
              <Line yAxisId="regime" type="stepAfter" dataKey="regime" stroke="#0f4d5c" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="surface-box">
          <h2>Latest Surface Image</h2>
          {selectedSurface ? <img src={surfaceUrl(selectedSurface)} alt="volatility surface" /> : <p>No surfaces yet.</p>}
          <select value={selectedSurface} onChange={(e) => setSelectedSurface(e.target.value)}>
            {surfaces.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </div>
      </section>
    </main>
  );
}
