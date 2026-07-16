"use client";

import { useEffect, useState } from "react";
import MetricChart from "@/components/MetricChart";
import { api } from "@/lib/api";
import type { MetricPoint, RunSummary } from "@/lib/types";

export default function TrainingPage() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [histories, setHistories] = useState<Record<string, MetricPoint[]>>({});
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .runs()
      .then((items) => {
        setRuns(items);
        if (items.length > 0) setSelectedRunId(items[0].run_id);
      })
      .catch((e) => setError(String(e)));
  }, []);

  useEffect(() => {
    if (!selectedRunId) return;
    const run = runs.find((r) => r.run_id === selectedRunId);
    if (!run) return;
    setHistories({});
    const keys = Object.keys(run.metrics);
    Promise.all(
      keys.map((key) =>
        api.metricHistory(selectedRunId, key).then((series) => [key, series.values] as const),
      ),
    )
      .then((entries) => setHistories(Object.fromEntries(entries)))
      .catch((e) => setError(String(e)));
  }, [selectedRunId, runs]);

  const selected = runs.find((r) => r.run_id === selectedRunId) ?? null;

  return (
    <div className="space-y-6">
      <header>
        <h2 className="text-2xl font-semibold">Training</h2>
        <p className="text-sm text-slate-400">MLFlow runs, parameters, and metric curves.</p>
      </header>

      {error && (
        <p className="rounded border border-red-800 bg-red-950 p-3 text-sm text-red-200">
          {error}
        </p>
      )}

      {runs.length === 0 && !error ? (
        <p className="text-sm text-slate-400">
          No runs found. Start the MLFlow server and run a training job first.
        </p>
      ) : (
        <>
          <label className="block text-sm text-slate-300">
            Run
            <select
              className="mt-1 block w-full max-w-xl rounded border border-slate-700 bg-slate-900 px-3 py-2 text-sm"
              value={selectedRunId ?? ""}
              onChange={(e) => setSelectedRunId(e.target.value)}
            >
              {runs.map((run) => (
                <option key={run.run_id} value={run.run_id}>
                  {run.run_id.slice(0, 8)} | {run.status} | {run.start_time ?? "unknown start"}
                </option>
              ))}
            </select>
          </label>

          {selected && (
            <section>
              <h3 className="mb-2 text-lg font-medium">Parameters</h3>
              <table className="w-full max-w-xl text-left text-sm">
                <tbody>
                  {Object.entries(selected.params).map(([key, value]) => (
                    <tr key={key} className="border-b border-slate-800">
                      <td className="py-1 pr-4 text-slate-400">{key}</td>
                      <td className="py-1">{value}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>
          )}

          <section className="grid gap-4 lg:grid-cols-2">
            {Object.entries(histories).map(([key, values]) => (
              <MetricChart key={key} title={key} data={values} />
            ))}
          </section>
        </>
      )}
    </div>
  );
}
