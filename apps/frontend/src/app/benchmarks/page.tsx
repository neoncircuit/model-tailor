"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "@/lib/api";
import type { BenchmarkReport, BenchmarkSummary } from "@/lib/types";

function numericEntries(metrics: Record<string, unknown>): { name: string; value: number }[] {
  return Object.entries(metrics)
    .filter(([, value]) => typeof value === "number" && Number.isFinite(value))
    .map(([name, value]) => ({ name, value: value as number }));
}

function MetricTable({
  title,
  data,
}: {
  title: string;
  data: Record<string, unknown> | undefined;
}) {
  if (!data || Object.keys(data).length === 0) return null;
  return (
    <section>
      <h3 className="mb-2 text-lg font-medium">{title}</h3>
      <table className="w-full max-w-2xl text-left text-sm">
        <tbody>
          {Object.entries(data).map(([key, value]) => (
            <tr key={key} className="border-b border-slate-800">
              <td className="py-1 pr-4 text-slate-400">{key}</td>
              <td className="py-1">
                {typeof value === "number" ? value.toFixed(4) : JSON.stringify(value)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

export default function BenchmarksPage() {
  const [summaries, setSummaries] = useState<BenchmarkSummary[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [report, setReport] = useState<BenchmarkReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .benchmarks()
      .then((items) => {
        setSummaries(items);
        if (items.length > 0) setSelected(items[items.length - 1].filename);
      })
      .catch((e) => setError(String(e)));
  }, []);

  useEffect(() => {
    if (!selected) return;
    api
      .benchmark(selected)
      .then(setReport)
      .catch((e) => setError(String(e)));
  }, [selected]);

  const overallMetrics = useMemo(() => {
    const metrics = (report?.report?.overall_metrics ?? {}) as Record<string, unknown>;
    return numericEntries(metrics);
  }, [report]);

  const gateMetrics = report?.report?.gate_metrics as Record<string, unknown> | undefined;
  const repairMetrics = report?.report?.repair_metrics as Record<string, unknown> | undefined;

  return (
    <div className="space-y-6">
      <header>
        <h2 className="text-2xl font-semibold">Benchmarks</h2>
        <p className="text-sm text-slate-400">
          Gate benchmark JSON reports from tasks/sql_generation/results/.
        </p>
      </header>

      {error && (
        <p className="rounded border border-red-800 bg-red-950 p-3 text-sm text-red-200">
          {error}
        </p>
      )}

      {summaries.length === 0 && !error ? (
        <p className="text-sm text-slate-400">
          No benchmark reports found. Run scripts/run_gate_benchmark.py to generate one.
        </p>
      ) : (
        <>
          <label className="block text-sm text-slate-300">
            Report
            <select
              className="mt-1 block w-full max-w-xl rounded border border-slate-700 bg-slate-900 px-3 py-2 text-sm"
              value={selected ?? ""}
              onChange={(e) => setSelected(e.target.value)}
            >
              {summaries.map((item) => (
                <option key={item.filename} value={item.filename}>
                  {item.filename}
                </option>
              ))}
            </select>
          </label>

          {report && (
            <>
              <section className="grid gap-4 md:grid-cols-3">
                <div className="rounded border border-slate-800 p-4">
                  <h3 className="text-sm font-medium text-slate-300">Model</h3>
                  <p className="mt-1 text-lg font-semibold">
                    {String(report.report.model ?? "unknown")}
                  </p>
                </div>
                <div className="rounded border border-slate-800 p-4">
                  <h3 className="text-sm font-medium text-slate-300">Examples</h3>
                  <p className="mt-1 text-lg font-semibold">
                    {String(report.report.num_examples ?? "-")}
                  </p>
                </div>
                <div className="rounded border border-slate-800 p-4">
                  <h3 className="text-sm font-medium text-slate-300">Elapsed</h3>
                  <p className="mt-1 text-lg font-semibold">
                    {typeof report.report.elapsed_seconds === "number"
                      ? `${report.report.elapsed_seconds.toFixed(1)}s`
                      : "-"}
                  </p>
                </div>
              </section>

              {overallMetrics.length > 0 && (
                <section>
                  <h3 className="mb-2 text-lg font-medium">Overall metrics</h3>
                  <div className="h-72 rounded border border-slate-800 p-4">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={overallMetrics} layout="vertical">
                        <CartesianGrid stroke="#1e293b" />
                        <XAxis type="number" stroke="#64748b" fontSize={12} />
                        <YAxis
                          type="category"
                          dataKey="name"
                          stroke="#64748b"
                          fontSize={11}
                          width={220}
                        />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: "#0f172a",
                            border: "1px solid #1e293b",
                          }}
                        />
                        <Bar dataKey="value" fill="#38bdf8" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </section>
              )}

              <MetricTable title="Gate metrics" data={gateMetrics} />
              <MetricTable title="Repair metrics" data={repairMetrics} />
            </>
          )}
        </>
      )}
    </div>
  );
}
