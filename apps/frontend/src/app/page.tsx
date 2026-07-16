"use client";

import { useEffect, useState } from "react";
import SystemCards from "@/components/SystemCards";
import { api } from "@/lib/api";
import type { BenchmarkSummary, RunSummary, SystemSnapshot } from "@/lib/types";

export default function OverviewPage() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [benchmarks, setBenchmarks] = useState<BenchmarkSummary[]>([]);
  const [system, setSystem] = useState<SystemSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.allSettled([api.runs(), api.benchmarks(), api.system()]).then(
      ([runsResult, benchmarksResult, systemResult]) => {
        if (runsResult.status === "fulfilled") setRuns(runsResult.value);
        if (benchmarksResult.status === "fulfilled") setBenchmarks(benchmarksResult.value);
        if (systemResult.status === "fulfilled") setSystem(systemResult.value);
        const failure = [runsResult, benchmarksResult, systemResult].find(
          (r) => r.status === "rejected",
        );
        if (failure && failure.status === "rejected") {
          setError(String(failure.reason));
        }
      },
    );
  }, []);

  const latestBenchmark = benchmarks.length > 0 ? benchmarks[benchmarks.length - 1] : null;

  return (
    <div className="space-y-8">
      <header>
        <h2 className="text-2xl font-semibold">Overview</h2>
        <p className="text-sm text-slate-400">
          Local training, benchmark, and system status for model-tailor.
        </p>
      </header>

      {error && (
        <p className="rounded border border-amber-800 bg-amber-950 p-3 text-sm text-amber-200">
          Some data sources are unavailable: {error}
        </p>
      )}

      <section className="grid gap-4 md:grid-cols-2">
        <div className="rounded border border-slate-800 p-4">
          <h3 className="text-sm font-medium text-slate-300">MLFlow runs</h3>
          <p className="mt-1 text-2xl font-semibold">{runs.length}</p>
          <p className="mt-1 text-xs text-slate-400">
            {runs.length > 0
              ? `Latest run ${runs[0].run_id.slice(0, 8)} (${runs[0].status})`
              : "No runs recorded yet (is the MLFlow server running?)"}
          </p>
        </div>
        <div className="rounded border border-slate-800 p-4">
          <h3 className="text-sm font-medium text-slate-300">Benchmark reports</h3>
          <p className="mt-1 text-2xl font-semibold">{benchmarks.length}</p>
          <p className="mt-1 text-xs text-slate-400">
            {latestBenchmark
              ? `Latest: ${latestBenchmark.filename}`
              : "No benchmark JSON reports found."}
          </p>
        </div>
      </section>

      <section>
        <h3 className="mb-3 text-lg font-medium">Live system snapshot</h3>
        <SystemCards snapshot={system} />
      </section>
    </div>
  );
}
