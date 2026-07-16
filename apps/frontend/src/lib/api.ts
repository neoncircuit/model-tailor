/** Typed fetch helpers for the dashboard backend (proxied via /api). */

import type {
  BenchmarkReport,
  BenchmarkSummary,
  ConfigFiles,
  HealthResponse,
  MetricSeries,
  MlflowConfigResponse,
  RunSummary,
  SystemSnapshot,
} from "./types";

const API_BASE = "/api";

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`API request failed: ${response.status} ${response.statusText}`);
  }
  return (await response.json()) as T;
}

export const api = {
  health: () => getJson<HealthResponse>("/health"),
  config: () => getJson<MlflowConfigResponse>("/config"),
  runs: () => getJson<RunSummary[]>("/runs"),
  /** metricKey may contain slashes (e.g. "train/loss"); the backend uses a path converter. */
  metricHistory: (runId: string, metricKey: string) =>
    getJson<MetricSeries>(`/runs/${runId}/metrics/${metricKey}`),
  benchmarks: () => getJson<BenchmarkSummary[]>("/benchmarks"),
  benchmark: (filename: string) =>
    getJson<BenchmarkReport>(`/benchmarks/${encodeURIComponent(filename)}`),
  system: () => getJson<SystemSnapshot>("/system"),
  configs: () => getJson<ConfigFiles>("/configs"),
};
