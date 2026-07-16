/** TypeScript interfaces mirroring the backend Pydantic schemas. */

export interface HealthResponse {
  status: string;
}

export interface MlflowConfigResponse {
  tracking_uri: string;
  experiment_name: string;
}

export interface RunSummary {
  run_id: string;
  status: string;
  start_time: string | null;
  params: Record<string, string>;
  metrics: Record<string, number>;
}

export interface MetricPoint {
  step: number;
  value: number;
}

export interface MetricSeries {
  run_id: string;
  metric: string;
  values: MetricPoint[];
}

export interface BenchmarkSummary {
  filename: string;
  model: string | null;
  num_examples: number | null;
  elapsed_seconds: number | null;
  overall_metrics: Record<string, unknown>;
}

export interface BenchmarkReport {
  filename: string;
  report: Record<string, unknown>;
}

export interface GpuDevice {
  index: number;
  name: string;
  utilisation_pct: number | null;
  memory_used_gb: number;
  memory_total_gb: number;
  memory_percent: number;
}

export interface StorageDrive {
  mountpoint: string;
  device: string;
  fstype: string;
  total_gb: number;
  used_gb: number;
  free_gb: number;
  percent: number;
  builtin: boolean;
  bus_type: string | null;
}

export interface SystemSnapshot {
  timestamp: number;
  cpu_percent: number;
  ram: {
    used_gb: number;
    total_gb: number;
    percent: number;
  };
  gpu: {
    available: boolean;
    devices: GpuDevice[];
  };
  storage?: StorageDrive[];
}

export interface ConfigFiles {
  base: Record<string, unknown>;
  task: Record<string, unknown>;
}
