"use client";

import type { SystemSnapshot } from "@/lib/types";

function UsageBar({ percent }: { percent: number }) {
  return (
    <div className="mt-2 h-2 w-full rounded bg-slate-800">
      <div
        className="h-2 rounded bg-sky-500 transition-all"
        style={{ width: `${Math.min(100, Math.max(0, percent))}%` }}
      />
    </div>
  );
}

function DriveBadge({ builtin, busType }: { builtin: boolean; busType: string | null }) {
  if (builtin) {
    return (
      <span className="inline-flex items-center rounded bg-sky-950 px-2 py-0.5 text-xs font-medium text-sky-300">
        Built-in
      </span>
    );
  }
  return (
    <span className="inline-flex items-center rounded bg-amber-950 px-2 py-0.5 text-xs font-medium text-amber-300">
      External{busType ? ` (${busType})` : ""}
    </span>
  );
}

export default function SystemCards({ snapshot }: { snapshot: SystemSnapshot | null }) {
  if (!snapshot) {
    return <p className="text-sm text-slate-400">Loading system snapshot...</p>;
  }

  const storage = snapshot.storage ?? [];

  return (
    <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
      <div className="rounded border border-slate-800 p-4">
        <h3 className="text-sm font-medium text-slate-300">CPU</h3>
        <p className="mt-1 text-2xl font-semibold">{snapshot.cpu_percent.toFixed(1)}%</p>
        <UsageBar percent={snapshot.cpu_percent} />
      </div>

      <div className="rounded border border-slate-800 p-4">
        <h3 className="text-sm font-medium text-slate-300">RAM</h3>
        <p className="mt-1 text-2xl font-semibold">
          {snapshot.ram.used_gb.toFixed(1)} / {snapshot.ram.total_gb.toFixed(1)} GB
        </p>
        <UsageBar percent={snapshot.ram.percent} />
        <p className="mt-1 text-xs text-slate-400">{snapshot.ram.percent.toFixed(1)}% used</p>
      </div>

      {snapshot.gpu.available ? (
        snapshot.gpu.devices.map((device) => (
          <div key={device.index} className="rounded border border-slate-800 p-4">
            <h3 className="text-sm font-medium text-slate-300">
              GPU {device.index}: {device.name}
            </h3>
            <p className="mt-1 text-2xl font-semibold">
              {device.memory_used_gb.toFixed(1)} / {device.memory_total_gb.toFixed(1)} GB
            </p>
            <UsageBar percent={device.memory_percent} />
            <p className="mt-1 text-xs text-slate-400">
              {device.memory_percent.toFixed(1)}% VRAM used
              {device.utilisation_pct !== null &&
                ` | ${device.utilisation_pct.toFixed(0)}% utilisation`}
            </p>
          </div>
        ))
      ) : (
        <div className="rounded border border-slate-800 p-4">
          <h3 className="text-sm font-medium text-slate-300">GPU</h3>
          <p className="mt-1 text-sm text-slate-400">No GPU detected</p>
        </div>
      )}

      <div className="rounded border border-slate-800 p-4 md:col-span-2 xl:col-span-3">
        <h3 className="text-sm font-medium text-slate-300">Storage</h3>
        {storage.length === 0 ? (
          <p className="mt-2 text-sm text-slate-400">No storage drives detected.</p>
        ) : (
          <div className="mt-2 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {storage.map((drive) => (
              <div
                key={drive.mountpoint}
                className="rounded bg-slate-900/50 p-3"
              >
                <div className="flex items-center justify-between gap-2">
                  <p className="truncate text-sm font-medium text-slate-200" title={drive.mountpoint}>
                    {drive.mountpoint}
                  </p>
                  <DriveBadge builtin={drive.builtin} busType={drive.bus_type} />
                </div>
                <p className="mt-1 text-xs text-slate-500">{drive.fstype}</p>
                <p className="mt-1 text-sm text-slate-300">
                  {drive.used_gb.toFixed(1)} / {drive.total_gb.toFixed(1)} GB
                </p>
                <UsageBar percent={drive.percent} />
                <p className="mt-1 text-xs text-slate-400">
                  {drive.percent.toFixed(1)}% used · {drive.free_gb.toFixed(1)} GB free
                </p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
