"use client";

import { useEffect, useState } from "react";
import SystemCards from "@/components/SystemCards";
import { api } from "@/lib/api";
import type { SystemSnapshot } from "@/lib/types";

const POLL_INTERVAL_MS = 2000;

export default function SystemPage() {
  const [snapshot, setSnapshot] = useState<SystemSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = () => {
      api
        .system()
        .then((s) => {
          if (!cancelled) {
            setSnapshot(s);
            setError(null);
          }
        })
        .catch((e) => {
          if (!cancelled) setError(String(e));
        });
    };
    load();
    const interval = setInterval(load, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  return (
    <div className="space-y-6">
      <header>
        <h2 className="text-2xl font-semibold">System</h2>
        <p className="text-sm text-slate-400">
          Live host metrics, refreshed every {POLL_INTERVAL_MS / 1000} seconds.
        </p>
      </header>
      {error && (
        <p className="rounded border border-red-800 bg-red-950 p-3 text-sm text-red-200">
          {error}
        </p>
      )}
      <SystemCards snapshot={snapshot} />
    </div>
  );
}
