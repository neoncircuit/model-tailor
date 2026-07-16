"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { ConfigFiles } from "@/lib/types";

function ConfigPanel({ title, data }: { title: string; data: Record<string, unknown> }) {
  return (
    <section className="rounded border border-slate-800 p-4">
      <h3 className="mb-2 text-sm font-medium text-slate-200">{title}</h3>
      <pre className="overflow-x-auto text-xs text-slate-300">
        {JSON.stringify(data, null, 2)}
      </pre>
    </section>
  );
}

export default function ConfigsPage() {
  const [configs, setConfigs] = useState<ConfigFiles | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .configs()
      .then(setConfigs)
      .catch((e) => setError(String(e)));
  }, []);

  return (
    <div className="space-y-6">
      <header>
        <h2 className="text-2xl font-semibold">Configs</h2>
        <p className="text-sm text-slate-400">
          Active YAML configuration used by the pipeline.
        </p>
      </header>

      {error && (
        <p className="rounded border border-red-800 bg-red-950 p-3 text-sm text-red-200">
          {error}
        </p>
      )}

      {configs ? (
        <div className="grid gap-4 lg:grid-cols-2">
          <ConfigPanel title="config/base.yaml" data={configs.base} />
          <ConfigPanel title="config/tasks/sql_generation.yaml" data={configs.task} />
        </div>
      ) : (
        !error && <p className="text-sm text-slate-400">Loading configs...</p>
      )}
    </div>
  );
}
