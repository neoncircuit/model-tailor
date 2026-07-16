"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { MetricPoint } from "@/lib/types";

export default function MetricChart({
  title,
  data,
}: {
  title: string;
  data: MetricPoint[];
}) {
  return (
    <div className="rounded border border-slate-800 p-4">
      <h3 className="mb-2 text-sm font-medium text-slate-200">{title}</h3>
      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid stroke="#1e293b" />
            <XAxis dataKey="step" stroke="#64748b" fontSize={12} />
            <YAxis stroke="#64748b" fontSize={12} />
            <Tooltip
              contentStyle={{ backgroundColor: "#0f172a", border: "1px solid #1e293b" }}
            />
            <Line
              type="monotone"
              dataKey="value"
              stroke="#38bdf8"
              dot={false}
              strokeWidth={2}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
