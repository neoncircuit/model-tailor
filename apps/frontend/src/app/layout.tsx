import type { Metadata } from "next";
import Link from "next/link";
import type { ReactNode } from "react";
import "./globals.css";

export const metadata: Metadata = {
  title: "model-tailor dashboard",
  description: "Local configuration and performance dashboard for model-tailor.",
};

const NAV_ITEMS = [
  { href: "/", label: "Overview" },
  { href: "/training", label: "Training" },
  { href: "/benchmarks", label: "Benchmarks" },
  { href: "/system", label: "System" },
  { href: "/configs", label: "Configs" },
];

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="flex min-h-screen">
          <aside className="w-56 shrink-0 border-r border-slate-800 p-6">
            <h1 className="text-lg font-semibold">model-tailor</h1>
            <p className="text-xs text-slate-400">dashboard</p>
            <nav className="mt-8 flex flex-col gap-1">
              {NAV_ITEMS.map((item) => (
                <Link
                  key={item.href}
                  href={item.href}
                  className="rounded px-3 py-2 text-sm text-slate-300 hover:bg-slate-800 hover:text-white"
                >
                  {item.label}
                </Link>
              ))}
            </nav>
          </aside>
          <main className="flex-1 overflow-x-hidden p-8">{children}</main>
        </div>
      </body>
    </html>
  );
}
