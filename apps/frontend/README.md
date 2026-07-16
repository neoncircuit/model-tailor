# model-tailor dashboard frontend

Next.js 15 + TypeScript + Tailwind CSS + Recharts frontend for the local
model-tailor configuration and performance dashboard.

## Development

Install dependencies:

```bash
npm install
```

Start the dev server (delegates to the next free port if 3000 is taken):

```bash
npm run dev
```

The dev launcher proxies `/api/*` to the FastAPI backend. The backend URL is
resolved in this order:

1. `DASHBOARD_BACKEND_URL` environment variable.
2. `../backend-py/.dev-port` (written by the backend dev launcher).
3. `http://localhost:8000` (fallback).

## Build

```bash
npm run build
```
