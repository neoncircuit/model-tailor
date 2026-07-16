/**
 * Development launcher with automatic port delegation.
 *
 * Finds the first free port starting at 3000, discovers the backend port
 * (waiting briefly for ../backend-py/.dev-port when the backend is still
 * starting), and runs `next dev`.
 */
import { spawn } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { createServer } from "node:net";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { setTimeout as delay } from "node:timers/promises";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const frontendRoot = join(scriptDir, "..");
const BACKEND_PORT_FILE = join(frontendRoot, "..", "backend-py", ".dev-port");
const BACKEND_WAIT_MS = 10000;

function isPortFree(port) {
  return new Promise((resolve) => {
    const server = createServer();
    server.once("error", () => resolve(false));
    server.once("listening", () => server.close(() => resolve(true)));
    server.listen(port, "127.0.0.1");
  });
}

async function findFreePort(start) {
  for (let port = start; port < 65535; port += 1) {
    if (await isPortFree(port)) {
      return port;
    }
  }
  throw new Error(`No free port found starting from ${start}`);
}

async function resolveBackendUrl() {
  if (process.env.DASHBOARD_BACKEND_URL) {
    return process.env.DASHBOARD_BACKEND_URL;
  }
  const deadline = Date.now() + BACKEND_WAIT_MS;
  do {
    if (existsSync(BACKEND_PORT_FILE)) {
      const port = readFileSync(BACKEND_PORT_FILE, "utf8").trim();
      if (port) {
        return `http://localhost:${port}`;
      }
    }
    await delay(250);
  } while (Date.now() < deadline);
  return "http://localhost:8000";
}

const port = await findFreePort(3000);
const backend = await resolveBackendUrl();

console.log(`Dashboard frontend starting at http://localhost:${port}`);
console.log(`Proxying /api/* to ${backend}`);

const child = spawn("npx", ["next", "dev", "-p", String(port)], {
  cwd: frontendRoot,
  stdio: "inherit",
  env: { ...process.env, DASHBOARD_BACKEND_URL: backend },
  shell: false,
});

child.on("exit", (code) => process.exit(code ?? 0));
