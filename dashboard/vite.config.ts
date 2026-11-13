import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run dev` serves the dashboard on :5173 and forwards /api to the Python server (scripts/serve.py, :8000).
// `npm run build` writes dist/, which the Python server serves at / (one port in production).
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": { target: "http://127.0.0.1:8000", changeOrigin: false } } },
  build: { outDir: "dist", sourcemap: false },
});
