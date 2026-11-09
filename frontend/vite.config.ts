import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://localhost:8000", changeOrigin: true, ws: true },
    },
  },
  build: {
    chunkSizeWarningLimit: 4500,
    rollupOptions: {
      output: { manualChunks: { plotly: ["plotly.js-dist-min"], maplibre: ["maplibre-gl"] } },
    },
  },
});
