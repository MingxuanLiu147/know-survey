import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:17860"
    }
  },
  build: {
    sourcemap: true,
    target: "es2022"
  }
});
