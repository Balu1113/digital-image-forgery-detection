import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// In development the API is served by Django on :8000. The proxy lets the
// frontend call relative URLs (/api, /media, /admin) with no CORS friction.
// For production, set VITE_API_URL to the public Django base URL instead.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://127.0.0.1:8000',
      '/media': 'http://127.0.0.1:8000',
      '/admin': 'http://127.0.0.1:8000',
    },
  },
});