import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/health':       'http://localhost:8000',
      '/defaults':     'http://localhost:8000',
      '/simulate':     'http://localhost:8000',
      '/monte_carlo':  'http://localhost:8000',
    },
  },
})