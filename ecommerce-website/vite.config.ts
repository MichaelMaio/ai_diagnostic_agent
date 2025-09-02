import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  base: '/VendingMachine/', // this fixes path resolution on GitHub Pages
  plugins: [react()],
});
