import {defineConfig} from 'vite';
import path from 'path';
import mkcert from 'vite-plugin-mkcert';

export default defineConfig({
    base: './',
    resolve: {
        alias: {
            'webGPU-framework': path.resolve(__dirname, './src/lib/'),
        },
    },
    build: {
        outDir: 'dist',
        emptyOutDir: true,
        minify: true,
        sourcemap: false,
    },
    server: {
        https: true,
    },
    plugins: [mkcert()],
});
