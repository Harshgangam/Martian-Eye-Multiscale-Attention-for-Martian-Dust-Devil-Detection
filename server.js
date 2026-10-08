const { spawn } = require('child_process');
const path = require('path');
const net = require('net');
const http = require('http');
const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');

function findAvailablePort(startPort) {
    return new Promise((resolve) => {
        const server = net.createServer();
        server.listen(startPort, '127.0.0.1', () => {
            const { port } = server.address();
            server.close(() => resolve(port));
        });
        server.on('error', () => {
            resolve(findAvailablePort(startPort + 1));
        });
    });
}

function waitForServer(port, timeoutMs = 15000) {
    const startTime = Date.now();
    return new Promise((resolve, reject) => {
        const check = () => {
            const req = http.get(`http://127.0.0.1:${port}/api/health`, (res) => {
                resolve();
            });
            req.on('error', () => {
                if (Date.now() - startTime > timeoutMs) {
                    reject(new Error(`Timeout waiting for backend port ${port}`));
                } else {
                    setTimeout(check, 300);
                }
            });
            req.end();
        };
        check();
    });
}

function listenWithFallback(app, startPort) {
    return new Promise((resolve) => {
        const server = app.listen(startPort, async () => {
            console.log(`\n======================================================`);
            console.log(`✨ Martian Eye Mission Control UI is Live!`);
            console.log(`🌐 Open browser at: http://localhost:${startPort}`);
            console.log(`======================================================\n`);

            try {
                const open = (await import('open')).default;
                await open(`http://localhost:${startPort}`);
            } catch (err) {
                // Fallback if open module is unavailable
            }
            resolve({ server, port: startPort });
        });

        server.on('error', (err) => {
            if (err.code === 'EADDRINUSE') {
                console.log(`Port ${startPort} in use, automatically trying port ${startPort + 1}...`);
                resolve(listenWithFallback(app, startPort + 1));
            } else {
                console.error('Server listen error:', err);
            }
        });
    });
}

async function startApp() {
    console.log('🚀 Martian Eye Mission Control Launcher starting...');

    const PYTHON_PORT = await findAvailablePort(8000);
    const START_PORT = 3000;

    const pythonExecutable = process.platform === 'win32' ? 'python' : 'python3';
    console.log(`📡 Spawning Python AI Backend on port ${PYTHON_PORT}...`);

    const pyProcess = spawn(pythonExecutable, [
        '-m', 'uvicorn', 'backend.app:app',
        '--host', '127.0.0.1',
        '--port', String(PYTHON_PORT)
    ], {
        cwd: __dirname,
        env: { ...process.env, PYTHONPATH: __dirname }
    });

    pyProcess.stdout.on('data', (data) => {
        console.log(`[Python AI]: ${data.toString().trim()}`);
    });

    pyProcess.stderr.on('data', (data) => {
        const msg = data.toString().trim();
        if (msg) console.log(`[Python AI Log]: ${msg}`);
    });

    pyProcess.on('close', (code) => {
        console.log(`[Python AI] Process exited with code ${code}`);
    });

    process.stdout.write('⏳ Waiting for Python AI backend to initialize...');
    await waitForServer(PYTHON_PORT);
    console.log(' Ready!');

    const app = express();

    app.use(createProxyMiddleware({
        target: `http://127.0.0.1:${PYTHON_PORT}`,
        changeOrigin: true,
        pathFilter: '/api'
    }));

    app.use(express.static(path.join(__dirname, 'frontend')));

    app.get('*', (req, res) => {
        res.sendFile(path.join(__dirname, 'frontend', 'index.html'));
    });

    await listenWithFallback(app, START_PORT);

    process.on('SIGINT', () => {
        console.log('\nStopping MARS SIGHT services...');
        pyProcess.kill();
        process.exit();
    });
}

startApp().catch(console.error);
