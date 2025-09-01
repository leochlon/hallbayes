const { app, BrowserWindow, dialog, shell } = require('electron');
const path = require('path');
const { spawn, execFileSync } = require('child_process');
const http = require('http');
const fs = require('fs');

const SERVER_PORT = process.env.APP_PORT ? Number(process.env.APP_PORT) : 8756;
const SERVER_URL = `http://127.0.0.1:${SERVER_PORT}`;

// App root (repo root, one level up from electron/)
const ROOT_DIR = path.resolve(__dirname, '..');
const isPackaged = app ? app.isPackaged : false;
// When packaged via electron-builder, extraResources are placed under process.resourcesPath
function getBaseDir() {
  return isPackaged ? path.join(process.resourcesPath, 'backend') : ROOT_DIR;
}

function pickPythonCommand() {
  if (process.platform === 'win32') return 'python';
  try { execFileSync('python3', ['--version']); return 'python3'; } catch {}
  return 'python';
}

function getOfflineBackendPath() {
  const baseDir = getBaseDir();
  const binDir = path.join(baseDir, 'bin');
  const exeName = process.platform === 'win32' ? 'hallucination-backend.exe' : 'hallucination-backend';
  const full = path.join(binDir, exeName);
  return fs.existsSync(full) ? full : null;
}

function startBackend() {
  const offline = getOfflineBackendPath();
  if (offline) {
    const child = spawn(offline, ['--port', String(SERVER_PORT), '--headless'], {
      cwd: path.dirname(offline),
      env: { ...process.env, STREAMLIT_SERVER_HEADLESS: 'true' },
      detached: process.platform !== 'win32',
      stdio: 'inherit',
    });
    return child;
  }
  const py = pickPythonCommand();
  const baseDir = getBaseDir();
  const launchPath = path.join(baseDir, 'app', 'launcher', 'entry.py');
  const args = [launchPath, '--port', String(SERVER_PORT), '--headless'];
  const child = spawn(py, args, {
    cwd: baseDir,
    env: {
      ...process.env,
      STREAMLIT_SERVER_HEADLESS: 'true',
      LAUNCH_VENV_DIR: path.join(app.getPath('userData'), 'python-venv')
    },
    detached: process.platform !== 'win32',
    stdio: 'inherit',
  });
  return child;
}

function killProcessTree(child) {
  if (!child || child.killed) return;
  try {
    if (process.platform === 'win32') {
      spawn('taskkill', ['/PID', String(child.pid), '/T', '/F']);
    } else {
      try { process.kill(-child.pid, 'SIGTERM'); } catch (e) { try { process.kill(child.pid, 'SIGTERM'); } catch {} }
    }
  } catch (e) {
    // ignore
  }
}

function waitForServer(url, timeoutMs = 60000, intervalMs = 500) {
  const start = Date.now();
  return new Promise((resolve, reject) => {
    const check = () => {
      const req = http.get(url, (res) => {
        res.resume();
        resolve(true);
      });
      req.on('error', () => {
        if (Date.now() - start > timeoutMs) return reject(new Error('Timeout waiting for server'));
        setTimeout(check, intervalMs);
      });
    };
    check();
  });
}

async function createWindow() {
  const win = new BrowserWindow({
    width: 1100,
    height: 760,
    webPreferences: {
      sandbox: true,
    }
  });
  await win.loadURL(SERVER_URL);
  win.webContents.setWindowOpenHandler(({ url }) => { shell.openExternal(url); return { action: 'deny' }; });
}

let backendChild = null;

app.whenReady().then(async () => {
  try {
    backendChild = startBackend();
    await waitForServer(SERVER_URL, 120000);
    await createWindow();
  } catch (e) {
    dialog.showErrorBox('Startup error', String(e && e.message || e));
  }

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('before-quit', () => {
  killProcessTree(backendChild);
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});
