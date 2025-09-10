# Electron Desktop Wrapper

This Electron app wraps the local Streamlit server for the Hallucination Risk UI.

## Dev

```bash
cd electron
npm install
npm run start
```

By default it launches the backend on `http://127.0.0.1:8756` and opens a window.

## Build DMG (macOS)

```bash
cd electron
npm install
export CSC_IDENTITY_AUTO_DISCOVERY=false  # if you don't have signing set up
npm run build
```

Artifacts are created under `../release`.

## Offline backend (optional)

Build the single-file backend to avoid pip installs at first run:

```bash
bash scripts/build_offline_backend.sh
```

Electron will use `bin/hallucination-backend` automatically if present; otherwise
it will create a venv and run Streamlit via `app/launcher/entry.py`.

