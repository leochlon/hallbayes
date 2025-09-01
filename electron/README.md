Electron Wrapper
================

This Electron app wraps the local Streamlit server for the Hallucination Risk UI.

Dev prerequisites
- Node.js 18+
- Python 3.8+ (for the backend)

Run (dev)
```
cd electron
npm install
npm run start
```

What happens
 - Electron spawns `python app/launcher/entry.py --port 8756 --headless` in the bundled backend directory.
- It waits for `http://127.0.0.1:8756`, then opens a window to that URL.
- On exit, it attempts to terminate the backend process tree.
 - First run creates a Python virtualenv under the Electron userData directory and installs requirements.

Build installers (optional)
```
cd electron
npm run build
```

This uses `electron-builder`. Configure signing/notarization as needed for your platform(s).

Notes
- Packaging copies the required Python files into the app `resources/backend` directory.
- The app downloads Python dependencies (streamlit/openai) on first launch; ensure internet access. For offline bundles, prepackage Python and site‑packages or use a frozen backend (PyInstaller) and adjust `main.js` to run it.
