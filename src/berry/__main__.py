from __future__ import annotations

import sys

# Handle PyInstaller single-file mode
if getattr(sys, "frozen", False):
    # Running as PyInstaller bundle
    from berry.cli import main
else:
    # Running as normal Python
    from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
