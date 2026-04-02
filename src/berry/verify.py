from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    message: str


def verify_blob_with_cosign(
    *, artifact: Path, signature: Path, public_key: Optional[Path]
) -> VerifyResult:
    cosign = shutil.which("cosign")
    if cosign is None:
        return VerifyResult(
            ok=False,
            message="cosign not found on PATH (install cosign to verify signatures).",
        )

    cmd: list[str] = [cosign, "verify-blob"]
    if public_key is not None:
        cmd += ["--key", str(public_key)]
    cmd += ["--signature", str(signature), str(artifact)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return VerifyResult(ok=True, message="signature verified")
    except subprocess.CalledProcessError as exc:
        msg = (
            exc.stderr or exc.stdout or ""
        ).strip() or f"cosign verify failed (exit={exc.returncode})"
        return VerifyResult(ok=False, message=msg)
