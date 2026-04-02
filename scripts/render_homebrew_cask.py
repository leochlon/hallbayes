from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--version", required=True)
    p.add_argument("--sha256-arm64", required=True)
    p.add_argument("--sha256-x86_64")
    p.add_argument("--identifier", default="com.hassana.berry")
    p.add_argument("--repo", default="hassana-labs/berry")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    version = str(args.version).strip().lstrip("v")
    out = Path(args.out)

    if args.sha256_x86_64:
        rb = f'''cask "berry" do
  version "{version}"
  arch arm: "arm64", intel: "x86_64"

  sha256 arm: "{args.sha256_arm64}", intel: "{args.sha256_x86_64}"

  url "https://github.com/{args.repo}/releases/download/v#{{version}}/berry-#{{version}}-macos-#{{arch}}.pkg"
  name "Berry"
  desc "Berry local MCP runtime + toolpack"
  homepage "https://github.com/{args.repo}"

  pkg "berry-#{{version}}-macos-#{{arch}}.pkg"

  uninstall pkgutil: "{args.identifier}"
end
'''
    else:
        rb = f'''cask "berry" do
  version "{version}"
  sha256 "{args.sha256_arm64}"

  url "https://github.com/{args.repo}/releases/download/v#{{version}}/berry-#{{version}}-macos-arm64.pkg"
  name "Berry"
  desc "Berry local MCP runtime + toolpack"
  homepage "https://github.com/{args.repo}"

  pkg "berry-#{{version}}-macos-arm64.pkg"

  uninstall pkgutil: "{args.identifier}"
end
'''

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rb, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
