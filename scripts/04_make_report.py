#!/usr/bin/env python3
"""Generate figures, CSV tables, and the markdown/HTML research report.

Usage:
    python scripts/04_make_report.py
"""
import _bootstrap  # noqa: F401
from lowlight import report as Rep


def main():
    out = Rep.render_report()
    print("Report written:")
    print("  markdown:", out["markdown"])
    print("  html    :", out["html"])
    print("  figures :", len(out["figures"]))
    for f in out["figures"]:
        print("    -", f)


if __name__ == "__main__":
    main()
