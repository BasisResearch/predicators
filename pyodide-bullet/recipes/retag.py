"""Retag the pybullet wheel from the new pyemscripten naming to the
older emscripten_X_Y_Z one that Pyodide 0.27.x runtime expects.

The binary is identical — only the platform tag in the wheel filename
and in ``WHEEL`` differs between the two conventions.
"""
from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

NEW_TAG = "pyemscripten_2024_0_wasm32"
OLD_TAG = "emscripten_3_1_58_wasm32"


def retag(src: Path) -> Path:
    if NEW_TAG not in src.name:
        raise SystemExit(f"unexpected wheel name: {src.name}")
    dst = src.with_name(src.name.replace(NEW_TAG, OLD_TAG))

    # Copy + rewrite WHEEL metadata to match.
    with zipfile.ZipFile(src) as zin, \
         zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename.endswith("/WHEEL"):
                text = data.decode()
                text = text.replace(NEW_TAG, OLD_TAG)
                data = text.encode()
            zout.writestr(item, data)
    return dst


if __name__ == "__main__":
    target = Path(sys.argv[1])
    out = retag(target)
    print(out)
