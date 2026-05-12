"""Source-level patches applied before each cross-build.

These edits remove unused OpenGL/X11 includes from files we want to
compile for headless use. Applied idempotently; safe to re-run.
"""
from __future__ import annotations

import pathlib
import sys

PATCHES = [
    # File, line-comment-to-find, replacement (or "" to drop the line).
    (
        "examples/Importers/ImportObjDemo/Wavefront2GLInstanceGraphicsShape.cpp",
        '#include "../../OpenGLWindow/SimpleOpenGL3App.h"',
        '// SimpleOpenGL3App include removed (unused; pulls in GL).\n'
        '#include "Bullet3Common/b3MinMax.h"',
    ),
]


def main(root: pathlib.Path) -> None:
    for rel, find, replace in PATCHES:
        path = root / rel
        if not path.exists():
            print(f"skip (missing): {rel}", file=sys.stderr)
            continue
        text = path.read_text()
        if find not in text:
            # Already patched or upstream changed; ignore quietly.
            continue
        text = text.replace(find, replace)
        path.write_text(text)
        print(f"patched: {rel}")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]).resolve())
