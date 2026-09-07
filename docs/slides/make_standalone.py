"""Build a fully standalone single-file version of the 2026-08-20 deck.

Produces ``weekly_sync_20260820_slides_standalone.html`` next to the
source deck, with everything inlined so the file works offline and can
be shared as a single attachment:

  - reveal.js CSS/JS + plugins (fetched from the pinned CDN version),
  - the white theme's Source Sans Pro font files (data URIs),
  - every ``assets/`` image and video the deck references (data URIs).

Needs network access at build time (for the reveal.js files only).
Re-run after editing the source deck.

Usage:
    python docs/slides/make_standalone.py
"""
import base64
import mimetypes
import re
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "weekly_sync_20260820_slides.html"
OUT = HERE / "weekly_sync_20260820_slides_standalone.html"
CDN_BASE = "https://cdn.jsdelivr.net/npm/reveal.js@4.6.1"

mimetypes.add_type("font/woff2", ".woff2")
mimetypes.add_type("font/woff", ".woff")
mimetypes.add_type("application/vnd.ms-fontobject", ".eot")


def data_uri(content: bytes, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(content).decode("ascii")


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def font_css_inlined() -> str:
    """source-sans-pro.css with every url(...) embedded as a data URI."""
    base = f"{CDN_BASE}/dist/theme/fonts/source-sans-pro"
    css = fetch(f"{base}/source-sans-pro.css").decode("utf-8")

    def repl(m: re.Match) -> str:
        ref = m.group(1).strip("'\"")
        path = ref.split("#")[0].split("?")[0].removeprefix("./")
        mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
        blob = fetch(f"{base}/{path}")
        print(f"  embedded font {path} ({len(blob)//1024} KB)")
        return f"url({data_uri(blob, mime)})"

    return re.sub(r"url\(([^)]+)\)", repl, css)


def main() -> int:
    html = SRC.read_text(encoding="utf-8")

    for path in ("dist/reveal.css", "dist/theme/white.css",
                 "plugin/highlight/monokai.css"):
        tag = f'<link rel="stylesheet" href="{CDN_BASE}/{path}">'
        assert tag in html, tag
        css = fetch(f"{CDN_BASE}/{path}").decode("utf-8")
        if path == "dist/theme/white.css":
            imp = "@import url(./fonts/source-sans-pro/source-sans-pro.css);"
            assert imp in css
            print("inlining Source Sans Pro fonts...")
            css = css.replace(imp, font_css_inlined())
        html = html.replace(tag, f"<style>\n{css}\n</style>")

    for path in ("dist/reveal.js", "plugin/markdown/markdown.js",
                 "plugin/highlight/highlight.js", "plugin/notes/notes.js"):
        tag = f'<script src="{CDN_BASE}/{path}"></script>'
        assert tag in html, tag
        js = fetch(f"{CDN_BASE}/{path}").decode("utf-8")
        # Inline JS must not close the surrounding tag early.
        assert "</script" not in js, f"{path} contains </script"
        html = html.replace(tag, f"<script>\n{js}\n</script>")

    for ref in sorted(set(re.findall(r'src="(assets/[^"]+)"', html))):
        f = HERE / ref
        assert f.is_file(), f
        mime = mimetypes.guess_type(f.name)[0]
        assert mime, f
        blob = f.read_bytes()
        html = html.replace(f'src="{ref}"', f'src="{data_uri(blob, mime)}"')
        print(f"embedded {ref} ({len(blob)//1024} KB)")

    assert "cdn.jsdelivr.net" not in html, "CDN reference survived"
    assert 'src="assets/' not in html, "asset reference survived"
    OUT.write_text(html, encoding="utf-8")
    print(f"\nwrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
