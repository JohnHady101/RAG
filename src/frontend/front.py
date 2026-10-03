"""Frontend helper: sync src/frontend -> src/backend/static for FastAPI serving.

The canonical JS frontend lives in ``src/frontend/`` (``index.html``,
``app.js``, ``styles.css``). The FastAPI app (``src/backend/app.py``)
serves ``src/backend/static/`` at ``/`` via ``StaticFiles``.

Run from the repo root:

    python src/frontend/front.py        # copy files, print URLs
    python src/frontend/front.py --serve  # copy + launch uvicorn on :8000
"""

import os
import shutil
import sys

FRONTEND_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(FRONTEND_DIR, "..", "backend", "static")
FILES = ("index.html", "app.js", "styles.css")


def sync() -> None:
    os.makedirs(STATIC_DIR, exist_ok=True)
    for name in FILES:
        src = os.path.join(FRONTEND_DIR, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(STATIC_DIR, name))
    print(f"synced {FRONTEND_DIR} -> {STATIC_DIR}")


if __name__ == "__main__":
    sync()
    if "--serve" in sys.argv:
        import uvicorn

        uvicorn.run("src.backend.app:app", host="0.0.0.0", port=8000, reload=True)
    else:
        print("open http://localhost:8000/ (run: uvicorn src.backend.app:app --reload)")
