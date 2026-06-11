import importlib
import os
import sys


def log(message):
    print(f"[startup] {message}", flush=True)


log(f"python={sys.version}")
log(f"cwd={os.getcwd()}")
log(f"HF_HOME={os.environ.get('HF_HOME')}")

for module in ["fastapi", "uvicorn", "pydantic", "eval_type_backport", "numpy"]:
    try:
        imported = importlib.import_module(module)
        log(f"import {module}: ok {getattr(imported, '__version__', '')}")
    except Exception as exc:
        log(f"import {module}: {type(exc).__name__}: {exc}")
        raise

log("import app: start")
import app  # noqa: F401
log("import app: ok")

log("starting uvicorn")
import uvicorn

uvicorn.run("app:app", host="0.0.0.0", port=7860, timeout_keep_alive=120)
