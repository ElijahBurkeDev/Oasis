"""
Main.py

Backend-agnostic model harness and chat shell.

Run:
    python main.py                          # uses lm_studio_backend.py next to this file
    python main.py --backend my_backend     # module name (my_backend.py next to main.py)
    python main.py --backend ./path/x.py    # or any file path

The default backend can also be set with the HARNESS_BACKEND environment variable.

A backend is any Python file that exposes (see lm_studio_backend.py for a full example):

    def create_backend():            # returns an object with:
        ...
        configure()      -> bool     interactive setup; False aborts
        send(text)       -> reply    one chat turn; a str, or an object with
                                     .text and optional .reasoning / .info
        handle_command(text) -> bool run a "/command"; False if not recognised
        # optional: help_text(), failure_hint(error), close(), name,
        #           on_ready()  called once, just before the first prompt
    # optional module attribute: BackendError (exception base class)
"""
from __future__ import annotations

import argparse
import datetime
import json
import importlib
import importlib.util
import os
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_BACKEND = os.environ.get("HARNESS_BACKEND", "lm_studio_backend")
REQUIRED_METHODS = ("configure", "send", "handle_command")
EXIT_COMMANDS = {"/exit", "/quit", "/q"}
DEBUG = os.environ.get("CHAT_DEBUG") == "1"


class HarnessError(Exception):
    """Problem loading or validating a backend."""


# ---------------------------------------------------------------------------
# Backend loading
# ---------------------------------------------------------------------------
def _exec_module_from_file(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise HarnessError(f"Cannot load a Python module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def load_backend_module(target: str) -> ModuleType:
    """Import a backend by module name or by .py file path."""
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))  # so sibling files import regardless of cwd

    try:
        if target.lower().endswith(".py") or "/" in target or os.sep in target:
            raw = Path(target).expanduser()
            candidates = [raw] if raw.is_absolute() else [Path.cwd() / raw, HERE / raw]
            path = next((c for c in candidates if c.is_file()), None)
            if path is None:
                raise HarnessError(f"Backend file not found: {target}")
            return _exec_module_from_file(path.resolve())
        try:
            return importlib.import_module(target)
        except ModuleNotFoundError as error:
            if error.name == target:
                raise HarnessError(
                    f"Backend module '{target}' not found. Put {target}.py in {HERE} "
                    "or pass --backend path/to/file.py"
                ) from error
            raise
    except HarnessError:
        raise
    except Exception as error:
        raise HarnessError(f"Backend '{target}' failed to import: {error!r}") from error


def build_backend(module: ModuleType) -> Any:
    factory = getattr(module, "create_backend", None)
    if not callable(factory):
        raise HarnessError(f"Backend '{module.__name__}' has no create_backend() function.")
    try:
        backend = factory()
    except Exception as error:
        raise HarnessError(f"create_backend() failed: {error!r}") from error
    missing = [m for m in REQUIRED_METHODS if not callable(getattr(backend, m, None))]
    if missing:
        raise HarnessError(f"Backend object is missing required method(s): {', '.join(missing)}")
    return backend


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------
def run_turn(backend: Any, text: str) -> None:
    """One user message in, one reply out."""
    print("\nAssistant: ", end="", flush=True)
    started = time.perf_counter()
    reply = backend.send(text)
    elapsed = time.perf_counter() - started

    answer = reply if isinstance(reply, str) else getattr(reply, "text", "")
    reasoning = getattr(reply, "reasoning", "")
    info = getattr(reply, "info", "")

    if reasoning:
        print(f"\n[thinking]\n{reasoning}\n[/thinking]\n")
    print(answer)
    print(f"\n[Generated in {elapsed:.1f} seconds{', ' + info if info else ''}]")


def print_help(backend: Any) -> None:
    print("\nCommands:\n  /help       Show this command list\n  /exit       End the program")
    help_text = getattr(backend, "help_text", None)
    if callable(help_text):
        print(help_text())


def report_error(error: Exception, backend: Any, backend_error: type, label: str) -> None:
    if isinstance(error, backend_error):
        print(f"\n{label}: {error}")
        hint = getattr(backend, "failure_hint", None)
        if callable(hint):
            try:
                text = hint(error)
                if text:
                    print(text)
            except Exception:
                pass
    else:
        print(f"\nUnexpected error: {error!r}")
        if not DEBUG:
            print("Set CHAT_DEBUG=1 for a full traceback.")
    if DEBUG:
        traceback.print_exc()


def chat_loop(module: ModuleType, backend: Any) -> None:
    backend_error = getattr(module, "BackendError", Exception)
    title = getattr(backend, "name", module.__name__)

    print("\n" + "=" * 60)
    print(f"Harness ready - backend: {title}")
    print("=" * 60)
    print_help(backend)
    on_ready = getattr(backend, "on_ready", None)  # optional startup hook (e.g. a mood chart)
    if callable(on_ready):
        try:
            on_ready()
        except Exception:
            if DEBUG:
                traceback.print_exc()

    while True:
        try:
            text = input("\nYou: ").strip()
        except EOFError:
            print("\nInput stream closed. Exiting.")
            return
        except KeyboardInterrupt:
            print("\n\nInterrupted. Type /exit to quit.")
            continue
        if not text:
            continue

        try:
            if text.startswith("/"):
                command = text.split()[0].lower()
                if command in EXIT_COMMANDS:
                    print("Goodbye.")
                    return
                if command == "/help":
                    print_help(backend)
                elif not backend.handle_command(text):
                    print(f"Unknown command '{command}'. Type /help for the list.")
            else:
                run_turn(backend, text)
        except KeyboardInterrupt:
            print("\nRequest interrupted.")
        except Exception as error:  # never let one bad turn kill the harness
            report_error(error, backend, backend_error, "Request failed")


def configure_stdio() -> None:
    """Prevent UnicodeEncodeError on consoles that are not UTF-8 (e.g. Windows)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:
            pass
    try:
        sys.stdin.reconfigure(errors="replace")  # type: ignore[union-attr]
    except Exception:
        pass

        


def main(argv: list[str] | None = None) -> int:
    if sys.version_info < (3, 10):
        print("This script requires Python 3.10 or newer.")
        return 1
    configure_stdio()

    parser = argparse.ArgumentParser(description="Backend-agnostic model harness")
    parser.add_argument("-b", "--backend", default=DEFAULT_BACKEND,
                        help="backend module name or .py path (default: %(default)s)")
    args = parser.parse_args(argv)

    try:
        module = load_backend_module(args.backend)
        backend = build_backend(module)
    except HarnessError as error:
        print(f"\n{error}")
        return 1

    try:
        if not backend.configure():
            return 1
        chat_loop(module, backend)
        return 0
    except KeyboardInterrupt:
        print("\nProgram interrupted.")
        return 130
    except Exception as error:
        print(f"\nUnexpected fatal error: {error}")
        if DEBUG:
            traceback.print_exc()
        else:
            print("Set CHAT_DEBUG=1 before running to display a complete traceback.")
        return 1
    finally:
        close = getattr(backend, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
