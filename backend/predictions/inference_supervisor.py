"""Lifecycle helper that runs the Flask inference service with ``runserver``.

The Django API gateway forwards every analysis to the Flask microservice at
``settings.FORGERY_API_URL``.  During local development that service used to
need a separate terminal, which is easy to forget and produced
``Inference service unreachable`` 503 responses.

The custom ``runserver`` management command uses this supervisor to boot the
Flask app together with Django and tear it down again when Django stops.
``start()`` is a no-op when:

* ``INFERENCE_AUTOSTART`` is disabled,
* ``FORGERY_API_URL`` points at a non-local host (e.g. Docker's
  ``http://inference:5000``), or
* the service is already healthy (started in another terminal/container).
"""

import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from django.conf import settings

LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}
HEALTH_POLL_INTERVAL = 0.5
_HEALTH_TIMEOUT = 2
_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _say(message):
    print(f"[inference] {message}", flush=True)


def _warn(message):
    print(f"[inference] WARNING: {message}", flush=True)


def _healthy(timeout=_HEALTH_TIMEOUT):
    """Return True when the inference service answers ``GET /health``."""
    url = settings.FORGERY_API_URL.rstrip("/") + "/health"
    try:
        return requests.get(url, timeout=timeout).status_code == 200
    except requests.RequestException:
        return False


def _probe(python_exe, condition):
    """Return True when ``python_exe`` satisfies ``condition`` (a Python expression)."""
    code = (
        "import importlib.util as u, sys; "
        f"sys.exit(0 if {condition} else 1)"
    )
    try:
        result = subprocess.run(
            [str(python_exe), "-c", code],
            capture_output=True,
            timeout=20,
            creationflags=_CREATE_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


# The inference app needs Flask plus TensorFlow (model_utils imports it lazily
# when the model is first used).
_WANTS = "u.find_spec('flask') and u.find_spec('tensorflow')"
_FLASK_ONLY = "u.find_spec('flask')"


class InferenceSupervisor:
    """Starts the Flask app as a child process and stops it again."""

    def __init__(self, verbosity=1):
        self.verbosity = int(verbosity)
        self._proc = None

    def _log(self, message):
        if self.verbosity >= 1:
            _say(message)

    def _app_dir(self):
        return Path(settings.INFERENCE_APP_DIR)

    def _targets_local_service(self):
        return urlparse(settings.FORGERY_API_URL).hostname in LOCAL_HOSTS

    def _candidate_interpreters(self):
        if settings.INFERENCE_PYTHON:
            yield Path(settings.INFERENCE_PYTHON)
        app_dir = self._app_dir()
        for base in (
            app_dir / "backend",
            app_dir,
            app_dir.parent,
        ):
            for name in ("env", ".venv", "venv"):
                if os.name == "nt":
                    yield base / name / "Scripts" / "python.exe"
                else:
                    yield base / name / "bin" / "python3"

    def resolve_interpreter(self):
        """Find the best Python interpreter to run the Flask app with.

        Prefers a virtualenv with Flask *and* TensorFlow (the model cannot run
        without TensorFlow), then falls back to Flask-only, then to the
        interpreter Django itself is running under.
        """
        candidates = [c for c in self._candidate_interpreters() if c.is_file()]
        if settings.INFERENCE_PYTHON and candidates:
            # Explicitly configured: use it as-is.
            return str(candidates[0])
        for condition in (_WANTS, _FLASK_ONLY):
            for candidate in candidates:
                if _probe(candidate, condition):
                    return str(candidate)
        _warn(
            "No interpreter with Flask found (set INFERENCE_PYTHON to point at "
            f"one); falling back to {sys.executable}."
        )
        return sys.executable

    def start(self):
        """Start the Flask service unless it is disabled or already running.

        Returns True when this supervisor owns the child process (and will
        therefore stop it in :meth:`stop`).
        """
        if not settings.INFERENCE_AUTOSTART:
            return False
        if not self._targets_local_service():
            self._log(f"remote inference at {settings.FORGERY_API_URL}; not auto-starting")
            return False

        app_dir = self._app_dir()
        if not (app_dir / "app.py").is_file():
            _warn(f"app.py not found in {app_dir}; not auto-starting")
            return False

        if _healthy(timeout=3):
            self._log(f"already running at {settings.FORGERY_API_URL}")
            return False

        interpreter = self.resolve_interpreter()
        command = [interpreter, "app.py"]
        self._log(f"starting Flask inference service: {' '.join(command)} (cwd={app_dir})")
        try:
            self._proc = subprocess.Popen(
                command,
                cwd=str(app_dir),
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
        except OSError as exc:
            _warn(f"could not start inference service: {exc}")
            return False

        if self._wait_until_healthy():
            return True
        self.stop()
        return False

    def _wait_until_healthy(self):
        deadline = time.monotonic() + settings.INFERENCE_STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                _warn(
                    f"inference service exited with code {self._proc.returncode}; "
                    "see its output above for the reason"
                )
                self._proc = None
                return False
            if _healthy():
                self._log(f"healthy at {settings.FORGERY_API_URL}")
                return True
            time.sleep(HEALTH_POLL_INTERVAL)
        _warn(
            f"service did not become healthy within "
            f"{settings.INFERENCE_STARTUP_TIMEOUT}s"
        )
        return False

    def stop(self):
        """Stop the Flask child process if this supervisor started it."""
        proc, self._proc = self._proc, None
        if proc is None or proc.poll() is not None:
            return
        self._log("stopping Flask inference service")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
