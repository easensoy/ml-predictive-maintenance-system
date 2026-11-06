"""WSGI entry point for gunicorn: ``gunicorn -w 1 --threads 32 predmaint.wsgi:app``.

Use a single worker: the fleet replay and Socket.IO rooms live in process memory.
"""

from predmaint.api import create_app

app = create_app()
