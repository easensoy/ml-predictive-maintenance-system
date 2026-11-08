"""Local development server: ``python -m predmaint.serve`` then open http://127.0.0.1:5000.

For anything reachable from other machines use gunicorn instead (see the Dockerfile):
``gunicorn -w 1 --threads 32 -b 0.0.0.0:8000 predmaint.wsgi:app``
"""

import ipaddress
import sys

from predmaint.api import create_app
from predmaint.config import Settings


def main() -> int:
    settings = Settings.from_env()
    try:
        loopback = settings.host == "localhost" or ipaddress.ip_address(settings.host).is_loopback
    except ValueError:
        loopback = False
    if not loopback:
        print(
            f"Refusing to bind the development server to {settings.host}. "
            "Use gunicorn for non-local serving: gunicorn -w 1 --threads 32 predmaint.wsgi:app",
            file=sys.stderr,
        )
        return 2
    app = create_app(settings)
    socketio = app.extensions["predmaint"]["socketio"]
    print(f"Dashboard: http://{settings.host}:{settings.port}")
    socketio.run(
        app,
        host=settings.host,
        port=settings.port,
        debug=settings.debug,
        use_reloader=False,
        allow_unsafe_werkzeug=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
