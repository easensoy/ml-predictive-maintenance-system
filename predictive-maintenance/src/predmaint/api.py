"""Flask + Socket.IO application: JSON API, live fleet replay and the dashboard."""

import logging
import math
import threading

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from flask.json.provider import DefaultJSONProvider
from flask_socketio import SocketIO, join_room, leave_room
from werkzeug.exceptions import HTTPException

from predmaint import __version__
from predmaint.config import Settings
from predmaint.data import SETTINGS
from predmaint.inference import ArtifactsMissing, NonFinitePrediction, Predictor
from predmaint.replay import FleetReplay

log = logging.getLogger(__name__)
MAX_READINGS = 1000
CSP = (
    "default-src 'self'; script-src 'self' https://cdn.jsdelivr.net; "
    "style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; "
    "frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
)


class BadRequest(ValueError):
    pass


def _sid() -> str:
    """The Socket.IO session id, which Flask-SocketIO attaches to the request in handlers."""
    return request.sid  # type: ignore[attr-defined]


def _finite_float(value: object) -> float | None:
    """The value as a float if it is a finite JSON number, else None.

    ``bool`` is excluded because it is an ``int`` subclass. ``float()`` is guarded because JSON
    integers have no size limit and ``float(10**400)`` raises OverflowError.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def parse_readings(body: object, required: list[str]) -> pd.DataFrame:
    if not isinstance(body, dict):
        raise BadRequest("Body must be a JSON object with a 'readings' list")
    readings = body.get("readings")
    if not isinstance(readings, list) or not 1 <= len(readings) <= MAX_READINGS:
        raise BadRequest(f"'readings' must be a list of 1 to {MAX_READINGS} cycles, oldest first")
    rows = []
    for i, reading in enumerate(readings):
        if not isinstance(reading, dict):
            raise BadRequest(f"readings[{i}] must be an object")
        missing = [key for key in required if key not in reading]
        if missing:
            raise BadRequest(f"readings[{i}] is missing {missing}")
        row = {}
        for key in required:
            number = _finite_float(reading[key])
            if number is None:
                raise BadRequest(f"readings[{i}].{key} must be a finite number")
            row[key] = number
        rows.append(row)
    return pd.DataFrame(rows)


def create_app(settings: Settings | None = None, predictor: Predictor | None = None) -> Flask:
    settings = settings or Settings.from_env()
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 1_000_000
    if isinstance(app.json, DefaultJSONProvider):
        app.json.sort_keys = False  # keep response fields in a readable order

    if predictor is None:
        try:
            predictor = Predictor(settings.model_dir)
        except ArtifactsMissing as exc:
            log.warning("%s", exc)
    replay = None
    replay_csv = settings.model_dir / "replay.csv"
    if predictor is not None and replay_csv.is_file():
        replay = FleetReplay(predictor, pd.read_csv(replay_csv), settings.replay_engines)

    # Without cors_allowed_origins, Flask-SocketIO accepts same-origin connections only.
    if settings.cors_origins:
        socketio = SocketIO(
            app, async_mode="threading", cors_allowed_origins=list(settings.cors_origins)
        )
    else:
        socketio = SocketIO(app, async_mode="threading")
    watching: dict[str, int] = {}
    watching_lock = threading.Lock()
    app.extensions["predmaint"] = {
        "settings": settings,
        "predictor": predictor,
        "replay": replay,
        "socketio": socketio,
    }

    def unavailable():
        return jsonify(error="Service Unavailable", detail="Model artifacts are not loaded"), 503

    @app.after_request
    def security_headers(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        response.headers.setdefault("Content-Security-Policy", CSP)
        return response

    @app.errorhandler(HTTPException)
    def http_error(exc: HTTPException):
        return jsonify(error=exc.name, detail=exc.description), exc.code

    @app.errorhandler(Exception)
    def unhandled(exc: Exception):
        log.exception("Unhandled error")
        return jsonify(error="Internal Server Error"), 500

    @app.get("/")
    def dashboard():
        return render_template("dashboard.html", version=__version__, subset=settings.subset)

    @app.get("/api/health")
    def health():
        if predictor is None:
            return jsonify(status="unavailable", detail="Model artifacts are not loaded"), 503
        try:
            window = np.zeros(
                (1, predictor.contract.window, predictor.contract.n_features), np.float32
            )
            point = float(predictor.predict_windows(window)["point"][0])
        except Exception:
            log.exception("Health check inference failed")
            return jsonify(status="unavailable", detail="Smoke inference failed"), 503
        if not math.isfinite(point):
            return jsonify(status="unavailable", detail="Smoke inference returned NaN"), 503
        return jsonify(
            status="ok",
            version=__version__,
            subset=predictor.contract.subset,
            ensemble_size=len(predictor.model.members),
            model_created_at=predictor.metrics["created_at"],
            replay_running=replay is not None and settings.replay_autostart,
        )

    @app.get("/api/model-info")
    def model_info():
        if predictor is None:
            return unavailable()
        return jsonify(predictor.info())

    @app.post("/api/predict")
    def predict():
        if predictor is None:
            return unavailable()
        try:
            frame = parse_readings(
                request.get_json(silent=True), [*SETTINGS, *predictor.contract.sensors]
            )
        except BadRequest as exc:
            return jsonify(error="Bad Request", detail=str(exc)), 400
        explain = request.args.get("explain", "true").lower() != "false"
        try:
            return jsonify(predictor.predict_readings(frame, explain=explain))
        except NonFinitePrediction as exc:
            return jsonify(
                error="Unprocessable Content", detail=str(exc), out_of_distribution=exc.report
            ), 422

    @app.get("/api/fleet")
    def fleet():
        if replay is None:
            return unavailable()
        return jsonify(replay.snapshot())

    @app.get("/api/engines/<int:unit>")
    def engine(unit: int):
        if replay is None:
            return unavailable()
        try:
            return jsonify(replay.detail(unit))
        except KeyError:
            return jsonify(error="Not Found", detail=f"Engine {unit} is not in the replay"), 404

    @socketio.on("watch_engine")
    def watch_engine(payload):
        if (
            replay is None
            or not isinstance(payload, dict)
            or not isinstance(payload.get("unit"), int)
        ):
            return {"ok": False}
        unit = payload["unit"]
        if unit not in replay.units:
            return {"ok": False}
        with watching_lock:
            previous = watching.get(_sid())
            watching[_sid()] = unit
        if previous is not None:
            leave_room(f"engine:{previous}")
        join_room(f"engine:{unit}")
        return {"ok": True, "detail": replay.detail(unit)}

    @socketio.on("disconnect")
    def disconnect(*_args):
        with watching_lock:
            watching.pop(_sid(), None)

    def replay_loop(fleet: FleetReplay) -> None:
        while True:
            # The stub says int, but Flask-SocketIO sleeps for any number of seconds.
            socketio.sleep(settings.replay_interval_s)  # type: ignore[arg-type]
            try:
                socketio.emit("fleet_update", fleet.step())
                with watching_lock:
                    watched = set(watching.values())
                for unit in watched:
                    socketio.emit("engine_update", fleet.detail(unit), to=f"engine:{unit}")
            except Exception:
                log.exception("Replay tick failed")

    if replay is not None and settings.replay_autostart:
        socketio.start_background_task(replay_loop, replay)

    return app
