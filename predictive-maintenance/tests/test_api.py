import json

import pytest

from predmaint import data as cmapss
from predmaint.api import MAX_READINGS, create_app
from predmaint.config import Settings


@pytest.fixture
def app(settings):
    return create_app(settings)


@pytest.fixture
def client(app):
    return app.test_client()


def _readings(cmapss_dir, unit=1, cycles=None):
    engine = cmapss.load_test(cmapss_dir, "FD001").query("unit == @unit")
    if cycles is not None:
        engine = engine.head(cycles)
    columns = [*cmapss.SETTINGS, *cmapss.SENSORS]
    return engine[columns].to_dict(orient="records")


def test_health_runs_a_smoke_inference(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "ok"
    assert body["ensemble_size"] == 2


def test_health_reports_unavailable_without_artifacts(tmp_path, cmapss_dir):
    client = create_app(
        Settings(data_dir=cmapss_dir, artifacts_dir=tmp_path, replay_autostart=False)
    ).test_client()
    assert client.get("/api/health").status_code == 503
    assert client.post("/api/predict", json={"readings": []}).status_code == 503


def test_model_info_comes_from_metrics_json(client, artifacts_dir):
    info = client.get("/api/model-info").get_json()
    metrics = json.loads((artifacts_dir / "FD001" / "metrics.json").read_text())
    assert info["test"]["rmse"] == metrics["test"]["rmse"]
    assert len(info["sensors"]) == 14


def test_predict_accepts_raw_cycles(client, cmapss_dir):
    response = client.post("/api/predict", json={"readings": _readings(cmapss_dir)})
    assert response.status_code == 200
    body = response.get_json()
    assert set(body) >= {"rul", "failure_within_horizon", "status", "explanation"}
    assert body["window_padded"] is False


def test_predict_pads_short_histories(client, cmapss_dir):
    body = client.post(
        "/api/predict?explain=false", json={"readings": _readings(cmapss_dir, cycles=3)}
    ).get_json()
    assert body["window_padded"] is True
    assert "explanation" not in body


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([1, 2, 3], "JSON object"),
        ({"readings": []}, "1 to"),
        ({"readings": "abc"}, "1 to"),
        ({"readings": [{"setting_1": 0}]}, "missing"),
        ({"readings": ["row"]}, "must be an object"),
    ],
)
def test_predict_rejects_malformed_bodies(client, payload, message):
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 400
    assert message in response.get_json()["detail"]


def test_predict_rejects_non_finite_and_boolean_values(client, cmapss_dir):
    reading = _readings(cmapss_dir, cycles=1)[0]
    for bad in (True, "12", None):
        response = client.post("/api/predict", json={"readings": [{**reading, "sensor_02": bad}]})
        assert response.status_code == 400
    raw = json.dumps({"readings": [reading]}).replace(str(reading["sensor_02"]), "NaN", 1)
    response = client.post("/api/predict", data=raw, content_type="application/json")
    assert response.status_code == 400
    assert "finite" in response.get_json()["detail"]


def test_predict_limits_history_length(client, cmapss_dir):
    reading = _readings(cmapss_dir, cycles=1)[0]
    response = client.post("/api/predict", json={"readings": [reading] * (MAX_READINGS + 1)})
    assert response.status_code == 400


def test_errors_are_json_without_internals(client):
    not_found = client.get("/api/nope")
    assert not_found.status_code == 404
    assert not_found.get_json()["error"] == "Not Found"
    wrong_method = client.get("/api/predict")
    assert wrong_method.status_code == 405
    assert wrong_method.is_json


def test_security_headers(client):
    headers = client.get("/").headers
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Frame-Options"] == "DENY"
    assert "default-src 'self'" in headers["Content-Security-Policy"]


def test_dashboard_has_no_hardcoded_metrics(client):
    html = client.get("/").get_data(as_text=True)
    assert "Turbofan fleet health" in html
    assert "92%" not in html
    assert "94%" not in html


def test_fleet_and_engine_detail(client, app):
    fleet = client.get("/api/fleet").get_json()
    assert len(fleet["engines"]) == 3
    unit = fleet["engines"][0]["unit"]
    detail = client.get(f"/api/engines/{unit}").get_json()
    assert detail["unit"] == unit
    assert detail["history"]
    assert client.get("/api/engines/99999").status_code == 404


def test_socket_watch_engine_returns_detail(app):
    socketio = app.extensions["predmaint"]["socketio"]
    unit = app.extensions["predmaint"]["replay"].units[0]
    ws = socketio.test_client(app)
    assert ws.is_connected()
    ack = ws.emit("watch_engine", {"unit": unit}, callback=True)
    assert ack["ok"]
    assert ack["detail"]["unit"] == unit
    assert ws.emit("watch_engine", {"unit": "not-an-int"}, callback=True) == {"ok": False}
    ws.disconnect()


def test_socket_handshake_rejects_cross_origin_by_default(client):
    """v1 accepted any origin (cors_allowed_origins="*"); v2 is same-origin unless configured."""
    handshake = "/socket.io/?EIO=4&transport=polling"
    assert client.get(handshake, headers={"Origin": "http://localhost"}).status_code == 200
    assert client.get(handshake, headers={"Origin": "http://evil.example"}).status_code == 400
