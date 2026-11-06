from pathlib import Path

import pytest

from predmaint.config import Settings


def test_defaults_are_safe():
    settings = Settings()
    assert settings.host == "127.0.0.1"
    assert settings.debug is False
    assert settings.cors_origins == ()


def test_environment_overrides_are_typed():
    settings = Settings.from_env(
        {
            "PM_SUBSET": "FD003",
            "PM_PORT": "8080",
            "PM_DEBUG": "yes",
            "PM_REPLAY_INTERVAL_S": "0.5",
            "PM_ARTIFACTS_DIR": "/tmp/models",
            "PM_CORS_ORIGINS": "https://a.example, https://b.example",
        }
    )
    assert settings.subset == "FD003"
    assert settings.port == 8080
    assert settings.debug is True
    assert settings.replay_interval_s == 0.5
    assert settings.artifacts_dir == Path("/tmp/models")
    assert settings.model_dir == Path("/tmp/models/FD003")
    assert settings.cors_origins == ("https://a.example", "https://b.example")


@pytest.mark.parametrize(
    "environ",
    [{"PM_SUBSET": "FD009"}, {"PM_PORT": "abc"}, {"PM_PORT": "70000"}, {"PM_DEBUG": "maybe"}],
)
def test_bad_values_fail_loudly(environ):
    with pytest.raises(ValueError, match="PM_"):
        Settings.from_env(environ)
