from __future__ import annotations

import importlib
from pathlib import Path
import sys


def test_top_level_import_does_not_load_verl_backend() -> None:
    sys.modules.pop("verl_mint", None)
    sys.modules.pop("verl_mint.backends.verl", None)

    module = importlib.import_module("verl_mint")

    assert module.create_app is not None
    assert "verl_mint.backends.verl" not in sys.modules


def test_verl_backend_export_is_lazy() -> None:
    sys.modules.pop("verl_mint", None)
    sys.modules.pop("verl_mint.backends.verl", None)

    module = importlib.import_module("verl_mint")
    backend_cls = module.VerlTrainingBackend

    assert backend_cls.__name__ == "VerlTrainingBackend"
    assert "verl_mint.backends.verl" in sys.modules


def test_client_smokes_use_client_surface() -> None:
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    forbidden = (
        "import urllib",
        "from urllib",
        "import requests",
        "from requests",
        "import httpx",
        "from httpx",
        "from server_smoke",
        "import server_smoke",
        "post_json",
        "retrieve_future",
        "TestClient",
        "import uvicorn",
        "from uvicorn",
        "from verl_mint import create_app",
        "from verl_mint.backends",
        "from verl_mint.contracts import BackendKind",
        "from verl_mint.contracts import BackendSpec",
        "from verl_mint.service import MintService",
        "BackendSpec(",
        "MintService(",
        "create_app(",
    )

    for path in scripts_dir.glob("client_*_smoke.py"):
        source = path.read_text(encoding="utf-8")
        assert "client_scenarios" in source, path.name
        for token in forbidden:
            assert token not in source, f"{path.name} must use the client surface, found {token!r}"
