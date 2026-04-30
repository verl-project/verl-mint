from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_STORAGE_ENV = "VERL_MINT_STORAGE_ROOT"
_STORAGE_SHARED_ENV = "VERL_MINT_STORAGE_SHARED"
_DEFAULT_STORAGE_ROOT = "/tmp/verl-mint-storage"
_KNOWN_SHARED_ROOTS = (Path("/vePFS-Mindverse/share").resolve(),)


def _env_truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class LocalStorageRepo:
    root: Path

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "LocalStorageRepo":
        env = env or os.environ
        raw = env.get(_STORAGE_ENV, _DEFAULT_STORAGE_ROOT)
        return cls(Path(raw).expanduser().resolve())

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def is_probably_node_local(self) -> bool:
        tmp_root = Path("/tmp").resolve()
        try:
            self.root.relative_to(tmp_root)
            return True
        except ValueError:
            return False

    def is_explicitly_shared(self, env: dict[str, str] | None = None) -> bool:
        env = env or os.environ
        if _env_truthy(env.get(_STORAGE_SHARED_ENV)):
            return True
        return self.is_path_explicitly_shared(self.root, env=env)

    def is_path_explicitly_shared(self, path: Path, env: dict[str, str] | None = None) -> bool:
        env = env or os.environ
        if _env_truthy(env.get(_STORAGE_SHARED_ENV)):
            return True
        resolved = path.expanduser().resolve()
        for shared_root in _KNOWN_SHARED_ROOTS:
            try:
                resolved.relative_to(shared_root)
                return True
            except ValueError:
                continue
        return False

    def require_shared_storage(self, *, reason: str) -> None:
        if self.is_probably_node_local() or not self.is_explicitly_shared():
            raise ValueError(
                f"{reason} requires shared {storage_env_name()} under a known shared root "
                f"or {shared_storage_env_name()}=1; current root={self.root}"
            )

    def require_shared_path(self, path: Path, *, reason: str) -> None:
        resolved = path.expanduser().resolve()
        if not self.is_path_explicitly_shared(resolved):
            raise ValueError(
                f"{reason} requires a shared checkpoint path under a known shared root "
                f"or {shared_storage_env_name()}=1; current path={resolved}"
            )

    def resolve_for_write(self, uri: str) -> str:
        path = self._resolve(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)

    def resolve_for_read(self, uri: str) -> str:
        path = self._resolve(uri)
        if not path.exists():
            raise FileNotFoundError(f"artifact not found: {path}")
        return str(path)

    def _resolve(self, uri: str) -> Path:
        if uri.startswith("mint://"):
            relative = uri[len("mint://") :]
            return self._safe_join(relative)
        if uri.startswith("tinker://"):
            relative = uri[len("tinker://") :]
            return self._safe_join(relative)
        if uri.startswith("repo://"):
            # Legacy alias kept so older tests and artifacts can still load.
            relative = uri[len("repo://") :]
            return self._safe_join(relative)
        if uri.startswith("file://"):
            path = Path(uri[len("file://") :]).expanduser().resolve()
            self._ensure_inside_root(path)
            return path
        return self._safe_join(uri)

    def _safe_join(self, relative: str) -> Path:
        path = (self.root / relative).resolve()
        self._ensure_inside_root(path)
        return path

    def _ensure_inside_root(self, path: Path) -> None:
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"path escapes storage root: {path}") from exc


def storage_env_name() -> str:
    return _STORAGE_ENV


def shared_storage_env_name() -> str:
    return _STORAGE_SHARED_ENV


def default_storage_root() -> str:
    return _DEFAULT_STORAGE_ROOT
