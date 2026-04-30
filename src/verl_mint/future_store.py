from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PendingFutureError(Exception):
    queue_state: str = "active"
    queue_state_reason: str | None = None


@dataclass
class FutureItem:
    payload: Any
    allow_try_again_once: bool = False
    metadata_only_threshold: int = 4096
    queue_state: str = "active"
    queue_state_reason: str | None = None
    pending_polls: int = 0
    metadata_sent: bool = False
    try_again_sent: bool = False
    payload_size: int = field(init=False)

    def __post_init__(self) -> None:
        self.payload_size = len(json.dumps(self.payload))


@dataclass
class InMemoryFutureStore:
    _items: dict[str, FutureItem]

    def __init__(self) -> None:
        self._items = {}

    def create_resolved(
        self,
        payload: Any,
        *,
        allow_try_again_once: bool = False,
        metadata_only_threshold: int = 4096,
        pending_polls: int = 0,
        queue_state: str = "active",
        queue_state_reason: str | None = None,
    ) -> str:
        request_id = uuid.uuid4().hex
        self._items[request_id] = FutureItem(
            payload=payload,
            allow_try_again_once=allow_try_again_once,
            metadata_only_threshold=metadata_only_threshold,
            pending_polls=pending_polls,
            queue_state=queue_state,
            queue_state_reason=queue_state_reason,
        )
        return request_id

    def retrieve(self, request_id: str, *, allow_metadata_only: bool = False) -> Any:
        if request_id not in self._items:
            raise KeyError(request_id)
        item = self._items[request_id]
        if item.pending_polls > 0:
            item.pending_polls -= 1
            raise PendingFutureError(queue_state=item.queue_state, queue_state_reason=item.queue_state_reason)
        if item.allow_try_again_once and not item.try_again_sent:
            item.try_again_sent = True
            return {"type": "try_again", "request_id": request_id, "queue_state": item.queue_state}
        if allow_metadata_only and not item.metadata_sent and item.payload_size > item.metadata_only_threshold:
            item.metadata_sent = True
            return {"status": "complete_metadata", "response_payload_size": item.payload_size}
        payload = item.payload
        del self._items[request_id]
        return payload
