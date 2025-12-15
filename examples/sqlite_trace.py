"""SQLite-backed TraceConfig for recording aiohttp client activity.

This module uses only the public aiohttp API. It defines a TraceConfig that
logs request lifecycle events into a SQLite database using the following
schema:

* ``events``: base row for every event with timestamp, request id, and type.
* ``request_start``: method, URL, and headers captured when a request begins.
* ``request_end``: response status, headers, and elapsed duration.
* ``request_exception``: exception text for failed requests.
* ``request_chunk``: bytes sent from the client along with an incremental
  chunk index.
* ``response_chunk``: bytes received in the response body with an incremental
  chunk index.

Example
=======

Create the tables once and attach the trace to a session:

>>> import asyncio
>>> import sqlite3
>>> from aiohttp import ClientSession
>>> from examples.sqlite_trace import SQLiteTraceRecorder, init_sqlite
>>>
>>> conn = sqlite3.connect("trace.db")
>>> init_sqlite(conn)
>>> recorder = SQLiteTraceRecorder(conn)
>>>
>>> async def main():
...     async with ClientSession(trace_configs=[recorder.trace_config]) as session:
...         async with session.get("https://example.com") as resp:
...             await resp.text()
...         # All events have been recorded at this point.
...
>>> asyncio.run(main())
>>> conn.close()

"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from types import SimpleNamespace
from typing import Iterable

from aiohttp import (
    ClientSession,
    TraceConfig,
    TraceRequestChunkSentParams,
    TraceRequestEndParams,
    TraceRequestExceptionParams,
    TraceRequestStartParams,
    TraceResponseChunkReceivedParams,
)


def init_sqlite(conn: sqlite3.Connection) -> None:
    """Create the required tables if they do not exist."""

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            request_id TEXT NOT NULL,
            event_type TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS request_start (
            event_id INTEGER NOT NULL REFERENCES events(id),
            method TEXT NOT NULL,
            url TEXT NOT NULL,
            headers TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS request_end (
            event_id INTEGER NOT NULL REFERENCES events(id),
            status INTEGER NOT NULL,
            reason TEXT,
            headers TEXT NOT NULL,
            elapsed REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS request_exception (
            event_id INTEGER NOT NULL REFERENCES events(id),
            exception TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS request_chunk (
            event_id INTEGER NOT NULL REFERENCES events(id),
            chunk_index INTEGER NOT NULL,
            size INTEGER NOT NULL,
            payload BLOB NOT NULL
        );

        CREATE TABLE IF NOT EXISTS response_chunk (
            event_id INTEGER NOT NULL REFERENCES events(id),
            chunk_index INTEGER NOT NULL,
            size INTEGER NOT NULL,
            payload BLOB NOT NULL
        );
        """
    )
    conn.commit()


def _serialize_headers(headers: Iterable[tuple[str, str]]) -> str:
    return json.dumps(list(headers))


def _insert_event(conn: sqlite3.Connection, request_id: str, event_type: str) -> int:
    cur = conn.execute(
        "INSERT INTO events(ts, request_id, event_type) VALUES (?, ?, ?)",
        (time.time(), request_id, event_type),
    )
    return cur.lastrowid


class SQLiteTraceRecorder:
    """TraceConfig that records aiohttp client events into SQLite.

    The recorder keeps per-request state (request id, start time, and chunk
    counters) inside the trace context. All inserts are executed synchronously
    on the provided SQLite connection; the caller is responsible for connection
    lifecycle management.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self.trace_config = TraceConfig(trace_config_ctx_factory=self._ctx_factory)
        self.trace_config.on_request_start.append(self._on_request_start)
        self.trace_config.on_request_end.append(self._on_request_end)
        self.trace_config.on_request_exception.append(self._on_request_exception)
        self.trace_config.on_request_chunk_sent.append(self._on_request_chunk_sent)
        self.trace_config.on_response_chunk_received.append(
            self._on_response_chunk_received
        )

    @staticmethod
    def _ctx_factory() -> SimpleNamespace:
        return SimpleNamespace(
            request_id=uuid.uuid4().hex,
            started=time.time(),
            request_chunk_index=0,
            response_chunk_index=0,
        )

    async def _on_request_start(
        self, session: ClientSession, ctx: SimpleNamespace, params: TraceRequestStartParams
    ) -> None:
        event_id = _insert_event(self._conn, ctx.request_id, "request_start")
        self._conn.execute(
            "INSERT INTO request_start(event_id, method, url, headers) VALUES (?, ?, ?, ?)",
            (
                event_id,
                params.method,
                str(params.url),
                _serialize_headers(params.headers.items()),
            ),
        )
        self._conn.commit()

    async def _on_request_end(
        self, session: ClientSession, ctx: SimpleNamespace, params: TraceRequestEndParams
    ) -> None:
        elapsed = time.time() - ctx.started
        event_id = _insert_event(self._conn, ctx.request_id, "request_end")
        response = params.response
        self._conn.execute(
            """
            INSERT INTO request_end(event_id, status, reason, headers, elapsed)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                event_id,
                response.status,
                response.reason,
                _serialize_headers(response.headers.items()),
                elapsed,
            ),
        )
        self._conn.commit()

    async def _on_request_exception(
        self,
        session: ClientSession,
        ctx: SimpleNamespace,
        params: TraceRequestExceptionParams,
    ) -> None:
        event_id = _insert_event(self._conn, ctx.request_id, "request_exception")
        self._conn.execute(
            "INSERT INTO request_exception(event_id, exception) VALUES (?, ?)",
            (event_id, repr(params.exception)),
        )
        self._conn.commit()

    async def _on_request_chunk_sent(
        self,
        session: ClientSession,
        ctx: SimpleNamespace,
        params: TraceRequestChunkSentParams,
    ) -> None:
        event_id = _insert_event(self._conn, ctx.request_id, "request_chunk")
        self._conn.execute(
            """
            INSERT INTO request_chunk(event_id, chunk_index, size, payload)
            VALUES (?, ?, ?, ?)
            """,
            (event_id, ctx.request_chunk_index, len(params.chunk), params.chunk),
        )
        ctx.request_chunk_index += 1
        self._conn.commit()

    async def _on_response_chunk_received(
        self,
        session: ClientSession,
        ctx: SimpleNamespace,
        params: TraceResponseChunkReceivedParams,
    ) -> None:
        event_id = _insert_event(self._conn, ctx.request_id, "response_chunk")
        self._conn.execute(
            """
            INSERT INTO response_chunk(event_id, chunk_index, size, payload)
            VALUES (?, ?, ?, ?)
            """,
            (event_id, ctx.response_chunk_index, len(params.chunk), params.chunk),
        )
        ctx.response_chunk_index += 1
        self._conn.commit()


__all__ = ["SQLiteTraceRecorder", "init_sqlite"]
