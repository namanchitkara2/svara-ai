"""WhatsAppProvider backed by WaCallsNative (jobasfernandes/WaCallsNative@develop).

Integration mechanism (see docs/research.md §2):
- Control: WaCalls REST API on loopback, `Authorization: Bearer $WACALLS_API_TOKEN`.
- Call state: the `/api/events` SSE stream (call-status / call-ended), with REST polling as a fallback.
- Audio: this process plays the role WaCalls' React client normally plays. It is the WebRTC
  *offerer*, opens a data channel labelled exactly "pcm", and POSTs the SDP offer to
  /api/sessions/{sid}/calls/{id}/webrtc. The server (pion) answers. Both directions carry raw
  16 kHz mono s16le PCM. No media tracks, no browser, no virtual audio device.
- The audio leg is attached right after the call is placed (still ringing), because WaCalls
  drops decoded peer audio when no bridge is attached.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator

import httpx
from aiortc import RTCPeerConnection, RTCSessionDescription

from services.audio.pacer import OutboundPacer
from services.config import mask_phone
from services.observability import event
from services.whatsapp.base import (
    AuthStatus,
    CallState,
    CallStatus,
    NotAuthenticated,
    WhatsAppError,
    WhatsAppProvider,
)

log = logging.getLogger("wake.wacalls")


def restrict_ice_hosts(hosts: list[str]) -> None:
    """Offer only these host candidates on the audio leg (default: loopback).

    The audio peer always runs next to WaCalls (same host, or the same container network
    namespace), so loopback is sufficient. It also sidesteps the macOS Application Firewall,
    which silently drops UDP between two unsigned local processes on en0/IPv6 host candidates
    (verified: ICE stuck in 'checking' until this restriction was applied). Pion accepts our
    127.0.0.1 checks as peer-reflexive, so WaCalls needs no configuration change.
    """
    import aioice.ice

    aioice.ice.get_host_addresses = lambda use_ipv4, use_ipv6: list(hosts)


PCM_LABEL = "pcm"
INBOUND_QUEUE_FRAMES = 50  # ~3 s of 60 ms frames; older frames are dropped if we fall behind


@dataclass
class _Call:
    call_id: str
    client_id: str
    status: CallStatus
    pc: RTCPeerConnection | None = None
    dc: object | None = None
    pacer: OutboundPacer | None = None
    inbound: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=INBOUND_QUEUE_FRAMES))
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    dc_open: asyncio.Event = field(default_factory=asyncio.Event)
    inbound_frames: int = 0
    inbound_dropped: int = 0


def _map_status(s: str) -> CallState:
    return {
        "starting": CallState.STARTING,
        "ringing": CallState.RINGING,
        "connected": CallState.CONNECTED,
        "reconnecting": CallState.RECONNECTING,
        "ended": CallState.ENDED,
    }.get(s, CallState.UNKNOWN)


class WaCallsProvider(WhatsAppProvider):
    def __init__(self, base_url: str, api_token: str, session_id: str = "", ice_hosts: list[str] | None = None):
        if not api_token:
            raise WhatsAppError("WACALLS_API_TOKEN is not set (see .env.example)")
        restrict_ice_hosts(ice_hosts or ["127.0.0.1"])
        self.base_url = base_url
        self._token = api_token
        self.session_id = session_id or None
        self._http: httpx.AsyncClient | None = None
        self._sse_task: asyncio.Task | None = None
        self._calls: dict[str, _Call] = {}
        self._sse_up = asyncio.Event()

    # ------------------------------------------------------------- lifecycle
    async def connect(self) -> None:
        if self._http is None:
            self._http = httpx.AsyncClient(
                base_url=self.base_url,
                headers={"Authorization": f"Bearer {self._token}"},
                timeout=httpx.Timeout(15.0),
            )
        try:
            r = await self._http.get("/healthz")
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise WhatsAppError(f"WaCalls not reachable at {self.base_url}: {type(e).__name__}") from e
        if self._sse_task is None:
            self._sse_task = asyncio.create_task(self._sse_loop(), name="wacalls-sse")
            try:
                await asyncio.wait_for(self._sse_up.wait(), 5)
            except asyncio.TimeoutError:
                log.warning("SSE not up yet; falling back to polling until it connects")

    async def close(self) -> None:
        for cid in list(self._calls):
            await self._teardown(cid)
        if self._sse_task:
            self._sse_task.cancel()
        if self._http:
            await self._http.aclose()
            self._http = None

    async def authenticate(self) -> AuthStatus:
        sessions = await self.list_sessions()
        chosen = None
        if self.session_id:
            chosen = next((s for s in sessions if s["id"] == self.session_id), None)
        else:
            chosen = next((s for s in sessions if s.get("paired") and s.get("state") == "open"), None) or next(
                (s for s in sessions if s.get("paired")), None
            )
        if not chosen:
            raise NotAuthenticated("no WhatsApp session is paired in WaCalls (run: wake-agent pair)")
        self.session_id = chosen["id"]
        jid = chosen.get("jid") or ""
        status = AuthStatus(
            session_id=chosen["id"],
            paired=bool(chosen.get("paired")),
            state=chosen.get("state", "unknown"),
            jid_masked=mask_phone(jid.split("@")[0].split(":")[0]) if jid else None,
        )
        if not status.paired:
            raise NotAuthenticated(f"session {status.session_id} is not paired (state={status.state})")
        if status.state != "open":
            raise NotAuthenticated(f"session {status.session_id} is paired but not connected (state={status.state})")
        return status

    async def list_sessions(self) -> list[dict]:
        r = await self._req("GET", "/api/sessions")
        return r.json().get("sessions", [])

    async def create_session(self, name: str) -> str:
        r = await self._req("POST", "/api/sessions", json={"name": name})
        return r.json()["id"]

    # ------------------------------------------------------------------ call
    async def call(self, phone: str) -> str:
        if not self.session_id:
            await self.authenticate()
        client_id = f"wake-agent-{uuid.uuid4().hex[:12]}"
        r = await self._req(
            "POST",
            f"/api/sessions/{self.session_id}/calls",
            json={"phone": phone},
            headers={"X-Client-Id": client_id},
            ok=(200,),
        )
        call_id = r.json()["call"]["callId"]
        c = _Call(call_id=call_id, client_id=client_id, status=CallStatus(call_id, CallState.RINGING))
        self._calls[call_id] = c
        event("WHATSAPP_CALL_STARTED", call=call_id[:10], to=mask_phone(phone))
        try:
            await self._attach_audio(c)
        except Exception as e:
            # she can decline within ~1 s, before the audio leg attaches (live: attempt 3, 20:17)
            await self._poll(call_id)
            if c.status.state == CallState.ENDED:
                event("CALL_ENDED_BEFORE_AUDIO", call=call_id[:10], reason=c.status.end_reason)
                return call_id  # WakeCall sees ENDED while ringing and classifies it (declined)
            event("AUDIO_ATTACH_FAILED", call=call_id[:10], error=type(e).__name__)
            await self.hangup(call_id)
            raise WhatsAppError(f"could not attach audio leg: {e}") from e
        return call_id

    async def _attach_audio(self, c: _Call) -> None:
        pc = RTCPeerConnection()
        dc = pc.createDataChannel(PCM_LABEL)  # must exist before createOffer so SDP has m=application
        c.pc, c.dc = pc, dc

        @dc.on("open")
        def _on_open() -> None:
            c.dc_open.set()

        @dc.on("message")
        def _on_message(msg) -> None:
            if not isinstance(msg, (bytes, bytearray)) or not msg:
                return
            c.inbound_frames += 1
            if c.inbound.full():
                try:
                    c.inbound.get_nowait()
                    c.inbound_dropped += 1
                except asyncio.QueueEmpty:
                    pass
            c.inbound.put_nowait(bytes(msg))

        @pc.on("connectionstatechange")
        async def _on_state() -> None:
            if pc.connectionState in ("failed", "closed"):
                log.warning("pcm peer connection %s", pc.connectionState)

        await pc.setLocalDescription(await pc.createOffer())  # aiortc gathers ICE here
        r = await self._req(
            "POST",
            f"/api/sessions/{self.session_id}/calls/{c.call_id}/webrtc",
            json={"sdp_offer": pc.localDescription.sdp},
            headers={"X-Client-Id": c.client_id},
        )
        await pc.setRemoteDescription(RTCSessionDescription(sdp=r.json()["sdp_answer"], type="answer"))
        await asyncio.wait_for(c.dc_open.wait(), 10)

        def _send(chunk: bytes) -> None:
            if dc.readyState == "open":
                dc.send(chunk)

        c.pacer = OutboundPacer(_send, backlog_bytes=lambda: getattr(dc, "bufferedAmount", 0))
        c.pacer.start()
        event("AUDIO_LEG_ATTACHED", call=c.call_id[:10])

    async def get_call_status(self, call_id: str) -> CallStatus:
        c = self._calls.get(call_id)
        if c and c.status.state == CallState.ENDED:
            return c.status
        if not self._sse_up.is_set():
            await self._poll(call_id)
        return self._calls[call_id].status if call_id in self._calls else CallStatus(call_id, CallState.UNKNOWN)

    async def wait_status_change(self, call_id: str, timeout: float) -> CallStatus:
        c = self._calls.get(call_id)
        if not c:
            return CallStatus(call_id, CallState.UNKNOWN)
        c.changed.clear()
        try:
            await asyncio.wait_for(c.changed.wait(), timeout if self._sse_up.is_set() else min(timeout, 2.0))
        except asyncio.TimeoutError:
            if not self._sse_up.is_set():
                await self._poll(call_id)
        return c.status

    async def _poll(self, call_id: str) -> None:
        c = self._calls.get(call_id)
        if not c or not self.session_id:
            return
        try:
            r = await self._http.get(f"/api/sessions/{self.session_id}/calls/{call_id}")
        except httpx.HTTPError:
            return
        if r.status_code == 200:
            self._apply(call_id, r.json().get("status", ""), None)
        elif r.status_code == 404:
            self._apply(call_id, "ended", await self._history_reason(call_id))

    async def _history_reason(self, call_id: str) -> str | None:
        try:
            r = await self._http.get(f"/api/sessions/{self.session_id}/history", params={"limit": 20})
            for rec in r.json().get("calls", []):
                if rec.get("callId") == call_id:
                    return rec.get("endReason")
        except Exception:
            pass
        return None

    def _apply(self, call_id: str, status: str, reason: str | None) -> None:
        c = self._calls.get(call_id)
        if not c:
            return
        new = _map_status(status)
        if c.status.state == CallState.ENDED:
            return
        if new == c.status.state and reason is None:
            return
        c.status.state = new
        if new == CallState.CONNECTED:
            c.status.ever_connected = True
        if new == CallState.ENDED:
            c.status.end_reason = reason or c.status.end_reason or "unknown"
        c.changed.set()

    # ----------------------------------------------------------------- audio
    async def receive_audio(self, call_id: str) -> AsyncIterator[bytes]:
        c = self._calls[call_id]
        while c.status.state != CallState.ENDED:
            try:
                yield await asyncio.wait_for(c.inbound.get(), 0.5)
            except asyncio.TimeoutError:
                continue

    async def send_audio(self, call_id: str, pcm: bytes) -> None:
        c = self._calls.get(call_id)
        if c and c.pacer:
            c.pacer.push(pcm)

    async def flush_audio(self, call_id: str) -> int:
        c = self._calls.get(call_id)
        return c.pacer.flush() if c and c.pacer else 0

    def pause_audio(self, call_id: str) -> None:
        c = self._calls.get(call_id)
        if c and c.pacer:
            c.pacer.pause()

    def resume_audio(self, call_id: str) -> None:
        c = self._calls.get(call_id)
        if c and c.pacer:
            c.pacer.resume()

    def audio_backlog_ms(self, call_id: str) -> int:
        c = self._calls.get(call_id)
        return c.pacer.queued_ms if c and c.pacer else 0

    def audio_sent_ms(self, call_id: str) -> int:
        c = self._calls.get(call_id)
        return c.pacer.sent_ms if c and c.pacer else 0

    def audio_stats(self, call_id: str) -> dict:
        c = self._calls.get(call_id)
        if not c:
            return {}
        return {
            "in_frames": c.inbound_frames,
            "in_dropped": c.inbound_dropped,
            "out_ms": c.pacer.sent_ms if c.pacer else 0,
            "out_dropped_ms": (c.pacer.dropped_bytes // 32) if c.pacer else 0,
        }

    async def wait_audio_drained(self, call_id: str, timeout: float) -> bool:
        c = self._calls.get(call_id)
        return await c.pacer.wait_idle(timeout) if c and c.pacer else True

    # ---------------------------------------------------------------- hangup
    async def hangup(self, call_id: str) -> None:
        try:
            await self._http.delete(f"/api/sessions/{self.session_id}/calls/{call_id}")
        except httpx.HTTPError as e:
            log.warning("hangup request failed: %s", type(e).__name__)
        self._apply(call_id, "ended", "user_ended")
        await self._teardown(call_id)

    async def _teardown(self, call_id: str) -> None:
        c = self._calls.get(call_id)
        if not c:
            return
        if c.pacer:
            await c.pacer.close()
            c.pacer = None
        if c.pc:
            pc, c.pc = c.pc, None
            try:  # aiortc's close can stall on SCTP shutdown; never let teardown hang the run
                await asyncio.wait_for(pc.close(), 5)
            except (asyncio.TimeoutError, Exception):
                log.debug("pcm peer close timed out")

    async def release(self, call_id: str) -> None:
        """Tear down our audio leg for a finished call and drop its state."""
        await self._teardown(call_id)
        self._calls.pop(call_id, None)

    # ------------------------------------------------------------------- SSE
    async def _sse_loop(self) -> None:
        backoff = 1.0
        while True:
            try:
                async with self._http.stream("GET", "/api/events", timeout=httpx.Timeout(15.0, read=60.0)) as r:
                    r.raise_for_status()
                    self._sse_up.set()
                    backoff = 1.0
                    async for line in r.aiter_lines():
                        if line.startswith("data: "):
                            self._on_sse(line[6:])
            except asyncio.CancelledError:
                return
            except Exception as e:
                log.warning("SSE disconnected (%s); reconnecting", type(e).__name__)
            self._sse_up.clear()
            # while SSE was down we may have missed an end event: poll live calls once
            for cid, c in list(self._calls.items()):
                if c.status.state != CallState.ENDED:
                    await self._poll(cid)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 15)

    def _on_sse(self, data: str) -> None:
        try:
            ev = json.loads(data)
        except json.JSONDecodeError:
            return
        t = ev.get("type")
        if t == "call-status":
            self._apply(ev.get("id", ""), ev.get("status", ""), None)
        elif t == "call-ended":
            self._apply(ev.get("id", ""), "ended", ev.get("reason"))
        elif t == "call-list":
            for rec in ev.get("calls") or []:
                self._apply(rec.get("callId", ""), rec.get("status", ""), None)

    # --------------------------------------------------------------- helpers
    async def _req(self, method: str, path: str, ok: tuple[int, ...] = (200, 201, 204), **kw) -> httpx.Response:
        if self._http is None:
            await self.connect()
        try:
            r = await self._http.request(method, path, **kw)
        except httpx.HTTPError as e:
            raise WhatsAppError(f"{method} {path} failed: {type(e).__name__}") from e
        if r.status_code not in ok:
            try:
                msg = r.json().get("error", "")
            except Exception:
                msg = r.text[:200]
            if r.status_code == 401:
                raise WhatsAppError("WaCalls rejected the API token (401)")
            if r.status_code == 503 and "paired" in msg:
                raise NotAuthenticated(msg)
            raise WhatsAppError(f"{method} {path} -> {r.status_code}: {msg}")
        return r
