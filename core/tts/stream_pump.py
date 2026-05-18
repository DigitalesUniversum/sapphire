"""Brain-side streaming TTS pump.

Wraps SpeechChunker + a small ThreadPoolExecutor to synthesize audio chunks
in the background while the LLM is still streaming tokens. Each finished
chunk becomes a `tts_chunk` SSE event. The pump is per-iteration: created
when streaming TTS is enabled+available, fed text from the LLM content
stream, and flushed at end-of-iteration to drain remaining audio.

Boundaries / chunking live in core.tts.streaming.SpeechChunker — this module
just orchestrates synth scheduling and event emission.
"""
import base64
import concurrent.futures
import logging
import uuid
from collections import deque

import config
from core.hooks import hook_runner, HookEvent
from core.tts.streaming import SpeechChunker

logger = logging.getLogger(__name__)


class StreamingTTSPump:
    """Push-based pump: feed it LLM content text via `push(text)`, get back
    a list of SSE event dicts (zero-or-more `tts_chunk`s plus an initial
    `tts_stream_start` on the first push). Call `flush_and_close()` at end
    of stream to drain remaining synth + emit `tts_stream_end`.

    `cancel()` aborts outstanding work and fires `tts_stream_end` with
    `interrupted=True` — used when the user hits Stop.

    Hook surface (fired in this order, see docs/PLUGINS.md):
        tts_stream_start: once per turn, before any synth. metadata has
            `voice`, `speed`, `stream_id`. Plugin may set `skip_tts` to
            disable the whole turn's streaming TTS.
        tts_chunk_text:   once per chunk, before synth. `event.tts_text`
            is the chunk text (mutable). metadata: `chunk_index`,
            `boundary`, `pause_after_ms`, `stream_id`. `skip_tts` skips
            this single chunk.
        tts_chunk_audio:  once per chunk, after synth returns, before
            SSE emission. `event.metadata['audio_bytes']` is mutable so
            plugins can transform or replace. metadata: `chunk_index`,
            `chunk_text`, `content_type`, `stream_id`.
        tts_stream_end:   once per turn. metadata: `chunk_count`,
            `total_chars`, `interrupted`, `stream_id`. Observational.
    """

    def __init__(self, system):
        self.system = system
        self.tts = getattr(system, "tts", None)
        self.provider = getattr(self.tts, "_provider", None) if self.tts else None
        self.chunker = SpeechChunker()
        self.pending: deque = deque()
        self.executor = None
        self._stream_started = False
        self._closed = False
        # Stable id for this turn — plugins correlate events across hooks.
        self._stream_id = uuid.uuid4().hex
        self._chunk_count = 0
        self._total_chars = 0
        # Plugin can disable the whole turn via tts_stream_start skip_tts.
        self._skip_turn = False

    @property
    def enabled(self) -> bool:
        return bool(
            getattr(config, "TTS_ENABLED", False)
            and getattr(config, "TTS_STREAMING_ENABLED", False)
            and self.provider is not None
            and getattr(self.provider, "supports_streaming", False)
        )

    def push(self, text: str) -> list:
        """Push LLM content text; return SSE event dicts to yield."""
        if not self.enabled or not text or self._closed or self._skip_turn:
            return []
        out: list = []
        if not self._stream_started:
            self._stream_started = True
            # Fire tts_stream_start hook — plugin can cancel whole turn.
            ev = self._fire_hook(
                "tts_stream_start",
                metadata={
                    "voice": getattr(self.tts, "voice_name", None),
                    "speed": getattr(self.tts, "speed", None),
                    "stream_id": self._stream_id,
                    "system": self.system,
                },
            )
            if ev and ev.skip_tts:
                logger.info(f"[TTS-STREAM] Plugin cancelled stream {self._stream_id} via tts_stream_start")
                self._skip_turn = True
                self._closed = True
                return []
            out.append({"type": "tts_stream_start", "stream_id": self._stream_id})
            self._executor()  # lazy-create
        for chunk in self.chunker.push(text):
            self._submit(chunk)
        out.extend(self._drain_ready())
        return out

    def flush_and_close(self) -> list:
        """Flush + block-drain remaining synth + emit `tts_stream_end`."""
        if not self._stream_started or self._closed:
            self._close()
            return []
        out: list = []
        for chunk in self.chunker.flush():
            self._submit(chunk)
        # Block until each remaining future resolves, preserving order.
        while self.pending:
            fut, meta = self.pending.popleft()
            audio = self._result_or_none(fut, meta)
            event = self._build_chunk_event(audio, meta)
            if event:
                out.append(event)
        self._fire_hook(
            "tts_stream_end",
            metadata={
                "stream_id": self._stream_id,
                "chunk_count": self._chunk_count,
                "total_chars": self._total_chars,
                "interrupted": False,
                "system": self.system,
            },
        )
        out.append({
            "type": "tts_stream_end",
            "stream_id": self._stream_id,
            "chunk_count": self._chunk_count,
            "interrupted": False,
        })
        self._close()
        return out

    def cancel(self):
        """Drop in-flight synth; fire tts_stream_end(interrupted=True) so
        plugins can finalize state (e.g. close a recording file)."""
        if self._closed:
            return
        for fut, _meta in self.pending:
            fut.cancel()
        self.pending.clear()
        if self._stream_started:
            self._fire_hook(
                "tts_stream_end",
                metadata={
                    "stream_id": self._stream_id,
                    "chunk_count": self._chunk_count,
                    "total_chars": self._total_chars,
                    "interrupted": True,
                    "system": self.system,
                },
            )
        self._close()

    def _executor(self) -> concurrent.futures.ThreadPoolExecutor:
        if self.executor is None:
            # 2 workers = enough to pipeline against LLM token rate; more
            # would just contend on Kokoro's single-process bottleneck.
            self.executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=2,
                thread_name_prefix="tts-stream",
            )
        return self.executor

    def _submit(self, chunk: dict):
        # Fire tts_chunk_text hook BEFORE synth — plugin can mutate text
        # or skip this chunk entirely via skip_tts.
        ev = self._fire_hook(
            "tts_chunk_text",
            tts_text=chunk["text"],
            metadata={
                "stream_id": self._stream_id,
                "chunk_index": chunk["index"],
                "boundary": chunk["boundary"],
                "pause_after_ms": chunk["pause_after_ms"],
                "system": self.system,
            },
        )
        # Pull text back out — may have been mutated.
        text_to_synth = (ev.tts_text if ev else chunk["text"]) or chunk["text"]
        if ev and ev.skip_tts:
            logger.debug(f"[TTS-STREAM] Plugin skipped chunk {chunk['index']} via tts_chunk_text")
            return
        meta = {
            "index": chunk["index"],
            "boundary": chunk["boundary"],
            "pause_after_ms": chunk["pause_after_ms"],
            "text": text_to_synth,
        }
        voice = getattr(self.tts, "voice_name", None) or "af_heart"
        speed = getattr(self.tts, "speed", None) or 1.0
        fut = self._executor().submit(self._synth, text_to_synth, voice, speed)
        self.pending.append((fut, meta))

    def _synth(self, text: str, voice: str, speed: float):
        try:
            return self.provider.generate(text, voice, speed)
        except Exception as e:
            logger.warning(f"[TTS-STREAM] synth raised: {e!r}")
            return None

    def _result_or_none(self, fut, meta):
        try:
            return fut.result(timeout=30)
        except Exception as e:
            logger.warning(f"[TTS-STREAM] synth result failed (chunk {meta.get('index')}): {e!r}")
            return None

    def _drain_ready(self) -> list:
        """Non-blocking: drain in-order futures that have already completed."""
        out: list = []
        while self.pending and self.pending[0][0].done():
            fut, meta = self.pending.popleft()
            audio = self._result_or_none(fut, meta)
            event = self._build_chunk_event(audio, meta)
            if event:
                out.append(event)
        return out

    def _build_chunk_event(self, audio_bytes, meta: dict):
        """Fire tts_chunk_audio hook (mutable bytes), then build the SSE
        event dict. Returns None when audio is empty or a plugin nulls it."""
        if not audio_bytes:
            return None
        content_type = getattr(self.provider, "audio_content_type", "audio/ogg")
        # Hook event uses a dict carrier so plugins can mutate or replace.
        carrier = {"audio_bytes": audio_bytes, "content_type": content_type}
        self._fire_hook(
            "tts_chunk_audio",
            metadata={
                "stream_id": self._stream_id,
                "chunk_index": meta["index"],
                "chunk_text": meta["text"],
                "boundary": meta["boundary"],
                "pause_after_ms": meta["pause_after_ms"],
                "audio": carrier,
                "system": self.system,
            },
        )
        final_bytes = carrier.get("audio_bytes")
        final_ct = carrier.get("content_type") or content_type
        if not final_bytes:
            return None
        self._chunk_count += 1
        self._total_chars += len(meta["text"] or "")
        return {
            "type": "tts_chunk",
            "audio_b64": base64.b64encode(final_bytes).decode("ascii"),
            "content_type": final_ct,
            "index": meta["index"],
            "boundary": meta["boundary"],
            "pause_after_ms": meta["pause_after_ms"],
            "text": meta["text"],
            "stream_id": self._stream_id,
        }

    def _fire_hook(self, hook_name: str, tts_text: str = None, metadata: dict = None):
        """Fire a hook if any handlers exist. Returns the (possibly mutated)
        event for callers to read tts_text / skip_tts back, or None if no
        handlers were registered (cheap no-op path)."""
        if not hook_runner.has_handlers(hook_name):
            return None
        ev = HookEvent(
            tts_text=tts_text,
            config=config,
            metadata=metadata or {},
        )
        try:
            hook_runner.fire(hook_name, ev)
        except Exception as e:
            logger.warning(f"[TTS-STREAM] {hook_name} hook fire failed: {e!r}")
        return ev

    def _close(self):
        if self._closed:
            return
        self._closed = True
        if self.executor is not None:
            self.executor.shutdown(wait=False)
            self.executor = None
