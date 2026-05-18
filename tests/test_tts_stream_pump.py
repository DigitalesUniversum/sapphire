"""Tests for the brain-side streaming TTS pump.

Focus: the orchestration glue around SpeechChunker — that push() yields
the right SSE events, that disabled/no-provider/no-stream-support all
correctly fall back to no-op, and that cancel/close don't leak threads.

Synth is mocked — the chunker itself is tested separately in
tests/test_tts_streaming.py.
"""
import base64
import time
from unittest.mock import MagicMock

import pytest

from core.tts.stream_pump import StreamingTTSPump


class _FakeProvider:
    audio_content_type = "audio/ogg"
    supports_streaming = True

    def __init__(self, audio_bytes=b"OGGS\x00fake", delay=0.0, fail=False):
        self.audio_bytes = audio_bytes
        self.delay = delay
        self.fail = fail
        self.calls = []

    def generate(self, text, voice, speed):
        self.calls.append((text, voice, speed))
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("simulated synth failure")
        return self.audio_bytes


def _make_system(provider=None, voice="af_heart", speed=1.0, tts_attr=True):
    sys = MagicMock()
    if tts_attr:
        sys.tts._provider = provider
        sys.tts.voice_name = voice
        sys.tts.speed = speed
    else:
        sys.tts = None
    return sys


@pytest.fixture
def enable_streaming(monkeypatch):
    import config
    monkeypatch.setattr(config, "TTS_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "TTS_STREAMING_ENABLED", True, raising=False)


def test_disabled_when_streaming_setting_off(monkeypatch):
    import config
    monkeypatch.setattr(config, "TTS_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "TTS_STREAMING_ENABLED", False, raising=False)
    pump = StreamingTTSPump(system=_make_system(provider=_FakeProvider()))
    assert pump.enabled is False
    assert pump.push("Hello. World. ") == []
    assert pump.flush_and_close() == []


def test_disabled_when_no_provider(enable_streaming):
    pump = StreamingTTSPump(system=_make_system(tts_attr=False))
    assert pump.enabled is False


def test_disabled_when_provider_no_stream_support(enable_streaming):
    class NoStream:
        supports_streaming = False
        def generate(self, *a, **kw): return b""
    pump = StreamingTTSPump(system=_make_system(provider=NoStream()))
    assert pump.enabled is False


def test_first_push_emits_stream_start(enable_streaming):
    pump = StreamingTTSPump(system=_make_system(provider=_FakeProvider()))
    out = pump.push("Just a short bit")  # no boundary yet
    types = [e["type"] for e in out]
    assert "tts_stream_start" in types
    pump.cancel()


def test_complete_sentence_produces_audio_event(enable_streaming):
    fp = _FakeProvider(audio_bytes=b"FAKE_OGG_BYTES")
    pump = StreamingTTSPump(system=_make_system(provider=fp))
    # Push enough to satisfy sentence boundary (period + space + capital)
    pump.push("Hello there. ")
    pump.push("More text here.")
    out = pump.flush_and_close()
    # Expect at least one tts_chunk and a final tts_stream_end
    types = [e["type"] for e in out]
    assert "tts_stream_end" in types
    chunks = [e for e in out if e["type"] == "tts_chunk"]
    assert len(chunks) >= 1
    # Audio is base64-encoded
    decoded = base64.b64decode(chunks[0]["audio_b64"])
    assert decoded == b"FAKE_OGG_BYTES"
    # Metadata round-trips
    assert chunks[0]["content_type"] == "audio/ogg"
    assert "boundary" in chunks[0]
    assert "pause_after_ms" in chunks[0]


def test_synth_failure_swallowed_no_event(enable_streaming):
    fp = _FakeProvider(fail=True)
    pump = StreamingTTSPump(system=_make_system(provider=fp))
    pump.push("Hello there. ")
    pump.push("More.")
    out = pump.flush_and_close()
    chunks = [e for e in out if e["type"] == "tts_chunk"]
    assert chunks == []  # all synths failed
    assert any(e["type"] == "tts_stream_end" for e in out)


def test_cancel_no_more_events(enable_streaming):
    fp = _FakeProvider(delay=0.2)
    pump = StreamingTTSPump(system=_make_system(provider=fp))
    pump.push("First sentence here. ")
    pump.push("Second sentence here. ")
    pump.push("Third.")
    # Cancel without waiting
    pump.cancel()
    # Subsequent push should be no-op
    assert pump.push("more") == []
    # flush_and_close after cancel is no-op
    assert pump.flush_and_close() == []


def test_push_with_no_text_no_events(enable_streaming):
    pump = StreamingTTSPump(system=_make_system(provider=_FakeProvider()))
    assert pump.push("") == []
    assert pump.push(None) == []


def test_flush_emits_end_only_when_started(enable_streaming):
    pump = StreamingTTSPump(system=_make_system(provider=_FakeProvider()))
    # No push happened
    out = pump.flush_and_close()
    assert out == []  # no stream_start means no stream_end either


# ---------------------------------------------------------------------------
# Hook surface tests (M5)
# ---------------------------------------------------------------------------


@pytest.fixture
def fresh_hooks():
    """Each hook test starts with a clean hook registry and restores after."""
    from core.hooks import hook_runner
    snapshot = dict(hook_runner._hooks)
    hook_runner._hooks.clear()
    hook_runner._sorted.clear()
    yield hook_runner
    hook_runner._hooks.clear()
    hook_runner._sorted.clear()
    hook_runner._hooks.update(snapshot)


def test_tts_stream_start_hook_fires(enable_streaming, fresh_hooks):
    seen = []
    fresh_hooks.register("tts_stream_start", lambda ev: seen.append(ev.metadata.copy()), plugin_name="t")
    pump = StreamingTTSPump(system=_make_system(provider=_FakeProvider()))
    pump.push("Anything.")
    assert len(seen) == 1
    assert "stream_id" in seen[0]
    assert seen[0]["voice"] == "af_heart"
    pump.cancel()


def test_tts_stream_start_skip_disables_turn(enable_streaming, fresh_hooks):
    def cancel(ev):
        ev.skip_tts = True
    fresh_hooks.register("tts_stream_start", cancel, plugin_name="killer")
    pump = StreamingTTSPump(system=_make_system(provider=_FakeProvider()))
    out = pump.push("Hello there. ")
    out += pump.push("More text here.")
    out += pump.flush_and_close()
    # Skip means no events emitted at all (not even stream_start in output)
    assert out == []


def test_tts_chunk_text_hook_can_mutate(enable_streaming, fresh_hooks):
    fp = _FakeProvider()
    def mutate(ev):
        ev.tts_text = (ev.tts_text or "").upper()
    fresh_hooks.register("tts_chunk_text", mutate, plugin_name="up")
    pump = StreamingTTSPump(system=_make_system(provider=fp))
    pump.push("Hello there. ")
    pump.push("More text.")
    pump.flush_and_close()
    # Provider should have been called with uppercased text
    assert any(call[0].startswith("HELLO") for call in fp.calls), fp.calls


def test_tts_chunk_text_skip_drops_chunk(enable_streaming, fresh_hooks):
    fp = _FakeProvider()
    def drop_first(ev):
        if ev.metadata.get("chunk_index") == 0:
            ev.skip_tts = True
    fresh_hooks.register("tts_chunk_text", drop_first, plugin_name="dropper")
    pump = StreamingTTSPump(system=_make_system(provider=fp))
    pump.push("First sentence here. ")
    pump.push("Second sentence here. ")
    pump.push("Third.")
    out = pump.flush_and_close()
    # First chunk was skipped — provider was NOT called for it
    chunk_texts = [c[0] for c in fp.calls]
    assert not any("First sentence" in t for t in chunk_texts)
    # Other chunks still synthesized
    assert any("Second sentence" in t for t in chunk_texts)


def test_tts_chunk_audio_hook_can_replace_bytes(enable_streaming, fresh_hooks):
    fp = _FakeProvider(audio_bytes=b"ORIGINAL")
    def replace(ev):
        carrier = ev.metadata.get("audio")
        if carrier is not None:
            carrier["audio_bytes"] = b"REPLACED"
    fresh_hooks.register("tts_chunk_audio", replace, plugin_name="swapper")
    pump = StreamingTTSPump(system=_make_system(provider=fp))
    pump.push("Hello there. ")
    pump.push("More.")
    out = pump.flush_and_close()
    chunks = [e for e in out if e["type"] == "tts_chunk"]
    assert chunks
    assert base64.b64decode(chunks[0]["audio_b64"]) == b"REPLACED"


def test_tts_chunk_audio_hook_can_drop(enable_streaming, fresh_hooks):
    fp = _FakeProvider(audio_bytes=b"ORIGINAL")
    def drop(ev):
        carrier = ev.metadata.get("audio")
        if carrier is not None:
            carrier["audio_bytes"] = None
    fresh_hooks.register("tts_chunk_audio", drop, plugin_name="muter")
    pump = StreamingTTSPump(system=_make_system(provider=fp))
    pump.push("Hello there. ")
    pump.push("More.")
    out = pump.flush_and_close()
    chunks = [e for e in out if e["type"] == "tts_chunk"]
    assert chunks == []  # all chunks muted


def test_tts_stream_end_hook_observes_completion(enable_streaming, fresh_hooks):
    seen = []
    fresh_hooks.register("tts_stream_end", lambda ev: seen.append(ev.metadata.copy()), plugin_name="t")
    pump = StreamingTTSPump(system=_make_system(provider=_FakeProvider()))
    pump.push("Hello there. ")
    pump.push("More.")
    pump.flush_and_close()
    assert len(seen) == 1
    assert seen[0]["interrupted"] is False
    assert seen[0]["chunk_count"] >= 1
    assert "stream_id" in seen[0]


def test_tts_stream_end_hook_fires_on_cancel_with_interrupted_true(enable_streaming, fresh_hooks):
    seen = []
    fresh_hooks.register("tts_stream_end", lambda ev: seen.append(ev.metadata.copy()), plugin_name="t")
    pump = StreamingTTSPump(system=_make_system(provider=_FakeProvider(delay=0.1)))
    pump.push("Hello there. ")
    pump.push("More.")
    pump.cancel()
    assert len(seen) == 1
    assert seen[0]["interrupted"] is True


def test_stream_id_stable_across_hooks(enable_streaming, fresh_hooks):
    """A plugin can correlate the 4 hooks via stream_id within one turn."""
    ids = {}
    fresh_hooks.register("tts_stream_start", lambda ev: ids.setdefault("start", ev.metadata["stream_id"]), plugin_name="t")
    fresh_hooks.register("tts_chunk_text", lambda ev: ids.setdefault("text", ev.metadata["stream_id"]), plugin_name="t")
    fresh_hooks.register("tts_chunk_audio", lambda ev: ids.setdefault("audio", ev.metadata["stream_id"]), plugin_name="t")
    fresh_hooks.register("tts_stream_end", lambda ev: ids.setdefault("end", ev.metadata["stream_id"]), plugin_name="t")
    pump = StreamingTTSPump(system=_make_system(provider=_FakeProvider()))
    pump.push("Hello there. ")
    pump.push("More.")
    pump.flush_and_close()
    assert ids["start"] == ids["text"] == ids["audio"] == ids["end"]
