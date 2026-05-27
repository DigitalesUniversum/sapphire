"""Mjolnir scenarios — the shared spec all three lanes execute.

Lane A (mocked Audio) and Lane B (real browser) consume the same scenario
definitions. Lane C (backend hammer) has its own subset since it doesn't
touch the playback path.

The three universal invariants — checked by every relevant scenario:
  1. CONCURRENT_PLAYBACK: at any T, at most one Audio is `paused === false`.
     Catches 10× overlap.
  2. NO_SKIPS: total enqueued chunks == total chunks that reached play() OR
     were intentionally dropped (counted in dropped events).
  3. ORDER_PRESERVED: chunks play in index order (no chunk N+1 starts before N).

Scenario-specific asserts layer on top of those.
"""
from dataclasses import dataclass, field
from typing import Optional


# ─── Type definitions ─────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """One synthetic tts_chunk SSE event to inject."""
    delay_ms: int             # delay AFTER previous chunk (or stream_start)
    text: str                 # human-readable label
    index: int                # SSE chunk index
    boundary: str = "sentence"
    pause_after_ms: int = 0
    # Audio payload (base64). Default is a tiny valid OGG marker.
    # Lane A's mock doesn't care about content; Lane B uses real audio
    # from the live backend, not these payloads.
    audio_b64: str = "T0dHU19GQUtFX1NFRw=="  # b'OGGS_FAKE_SEG' base64


@dataclass
class UserAction:
    """A simulated user action to fire mid-stream."""
    delay_ms: int             # after stream_start
    action: str               # "stop" | "replay" | "newstream"


@dataclass
class MockConfig:
    """Lane A — controls the mocked Audio constructor's behavior."""
    play_latency_ms: int = 10       # how long play() takes to resolve
    audio_duration_ms: int = 100    # how long audio "plays" before onended fires
    autoplay_blocked: bool = False  # first play() rejects with NotAllowedError
    play_rejects_with: Optional[str] = None  # "AbortError" | "NotAllowedError" | None — every play() rejects
    onended_never_fires: bool = False        # audio plays forever, no end event
    onerror_after_ms: Optional[int] = None   # fire onerror after N ms instead


@dataclass
class CdpThrottle:
    """Lane B — CDP-driven CPU/network throttling. Chromium-family only."""
    cpu_rate: float = 1.0           # 1=normal, 4=mid laptop, 8=slow, 20=slideshow
    network_profile: Optional[str] = None  # "slow-3g" | "fast-3g" | None


@dataclass
class Scenario:
    name: str
    description: str
    chunks: list                = field(default_factory=list)
    user_actions: list          = field(default_factory=list)
    mock_config: MockConfig     = field(default_factory=MockConfig)
    cdp_throttle: CdpThrottle   = field(default_factory=CdpThrottle)
    # Which lanes this scenario applies to (subset of {"a", "b", "c"}).
    lanes: set                  = field(default_factory=lambda: {"a", "b"})
    # Expected invariant outcomes — most scenarios expect ALL three.
    expects_overlap: bool       = False
    expects_skips: bool         = False
    expects_order_break: bool   = False
    # Lane C-specific
    concurrency: int            = 1   # parallel POSTs to /api/tts/stream
    request_text: str           = ""   # text body for each request
    request_count: int          = 1


# ─── Helper: standard chunk sequences ─────────────────────────────────────────

def _seq(count: int, delay_ms: int, prefix: str = "chunk") -> list:
    """Build a list of `count` chunks each separated by `delay_ms`."""
    return [Chunk(delay_ms=delay_ms, text=f"{prefix}{i}", index=i)
            for i in range(count)]


# ─── Scenarios ────────────────────────────────────────────────────────────────

SCENARIOS = [
    # ─── Cadence variations (Lane A + B) ─────────────────────────────────────
    Scenario(
        name="uniform-fast",
        description="8 chunks 50ms apart, fast play resolution (dev rig pattern)",
        chunks=_seq(8, 50),
        mock_config=MockConfig(play_latency_ms=5, audio_duration_ms=50),
    ),
    Scenario(
        name="uniform-slow",
        description="8 chunks 800ms apart, slow play (CPU-Kokoro user pattern)",
        chunks=_seq(8, 800),
        mock_config=MockConfig(play_latency_ms=200, audio_duration_ms=2000),
    ),
    Scenario(
        name="bursty",
        description="4 chunks @ 30ms, 2s gap, 4 more @ 30ms — slow-CPU stall + restart",
        chunks=(
            [Chunk(delay_ms=30, text=f"c{i}", index=i) for i in range(4)]
            + [Chunk(delay_ms=2000, text="c4", index=4)]
            + [Chunk(delay_ms=30, text=f"c{i}", index=i) for i in range(5, 8)]
        ),
        mock_config=MockConfig(play_latency_ms=10, audio_duration_ms=200),
    ),
    Scenario(
        name="slow-start-fast-tail",
        description="First chunk takes 3s to deliver, rest 50ms (Kokoro cold-start pattern)",
        chunks=(
            [Chunk(delay_ms=3000, text="c0", index=0)]
            + [Chunk(delay_ms=50, text=f"c{i}", index=i) for i in range(1, 8)]
        ),
        mock_config=MockConfig(play_latency_ms=10, audio_duration_ms=100),
    ),
    Scenario(
        name="single-chunk",
        description="One small chunk — the user's exact 'Testing, testing' repro path",
        chunks=[Chunk(delay_ms=10, text="single", index=0, boundary="end")],
        mock_config=MockConfig(play_latency_ms=10, audio_duration_ms=100),
    ),

    # ─── Play-resolution timing (Brave/Linux PipeWire jitter sim) ────────────
    Scenario(
        name="slow-play-resolution",
        description="play() takes 1.5s to resolve — Brave Shields/PipeWire jitter analog",
        chunks=_seq(5, 100),
        mock_config=MockConfig(play_latency_ms=1500, audio_duration_ms=500),
    ),
    Scenario(
        name="very-slow-play-resolution",
        description="play() takes 3s, chunks arrive fast — extreme slowness",
        chunks=_seq(5, 100),
        mock_config=MockConfig(play_latency_ms=3000, audio_duration_ms=500),
    ),

    # ─── Failure-mode scenarios ──────────────────────────────────────────────
    Scenario(
        name="autoplay-blocked-first",
        description="First chunk's play() rejects with autoplay error (no user gesture)",
        chunks=_seq(5, 100),
        mock_config=MockConfig(play_latency_ms=10, audio_duration_ms=100,
                                autoplay_blocked=True),
        # Expected: all chunks fail with NotAllowedError, no audio plays, no overlap
        expects_skips=True,
    ),
    Scenario(
        name="autoplay-blocked-every-chunk",
        description=(
            "REPRODUCES the user's no-audio bug: every play() rejects with "
            "NotAllowedError. Real Chrome MEI=0 + no-gesture scenario. Each "
            "new Audio element is locked, transient activation has expired by "
            "the time the first chunk arrives (LLM latency >5s), so every "
            "chunk's play() rejects. Audio.js silently advances → silent failure."
        ),
        chunks=_seq(5, 100),
        mock_config=MockConfig(play_latency_ms=10, audio_duration_ms=100,
                                play_rejects_with="NotAllowedError"),
        expects_skips=True,
    ),
    Scenario(
        name="abort-cascade",
        description="Every play() rejects with AbortError — original no-audio bug class",
        chunks=_seq(5, 100),
        mock_config=MockConfig(play_latency_ms=10, audio_duration_ms=100,
                                play_rejects_with="AbortError"),
        expects_skips=True,
    ),
    Scenario(
        name="onended-never-fires",
        description="Audio plays but onended event never fires — stream wedges",
        chunks=_seq(3, 100),
        mock_config=MockConfig(play_latency_ms=10, audio_duration_ms=500,
                                onended_never_fires=True),
        # First chunk should play, rest stuck in queue. No overlap.
    ),
    Scenario(
        name="onerror-mid-stream",
        description="Chunk 2's audio fires onerror mid-playback — queue must drain",
        chunks=_seq(5, 100),
        mock_config=MockConfig(play_latency_ms=10, audio_duration_ms=2000,
                                onerror_after_ms=300),
    ),

    # ─── User-action sequences (preempt/stop/replay) ─────────────────────────
    Scenario(
        name="stop-mid-stream",
        description="Stop button pressed after 3 chunks are queued",
        chunks=_seq(8, 100),
        user_actions=[UserAction(delay_ms=350, action="stop")],
        mock_config=MockConfig(play_latency_ms=20, audio_duration_ms=300),
    ),
    Scenario(
        name="stop-before-first-chunk",
        description="Stop fires before first chunk's play() resolves",
        chunks=_seq(5, 200),
        user_actions=[UserAction(delay_ms=50, action="stop")],
        mock_config=MockConfig(play_latency_ms=500, audio_duration_ms=500),
    ),
    Scenario(
        name="replay-during-stream",
        description="User clicks Replay while stream is playing — preempt + new stream",
        chunks=_seq(8, 100),
        user_actions=[UserAction(delay_ms=350, action="replay")],
        mock_config=MockConfig(play_latency_ms=10, audio_duration_ms=300),
    ),
    Scenario(
        name="rapid-replay-spam",
        description="Replay clicked 3x in rapid succession — stress preempt logic",
        chunks=_seq(5, 100),
        user_actions=[
            UserAction(delay_ms=200, action="replay"),
            UserAction(delay_ms=300, action="replay"),
            UserAction(delay_ms=400, action="replay"),
        ],
        mock_config=MockConfig(play_latency_ms=50, audio_duration_ms=300),
    ),

    # ─── CDP throttle scenarios (Lane B only — chromium family) ──────────────
    Scenario(
        name="cpu-throttle-4x",
        description="Real Audio on 4x CPU throttle — mid-tier laptop simulation",
        chunks=_seq(5, 100),
        cdp_throttle=CdpThrottle(cpu_rate=4.0),
        lanes={"b"},
    ),
    Scenario(
        name="cpu-throttle-8x",
        description="Real Audio on 8x CPU throttle — old hardware simulation",
        chunks=_seq(5, 200),
        cdp_throttle=CdpThrottle(cpu_rate=8.0),
        lanes={"b"},
    ),

    # ─── Lane C: backend hammer scenarios ────────────────────────────────────
    Scenario(
        name="backend-single-short",
        description="One request, short text — baseline smoke",
        lanes={"c"},
        concurrency=1,
        request_count=1,
        request_text="This is a short test sentence to stream.",
    ),
    Scenario(
        name="backend-concurrent-3",
        description="3 concurrent /api/tts/stream requests — pump under parallel load",
        lanes={"c"},
        concurrency=3,
        request_count=3,
        request_text=(
            "This is a longer test paragraph. It contains multiple sentences. "
            "The streaming TTS pump should handle each sentence as its own chunk. "
            "Concurrent requests stress the executor and the kokoro pipeline lock."
        ),
    ),
    Scenario(
        name="backend-concurrent-10",
        description="10 concurrent requests — find the breaking point",
        lanes={"c"},
        concurrency=10,
        request_count=10,
        request_text="Stress test sentence number one. Sentence two follows. Three.",
    ),
    Scenario(
        name="backend-cjk-content",
        description="CJK text — should drop with friendly notice, not crash",
        lanes={"c"},
        concurrency=1,
        request_count=1,
        request_text="心 心 心 心 心 心 心 心 心 心.",
    ),
    Scenario(
        name="backend-emoji-only",
        description="Emoji-only text — should be filtered before synth",
        lanes={"c"},
        concurrency=1,
        request_count=1,
        request_text="🎉 ✨ 🌊 🔥 🎵 ⭐ 🚀 😀 💫 🌟.",
    ),
    Scenario(
        name="backend-empty-text",
        description="Empty text body — should 400, no crash",
        lanes={"c"},
        concurrency=1,
        request_count=1,
        request_text="",
    ),
]


def scenarios_for_lane(lane: str) -> list:
    """Filter scenarios that apply to a given lane."""
    return [s for s in SCENARIOS if lane in s.lanes]
