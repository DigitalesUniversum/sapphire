"""Lane B — real-browser playback with CDP throttling.

Same scenarios as Lane A but uses the REAL window.Audio constructor (wrapped
for observation, not replaced). Audio_b64 payloads are a tiny real silent
OGG/Opus blob the browser can actually decode. CDP CPU throttling per
scenario simulates slower hardware.

Tests:
- Real autoplay policy + first-user-interaction behavior
- Real audio decoder pipeline timing
- CPU-bound state machine performance
- Brave/Chromium media-stack quirks (if Brave binary available)

Doesn't hit Sapphire backend — uses the same isolated test page as Lane A.
Backend integration is Lane C's job.
"""
import asyncio
import base64
import dataclasses
import os
import shutil
from pathlib import Path
from typing import Optional

from playwright.async_api import async_playwright

from . import scenarios
from .lane_a import _start_static_server, _check_invariants, PROJECT_ROOT


MJOLNIR_DIR = Path(__file__).parent
SILENT_OGG_PATH = MJOLNIR_DIR / "silent.ogg"


def _silent_audio_b64() -> str:
    """Load the pre-generated silent OGG/Opus and base64-encode."""
    return base64.b64encode(SILENT_OGG_PATH.read_bytes()).decode("ascii")


def _detect_brave_binary() -> Optional[str]:
    """Look for a Brave executable in common locations. None if not found."""
    env_override = os.environ.get("BRAVE_PATH")
    if env_override and Path(env_override).exists():
        return env_override
    candidates = [
        "/usr/bin/brave-browser",
        "/usr/bin/brave",
        "/snap/bin/brave",
        "/opt/brave.com/brave/brave-browser",
        shutil.which("brave-browser") or "",
        shutil.which("brave") or "",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    return None


async def _apply_cdp_throttle(context, page, throttle: scenarios.CdpThrottle):
    """Apply CPU + network throttling via Chromium DevTools Protocol.
    Returns the CDP session so the caller can clear throttling later."""
    cdp = await context.new_cdp_session(page)
    if throttle.cpu_rate > 1.0:
        await cdp.send("Emulation.setCPUThrottlingRate",
                        {"rate": float(throttle.cpu_rate)})
    if throttle.network_profile:
        profiles = {
            "slow-3g": {"offline": False, "downloadThroughput": 50 * 1024,
                         "uploadThroughput": 50 * 1024, "latency": 500},
            "fast-3g": {"offline": False, "downloadThroughput": 1.6 * 1024 * 1024 / 8,
                         "uploadThroughput": 750 * 1024 / 8, "latency": 150},
        }
        prof = profiles.get(throttle.network_profile)
        if prof:
            await cdp.send("Network.emulateNetworkConditions", prof)
    return cdp


async def _run_scenario_on_page(page, scenario: scenarios.Scenario,
                                  audio_b64: str) -> dict:
    """Inject scenario, dispatch chunks, wait, snapshot."""
    await page.evaluate("() => window.MjolnirObserver.reset()")

    stream_id = f"mjolnir-b-{scenario.name}"
    await page.evaluate(
        "(data) => window._busDispatch('tts_stream_start', data)",
        {"type": "tts_stream_start", "stream_id": stream_id},
    )

    # Build timeline same as Lane A
    timeline = []
    elapsed = 0
    for chunk in scenario.chunks:
        elapsed += chunk.delay_ms
        timeline.append({"at": elapsed, "kind": "chunk",
                         "data": dataclasses.asdict(chunk)})
    for action in scenario.user_actions:
        timeline.append({"at": action.delay_ms, "kind": "action",
                         "data": dataclasses.asdict(action)})
    timeline.sort(key=lambda e: e["at"])

    last = 0
    for entry in timeline:
        delta = entry["at"] - last
        if delta > 0:
            await asyncio.sleep(delta / 1000.0)
        last = entry["at"]

        if entry["kind"] == "chunk":
            chunk = entry["data"]
            await page.evaluate(
                """(args) => {
                    const before = window.MjolnirObserver.instances.length;
                    window._busDispatch('tts_stream_chunk', {
                        type: 'tts_chunk',
                        audio_b64: args.audio_b64,
                        content_type: 'audio/ogg',
                        index: args.index,
                        boundary: args.boundary,
                        pause_after_ms: args.pause_after_ms,
                        stream_id: args.stream_id,
                        text: args.text,
                    });
                    const after = window.MjolnirObserver.instances.length;
                    if (after > before) {
                        window.MjolnirObserver.instances[after - 1]._chunkIndex = args.index;
                    }
                }""",
                {**chunk, "audio_b64": audio_b64, "stream_id": stream_id},
            )
        else:
            action = entry["data"]["action"]
            if action == "stop":
                await page.evaluate("() => window._audio.stop(true)")
            elif action in ("replay", "newstream"):
                await page.evaluate(
                    "(sid) => window._busDispatch('tts_stream_start', "
                    "{type:'tts_stream_start', stream_id: sid})",
                    f"{stream_id}-{action}-{entry['at']}",
                )

    # Real audio takes longer than the mock — wait long enough for all
    # chunks to play through. Silent.ogg is 300ms; pad generously.
    settle = max(2000, 500 * max(len(scenario.chunks), 1))
    await asyncio.sleep(settle / 1000.0)

    await page.evaluate(
        "(data) => window._busDispatch('tts_stream_end', data)",
        {
            "type": "tts_stream_end",
            "stream_id": stream_id,
            "chunk_count": len(scenario.chunks),
            "interrupted": False,
        },
    )
    await asyncio.sleep(0.3)

    return await page.evaluate("() => window.MjolnirObserver.snapshot()")


async def _run_browser(p, browser_kind: str, browser_label: str,
                       page_url: str, observer_src: str,
                       audio_b64: str, only: Optional[str],
                       no_autoplay_bypass: bool = False) -> list:
    """Run all Lane B scenarios on one browser.

    When `no_autoplay_bypass` is True, browsers launch WITHOUT the
    `--autoplay-policy=no-user-gesture-required` flag (and equivalent
    Firefox prefs). That puts the test in the REAL autoplay-policy
    environment — used to reproduce the MEI / transient-activation
    bug class that the bypass flag normally hides.
    """
    results = []

    # Chromium-family autoplay args (only added when not bypassing)
    chromium_autoplay_args = (
        [] if no_autoplay_bypass
        else ["--autoplay-policy=no-user-gesture-required"]
    )

    if browser_kind == "brave":
        brave_path = _detect_brave_binary()
        if not brave_path:
            print(f"  [{browser_label}] skipped — brave binary not found "
                  f"(set BRAVE_PATH env var to override)")
            return results
        # Use chromium with Brave's executable
        browser = await p.chromium.launch(
            headless=True, executable_path=brave_path,
            args=["--no-sandbox", *chromium_autoplay_args],
        )
    elif browser_kind == "chromium":
        browser = await p.chromium.launch(
            headless=True,
            args=chromium_autoplay_args,
        )
    elif browser_kind == "firefox":
        # Firefox needs autoplay overrides via prefs. When bypass is on,
        # set media.autoplay.default=0 (allowed); when bypass is OFF, leave
        # the browser default in place (block all autoplay).
        ff_prefs = {} if no_autoplay_bypass else {
            "media.autoplay.default": 0,
            "media.autoplay.blocking_policy": 0,
        }
        browser = await p.firefox.launch(
            headless=True,
            firefox_user_prefs=ff_prefs,
        )
    else:
        print(f"  [{browser_label}] unknown browser_kind: {browser_kind}")
        return results

    try:
        context = await browser.new_context()
        await context.add_init_script(observer_src)
        page = await context.new_page()
        page.on("pageerror", lambda exc: print(f"    [{browser_label} page-error] {exc}"))

        await page.goto(page_url, wait_until="networkidle")
        await page.wait_for_function("() => window._mjolnirReady === true",
                                       timeout=10_000)

        for scenario in scenarios.scenarios_for_lane("b"):
            if only and scenario.name != only:
                continue
            print(f"  [{browser_label}] ► {scenario.name} "
                  f"(cpu_throttle={scenario.cdp_throttle.cpu_rate}x)")

            # Apply per-scenario CDP throttling (Chromium-family only)
            cdp = None
            if browser_kind in ("chromium", "brave") and (
                scenario.cdp_throttle.cpu_rate > 1.0
                or scenario.cdp_throttle.network_profile
            ):
                cdp = await _apply_cdp_throttle(context, page, scenario.cdp_throttle)

            try:
                snapshot = await _run_scenario_on_page(page, scenario, audio_b64)
                invariants = _check_invariants(scenario, snapshot)
            except Exception as e:
                results.append({
                    "scenario": scenario.name, "lane": "b", "browser": browser_label,
                    "ok": False, "error": f"{type(e).__name__}: {e}", "invariants": {},
                })
                print(f"    ✗ exception: {type(e).__name__}: {e}")
                if cdp:
                    try:
                        await cdp.send("Emulation.setCPUThrottlingRate", {"rate": 1})
                    except Exception:
                        pass
                continue

            # Clear throttling between scenarios so page recovers
            if cdp:
                try:
                    await cdp.send("Emulation.setCPUThrottlingRate", {"rate": 1})
                except Exception:
                    pass

            all_pass = all(p for p, _ in invariants.values())
            results.append({
                "scenario": scenario.name, "lane": "b", "browser": browser_label,
                "ok": all_pass,
                "invariants": {k: {"ok": v[0], "msg": v[1]} for k, v in invariants.items()},
                "max_concurrent": snapshot["maxConcurrentPlaying"],
                "instances": snapshot["instanceCount"],
                "attempted": snapshot["attemptedIndexes"],
                "played": snapshot["playedIndexes"],
                "aborts": snapshot["abortErrorCount"],
            })
            for name, (passed, msg) in invariants.items():
                sym = "✓" if passed else "✗"
                print(f"    {sym} {name}: {msg}")
    finally:
        await browser.close()
    return results


async def run_lane_b(only: Optional[str] = None,
                      browsers: Optional[list] = None,
                      no_autoplay_bypass: bool = False) -> list:
    """Run Lane B across requested browsers. Default: chromium + firefox + brave (if found).

    no_autoplay_bypass=True puts the browsers in real-world autoplay-policy
    mode (MEI gate, transient-activation rules). Used to reproduce the
    autoplay bug class.
    """
    if browsers is None:
        browsers = ["chromium", "firefox", "brave"]

    port, shutdown_server = _start_static_server()
    page_url = f"http://127.0.0.1:{port}/tools/mjolnir/page.html"
    observer_src = (Path(__file__).parent / "audio_real_observer.js").read_text()
    audio_b64 = _silent_audio_b64()

    if no_autoplay_bypass:
        print("\n[Lane B] NO-AUTOPLAY-BYPASS mode — testing real MEI/gesture rules")

    all_results = []
    try:
        async with async_playwright() as p:
            for b in browsers:
                label = b
                if b == "brave":
                    if not _detect_brave_binary():
                        print(f"\n=== {label}: SKIPPED (binary not found) ===")
                        continue
                print(f"\n=== Lane B / {label} ===")
                all_results.extend(
                    await _run_browser(p, b, label, page_url,
                                        observer_src, audio_b64, only,
                                        no_autoplay_bypass=no_autoplay_bypass)
                )
    finally:
        shutdown_server()
    return all_results


if __name__ == "__main__":
    asyncio.run(run_lane_b())
