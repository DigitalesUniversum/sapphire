// settings-tabs/stt.js - Speech-to-text provider settings + VAD tuning
import { renderProviderTab, attachProviderListeners, mergeRegistryProviders } from '../../shared/provider-selector.js';

let _mergedConfig = null;

const tabConfig = {
    providerKey: 'STT_PROVIDER',
    disabledMessage: 'Speech-to-text is disabled. Select a provider above to enable voice input.',

    providers: {
        none: {
            label: 'Disabled',
            essentialKeys: [],
            advancedKeys: []
        },
        faster_whisper: {
            label: 'Local (Faster Whisper)',
            essentialKeys: ['STT_MODEL_SIZE'],
            advancedKeys: [
                'FASTER_WHISPER_DEVICE', 'FASTER_WHISPER_CUDA_DEVICE', 'FASTER_WHISPER_COMPUTE_TYPE',
                'FASTER_WHISPER_BEAM_SIZE', 'FASTER_WHISPER_NUM_WORKERS', 'FASTER_WHISPER_VAD_FILTER'
            ]
        },
        fireworks_whisper: {
            label: 'Fireworks Whisper',
            essentialKeys: ['STT_FIREWORKS_API_KEY', 'STT_FIREWORKS_MODEL'],
            advancedKeys: []
        }
    },

    // Always-visible common keys — these are capture-loop concerns shared by
    // every recording, agnostic of which VAD backend is active.
    commonKeys: window.__managed
        ? ['STT_LANGUAGE']
        : ['STT_LANGUAGE', 'RECORDER_SILENCE_DURATION', 'RECORDER_MAX_SECONDS'],
    commonAdvancedKeys: window.__managed
        ? []
        : ['RECORDER_SPEECH_DURATION', 'RECORDER_NO_SPEECH_TIMEOUT', 'RECORDER_BEEP_WAIT_TIME']
    // VAD backend keys (STT_VAD_BACKEND, STT_VAD_SPEECH_THRESHOLD, and the
    // classic-amplitude knobs) are rendered in their own sections below.
};

// ── VAD section render ───────────────────────────────────────────────────────
// Rendered AFTER the provider-tab block. Two accordions: Silero (with check-
// box + status badge + threshold + test button) and Classic (amplitude fall-
// back knobs, always present but only used when Silero is off or has failed).

function renderVADSection(ctx) {
    if (window.__managed) return '';   // hosted/Docker — no local mic

    // NOTE: no data-key on the checkbox — it's a UI control that maps
    // boolean to the string setting STT_VAD_BACKEND via the change handler.
    // If we put data-key here, the settings save machinery would scoop it
    // up as a bool and overwrite our string with `true`/`false`. 2026-05-16.
    const siloreroCheckboxHtml = `
        <div class="settings-field" style="display:flex;align-items:center;gap:10px;margin-bottom:12px">
            <input type="checkbox" id="stt-silero-toggle" style="width:auto;margin:0">
            <label for="stt-silero-toggle" style="margin:0;cursor:pointer">
                <strong>Use Silero VAD</strong> (recommended — ML-based, robust to noise)
            </label>
            <span id="vad-status-badge" style="margin-left:auto;font-size:0.85em;opacity:0.8">checking…</span>
        </div>`;

    const testButtonHtml = `
        <div class="settings-field" style="margin-top:16px;padding-top:12px;border-top:1px solid rgba(255,255,255,0.08)">
            <button id="vad-test-btn" type="button" style="padding:6px 14px">Test my voice (5s)</button>
            <div id="vad-test-result" style="margin-top:10px;font-family:monospace;font-size:0.9em;opacity:0.85"></div>
        </div>`;

    const sileroFields = ctx.renderFields(['STT_VAD_SPEECH_THRESHOLD']);

    const classicKeys = [
        'RECORDER_BACKGROUND_PERCENTILE',
        'RECORDER_SILENCE_THRESHOLD',
        'RECORDER_NOISE_MULTIPLIER',
        'RECORDER_LEVEL_HISTORY_SIZE',
    ];

    return `
        <div style="margin-top:28px;padding-top:8px;border-top:2px solid rgba(255,255,255,0.12)">
            <h3 style="margin:8px 0 12px 0">Voice Activity Detection</h3>
            <div class="settings-accordion" data-accordion="stt-silero">
                <div class="settings-accordion-header" data-accordion-toggle="stt-silero">
                    <span class="accordion-arrow">▼</span>
                    <h4>Silero VAD</h4>
                </div>
                <div class="settings-accordion-body" data-accordion-body="stt-silero">
                    ${siloreroCheckboxHtml}
                    ${sileroFields}
                    ${testButtonHtml}
                </div>
            </div>
            ${ctx.renderAccordion('stt-classic', classicKeys, 'Classic VAD (fallback engine — used when Silero is off or unavailable)')}
        </div>
    `;
}

// ── Helpers ──────────────────────────────────────────────────────────────────

async function updateVadStatusBadge(el, syncCheckbox = false) {
    const badge = el.querySelector('#vad-status-badge');
    if (!badge) return;
    try {
        const resp = await fetch('/api/stt/vad-status', { credentials: 'same-origin' });
        const data = await resp.json();
        // ONLY sync checkbox from server state on initial load. Subsequent
        // polls update the badge text but never touch the checkbox — otherwise
        // a user mid-edit gets their unsaved click reverted by the next poll.
        // 2026-05-16: this was the actual cause of the "checkbox unchecks
        // itself after I check it" bug Krem was hitting.
        if (syncCheckbox) {
            const checkbox = el.querySelector('#stt-silero-toggle');
            if (checkbox) {
                checkbox.checked = data.intent === 'silero';
            }
        }
        let label, color;
        if (data.state === 'ready') { label = '✓ ready'; color = '#5fd17a'; }
        else if (data.state === 'pending') { label = '⟳ downloading…'; color = '#e0c068'; }
        else { label = `✗ unavailable: ${data.reason || 'unknown'}`; color = '#e07060'; }
        badge.textContent = label;
        badge.style.color = color;
    } catch (e) {
        badge.textContent = '? status check failed';
        badge.style.color = '#e07060';
    }
}

function attachSileroCheckboxListener(ctx, el) {
    const checkbox = el.querySelector('#stt-silero-toggle');
    if (!checkbox) return;
    checkbox.addEventListener('change', () => {
        // Boolean checkbox → string setting value
        const newValue = checkbox.checked ? 'silero' : 'amplitude';
        if (ctx.markChanged) {
            ctx.markChanged('STT_VAD_BACKEND', newValue);
        }
    });
}

function attachVadTestListener(el) {
    const btn = el.querySelector('#vad-test-btn');
    const resultEl = el.querySelector('#vad-test-result');
    if (!btn || !resultEl) return;
    btn.addEventListener('click', async () => {
        btn.disabled = true;
        const originalLabel = btn.textContent;
        btn.textContent = 'Recording 5s — speak now…';
        resultEl.textContent = '';
        try {
            // Get CSRF token from meta tag (same pattern as elsewhere)
            const csrf = document.querySelector('meta[name="csrf-token"]')?.content || '';
            const resp = await fetch('/api/stt/vad-test', {
                method: 'POST',
                credentials: 'same-origin',
                headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': csrf },
                body: JSON.stringify({ duration_s: 5.0 })
            });
            const data = await resp.json();
            if (!data.ok) {
                resultEl.innerHTML = `<span style="color:#e07060">✗ ${data.error || 'test failed'}</span>`;
            } else {
                const colorMap = { comfortable: '#5fd17a', marginal: '#e0c068', too_high: '#e07060' };
                const color = colorMap[data.verdict] || '#fff';
                resultEl.innerHTML = `
                    <div>peak <strong style="color:${color}">${data.max_prob}</strong>
                         · mean ${data.mean_prob}
                         · ${data.num_chunks_scored} chunks scored
                         · audio amp ${data.peak_amp}</div>
                    <div style="margin-top:6px;color:${color}">${data.suggestion}</div>`;
            }
        } catch (e) {
            resultEl.innerHTML = `<span style="color:#e07060">✗ test request failed: ${e.message}</span>`;
        } finally {
            btn.disabled = false;
            btn.textContent = originalLabel;
        }
    });
}

export default {
    id: 'stt',
    name: 'STT',
    icon: '🎤',
    description: 'Speech-to-text engine and voice detection',

    render(ctx) {
        const cfg = _mergedConfig || tabConfig;
        return renderProviderTab(cfg, ctx) + renderVADSection(ctx);
    },

    async attachListeners(ctx, el) {
        {
            _mergedConfig = await mergeRegistryProviders(tabConfig);
            if (Object.keys(_mergedConfig.providers).length > Object.keys(tabConfig.providers).length) {
                const body = el.querySelector('.settings-tab-body') || el;
                body.innerHTML = this.render(ctx);
                if (ctx.attachAccordionListeners) ctx.attachAccordionListeners(el);
            }
        }
        const cfg = _mergedConfig || tabConfig;
        attachProviderListeners(cfg, ctx, el);

        // VAD-specific wiring. Initial call syncs checkbox from server state
        // (= last saved intent). Subsequent polls only update the badge —
        // never the checkbox — so a user's mid-edit click can't be reverted.
        updateVadStatusBadge(el, true);
        attachSileroCheckboxListener(ctx, el);
        attachVadTestListener(el);
        // Refresh status every 5s while user is on this tab — picks up
        // background warmup completion if user opens settings during boot
        const refreshInterval = setInterval(() => updateVadStatusBadge(el, false), 5000);
        // Stop polling when tab is removed
        const observer = new MutationObserver(() => {
            if (!document.body.contains(el)) {
                clearInterval(refreshInterval);
                observer.disconnect();
            }
        });
        observer.observe(document.body, { childList: true, subtree: true });
    }
};
