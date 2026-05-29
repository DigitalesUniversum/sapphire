// views/webhooks.js - Triggers › Webhooks. External HTTP triggers that let
// other services poke Sapphire via a URL. Single column. Per-webhook activity
// history is deferred to the notification-history system.
import { renderTriggerTabs, bindTriggerTabs } from '../shared/trigger-tabs.js';
import { helpPills } from '../features/video-link.js';
import * as TR from '../shared/trigger-common.js';
import { fetchTasksByType, fetchStatus } from '../shared/continuity-api.js';

let container = null;
let webhooks = [];
let status = {};
let pollTimer = null;

export default {
    init(el) { container = el; },
    async show() { await load(); render(); startPoll(); },
    hide() { stopPoll(); }
};

function startPoll() { stopPoll(); pollTimer = setInterval(refresh, 5000); }
function stopPoll() { if (pollTimer) { clearInterval(pollTimer); pollTimer = null; } }

async function load() {
    try {
        const [w, s] = await Promise.all([fetchTasksByType('webhook'), fetchStatus()]);
        webhooks = w; status = s;
    } catch (e) { console.warn('Webhooks load failed:', e); }
}

async function refresh() { await load(); update(); }

function render() {
    if (!container) return;
    container.innerHTML = `
        ${renderTriggerTabs('webhooks', helpPills('Webhooks', { video: '1DiQ4oUC6R0', doc: 'DAEMONS-WEBHOOKS.md', inline: true }))}
        <div id="w-status"></div>
        <div class="view-body view-scroll">
            <div class="trigger-single">
                <div class="sched-col-header">
                    <h3>Webhooks</h3>
                    <button class="btn-sm" id="w-import" title="Import webhook">⬇</button>
                    <button class="btn-sm btn-primary" id="w-new">+ Webhook</button>
                </div>
                <div id="w-list"></div>
            </div>
        </div>`;
    bindTriggerTabs(container);
    container.querySelector('#w-new')?.addEventListener('click', () => TR.openEditor(null, 'webhook', refresh));
    container.querySelector('#w-import')?.addEventListener('click', () => TR.importTask('webhook', webhooks, refresh));
    TR.bindActions(container.querySelector('.view-body'), () => webhooks, refresh);
    update();
}

function update() {
    const scrollEl = container?.querySelector('.view-scroll');
    const scrollTop = scrollEl?.scrollTop || 0;

    const list = container?.querySelector('#w-list');
    if (list) list.innerHTML = TR.renderWebhookList(webhooks);
    if (scrollEl) scrollEl.scrollTop = scrollTop;

    const enabled = webhooks.filter(w => w.enabled).length;
    const statusEl = container?.querySelector('#w-status');
    if (statusEl) statusEl.innerHTML = TR.statusRow({
        enabled, total: webhooks.length, running: status.running,
        desc: 'External HTTP triggers — let other services poke Sapphire via a URL.'
    });
}
