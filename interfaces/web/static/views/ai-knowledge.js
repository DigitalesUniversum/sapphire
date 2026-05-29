// views/ai-knowledge.js - Mind › AI Knowledge. Reference data the AI writes on
// its own (research, notes). Shares the knowledge scope domain + renderer with
// Human Knowledge; only the AI creates entries here (tabType 'ai').
import { renderSectionHeader, bindSectionHeader } from '../shared/section-header.js';
import { helpPills } from '../features/video-link.js';
import { renderScopeSidebar, bindScopeSidebar } from '../shared/scope-sidebar.js';
import { listScopes } from '../shared/scope-api.js';
import { MIND_TABS, scopeForChatTab, subscribeMindDomain } from '../shared/mind-common.js';
import { renderKnowledge } from '../shared/mind-knowledge.js';

const SCOPE_KEY = 'knowledge_scope';
const DOMAIN = 'knowledge';
const SCOPE_ENDPOINT = '/api/knowledge/scopes';
const TAB_TYPE = 'ai';

let container = null;
let scope = 'default';
let scopes = [];
let unsub = null;

export default {
    init(el) { container = el; },
    async show() {
        if (!unsub) unsub = subscribeMindDomain(DOMAIN, () => scope, () => container?.offsetParent !== null,
            () => renderKnowledge(content(), TAB_TYPE, scope));
        if (window._mindScope) { scope = window._mindScope; delete window._mindScope; }
        else { const s = await scopeForChatTab(SCOPE_KEY); if (s) scope = s; }
        delete window._mindTab;
        scopes = await listScopes(SCOPE_ENDPOINT);
        render();
    },
    hide() { if (unsub) { unsub(); unsub = null; } }
};

function content() { return container?.querySelector('#aikn-content'); }

function render() {
    if (!container) return;
    container.innerHTML = `
        ${renderSectionHeader({ tabs: MIND_TABS, active: 'ai-knowledge', help: helpPills('AI Knowledge', { video: 'I3g3tzukpV0', doc: 'KNOWLEDGE.md', inline: true }), status: 'Reference data Sapphire writes on her own — you can read and delete.' })}
        <div class="two-panel">
            ${renderScopeSidebar(scopes, scope)}
            <div class="panel-right">
                <div class="view-body view-scroll" id="aikn-content"></div>
            </div>
        </div>`;
    bindSectionHeader(container);
    bindScopeSidebar(container, {
        onScopeChange: (s) => { scope = s; render(); },
        onChanged: async (s) => { scope = s || 'default'; scopes = await listScopes(SCOPE_ENDPOINT); render(); },
    });
    renderKnowledge(content(), TAB_TYPE, scope);
}
