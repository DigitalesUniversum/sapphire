// shared/trigger-tabs.js - Shared tab bar for the Triggers group views
// (Heartbeat / Scheduled / Daemons / Webhooks). Twin of persona-tabs.js:
// reuses the .persona-tabs / .persona-tab styling and navigates via switchView.
import { switchView } from '../core/router.js';

const TABS = [
    { id: 'heartbeat', label: 'Heartbeat', icon: '\u{1F493}' },
    { id: 'scheduled', label: 'Scheduled', icon: '\u{1F4C5}' },
    { id: 'daemons', label: 'Daemons', icon: '\u{1F4E1}' },
    { id: 'webhooks', label: 'Webhooks', icon: '\u{1F517}' },
];

/**
 * Render the shared tab bar for the Triggers group.
 * @param {string} activeId - currently active tab/view id
 * @param {string} rightSlot - optional HTML pinned to the right (e.g. help pills)
 * @returns {string} HTML string
 */
export function renderTriggerTabs(activeId, rightSlot = '') {
    return `<div class="persona-tabs">
        ${TABS.map(t => `<button class="persona-tab${t.id === activeId ? ' active' : ''}" data-view="${t.id}">${t.icon} ${t.label}</button>`).join('')}
        ${rightSlot}
    </div>`;
}

/** Bind tab clicks → switchView. Call once per render (delegation-safe). */
export function bindTriggerTabs(container) {
    const tabs = container.querySelector('.persona-tabs');
    if (!tabs) return;
    tabs.addEventListener('click', e => {
        const btn = e.target.closest('.persona-tab');
        if (!btn) return;
        const viewId = btn.dataset.view;
        if (viewId) switchView(viewId);
    });
}
