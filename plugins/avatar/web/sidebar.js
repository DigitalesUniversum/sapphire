// 3D Avatar — threejs GLTF with animation blending driven by SSE events
import * as eventBus from '/static/core/event-bus.js';
import * as audio from '/static/audio.js';
import { triggerSendWithText, handleStop } from '/static/handlers/send-handlers.js';
import { createEnvironment } from './environment.js';
import { createCameraOrbitSystem } from './camera-orbits.js';
import { createPlayerController } from './player-controller.js';
import { createMissileCommand } from './missile-command.js';

// Routed through Sapphire's /cdn-cache/ proxy — first request fetches from
// esm.sh and saves to user/cdn_cache/, subsequent serves from disk. ?bundle
// inlines all internal deps so the cached file is self-contained (otherwise
// internal esm.sh imports would still hit the CDN). 2026-05-13.
const THREE_CDN = '/cdn-cache/esm.sh/three@0.170.0?bundle&target=es2022';
const GLTF_CDN  = '/cdn-cache/esm.sh/three@0.170.0/addons/loaders/GLTFLoader.js?bundle&target=es2022&external=three';
const ORBIT_CDN = '/cdn-cache/esm.sh/three@0.170.0/addons/controls/OrbitControls.js?bundle&target=es2022&external=three';
const CROSSFADE_MS = 400;

// Hardcoded fallback defaults (used when no config exists). Keys must match
// state names that some entry in TRANSITIONS actually targets — anything else
// is dead. 'error' routes to toasts; 'wave' is the greeting (played by clip
// name directly, not via this map); 'thinking' had no event mapping.
const FALLBACK_TRACK_MAP = {
    idle: 'idle', listening: 'listening', processing: 'thinking',
    typing: 'defaultanim', speaking: 'attention',
    toolcall: 'attention2', wakeword: 'attention', happy: 'happy',
    agent: 'thinking', cron: 'thinking',
    user_typing: 'attention', reading: 'listening',
};
const FALLBACK_IDLE_POOL = [
    { track: 'idle', weight: 60, oneshot: false },
    { track: 'defaultanim', weight: 10, oneshot: false },  // typing on keyboard — idle variety
    { track: 'listening', weight: 10, oneshot: false },
    { track: 'attention', weight: 5, oneshot: true },
    { track: 'happy', weight: 4, oneshot: true },
    { track: 'wave', weight: 3, oneshot: true },
];
const FALLBACK_CAMERA = { x: 0, y: 1.3, z: 4.4 };
const FALLBACK_TARGET = { x: 0, y: 1.1, z: 0 };

// Flat event → state-name map. The rail system being built next will split
// this into AI-initiated vs user-initiated rails; for now every event just
// crossfades to its mapped state and the existing oneshot machinery returns
// to idle when a transient clip finishes. Error events (LLM/TTS/STT) now
// route to toasts instead of the avatar — dropped from this map intentionally.
const TRANSITIONS = {
    [eventBus.Events.STT_RECORDING_START]:  'listening',
    [eventBus.Events.STT_RECORDING_END]:    'processing',
    [eventBus.Events.STT_PROCESSING]:       'processing',
    [eventBus.Events.AI_TYPING_START]:      'typing',
    [eventBus.Events.AI_TYPING_END]:        'happy',
    [eventBus.Events.TTS_PLAYING]:          'speaking',
    [eventBus.Events.TTS_STOPPED]:          'idle',
    [eventBus.Events.TOOL_EXECUTING]:       'toolcall',
    [eventBus.Events.TOOL_COMPLETE]:        'typing',
    [eventBus.Events.WAKEWORD_DETECTED]:    'wakeword',
    [eventBus.Events.AGENT_SPAWNED]:        'agent',
    [eventBus.Events.AGENT_COMPLETED]:      'happy',
    [eventBus.Events.AGENT_DISMISSED]:      'idle',
    [eventBus.Events.CONTINUITY_TASK_STARTING]: 'cron',
    [eventBus.Events.CONTINUITY_TASK_COMPLETE]: 'idle',
    [eventBus.Events.USER_TYPING]:          'user_typing',
    [eventBus.Events.USER_SENT]:            'reading',
};

// Track cleanup between sidebar reloads
let _cleanup = null;
// Monotonic init token. Bumped on each init() entry. The setup body is
// async — without a token, two concurrent loadSidebar()/init() calls (chat
// switch + SPICE_CHANGED + PROMPT_DELETED fire in burst) both see _cleanup
// as null (the first hasn't assigned yet because its setup is mid-flight)
// and both run full WebGL + observer + RAF setup. The earlier one's
// resources get orphaned, leaking WebGL contexts toward the browser cap.
// Token check at end of init() lets only the latest entry win. 2026-05-14.
let _initToken = 0;

export async function init(container) {
    if (_cleanup) _cleanup();
    const myToken = ++_initToken;

    const canvas = container.querySelector('#avatar-canvas');
    const statusEl = container.querySelector('#avatar-status');
    if (!canvas) return;

    // --- Display mode controls ---
    const displayEl = container.querySelector('#avatar-display');
    const btnExpand = container.querySelector('#avatar-btn-expand');
    const btnFullscreen = container.querySelector('#avatar-btn-fullscreen');
    let displayMode = 'sidebar';
    let _onDisplayModeChange = null;  // set after scene is ready
    let _isPlayerMode = () => false;   // set after player controller is created
    let _setPlayerMode = null;        // set after player controller is created
    let _stopGame = null;             // set after game is created

    function setDisplayMode(mode) {
        if (mode === displayMode) return;
        if (displayMode === 'fullscreen' && document.fullscreenElement) {
            document.exitFullscreen?.();
        }
        displayMode = mode;
        if (mode === 'sidebar') {
            displayEl.classList.remove('avatar-fullwindow');
            canvas.style.height = '280px';
            if (btnExpand) { btnExpand.innerHTML = '&#x2922;'; btnExpand.title = 'Expand'; }
            if (btnFullscreen) btnFullscreen.style.display = '';
            // Exit player mode and reset camera
            if (_isPlayerMode()) _setPlayerMode?.(false);
        } else if (mode === 'fullwindow') {
            displayEl.classList.add('avatar-fullwindow');
            canvas.style.height = '100%';
            if (btnExpand) { btnExpand.innerHTML = '&#x2715;'; btnExpand.title = 'Collapse'; }
            if (btnFullscreen) btnFullscreen.style.display = '';
        } else if (mode === 'fullscreen') {
            displayEl.classList.add('avatar-fullwindow');
            canvas.style.height = '100%';
            displayEl.requestFullscreen?.().catch(() => { displayMode = 'fullwindow'; });
            if (btnExpand) { btnExpand.innerHTML = '&#x2715;'; btnExpand.title = 'Exit fullscreen'; }
            if (btnFullscreen) btnFullscreen.style.display = 'none';
        }
        if (_onDisplayModeChange) _onDisplayModeChange(mode);
    }

    btnExpand?.addEventListener('click', () => {
        setDisplayMode(displayMode === 'sidebar' ? 'fullwindow' : 'sidebar');
    });
    btnFullscreen?.addEventListener('click', () => setDisplayMode('fullscreen'));

    const _onEscKey = (e) => {
        if (e.key === 'Escape' && displayMode !== 'sidebar') {
            // First ESC: exit player mode. Second ESC: exit fullwindow.
            if (_isPlayerMode()) {
                _setPlayerMode?.(false);
                return;
            }
            if (displayMode === 'fullwindow') {
                setDisplayMode('sidebar');
            }
        }
    };
    const _onFsChange = () => {
        if (!document.fullscreenElement && displayMode === 'fullscreen') {
            displayMode = 'fullwindow'; // prevent recursive exitFullscreen call
            setDisplayMode('sidebar');
        }
    };
    document.addEventListener('keydown', _onEscKey);
    document.addEventListener('fullscreenchange', _onFsChange);

    // Early cleanup — covers cases where Three.js or model loading fails
    const _earlyCleanup = () => {
        clearInterval(_micPoll);
        _chatUnsubs.forEach(fn => fn());
        document.removeEventListener('keydown', _onEscKey);
        document.removeEventListener('fullscreenchange', _onFsChange);
        if (displayMode !== 'sidebar') {
            displayEl.classList.remove('avatar-fullwindow');
            canvas.style.height = '280px';
        }
        if (_cleanup === _earlyCleanup) _cleanup = null;
    };
    // Stale-init guard: another init() may have entered while this one was
    // running. If our token isn't current, tear down what we built and abort.
    if (myToken !== _initToken) {
        _earlyCleanup();
        return;
    }
    _cleanup = _earlyCleanup;

    // --- Fullwindow STT mic ---
    let _micPoll = null;
    const avatarMic = container.querySelector('#avatar-mic');
    if (avatarMic) {
        const updateMicState = () => {
            const ttsActive = audio.isTtsPlaying() || audio.isLocalTtsPlaying();
            if (ttsActive) {
                avatarMic.classList.add('tts-playing');
                avatarMic.classList.remove('recording');
                avatarMic.textContent = '\u23F9';
            } else if (audio.getRecState()) {
                avatarMic.classList.add('recording');
                avatarMic.classList.remove('tts-playing');
                avatarMic.textContent = '\u23FA';
            } else {
                avatarMic.classList.remove('recording', 'tts-playing');
                avatarMic.textContent = '\uD83C\uDFA4';
            }
        };
        avatarMic.addEventListener('mousedown', async (e) => {
            e.preventDefault();
            if (audio.isTtsPlaying() || audio.isLocalTtsPlaying()) {
                audio.stop(true);
            } else {
                await audio.handlePress(avatarMic);
            }
            updateMicState();
        });
        avatarMic.addEventListener('mouseup', async () => {
            await audio.handleRelease(avatarMic, triggerSendWithText);
            updateMicState();
        });
        avatarMic.addEventListener('touchstart', async (e) => {
            e.preventDefault();
            if (audio.isTtsPlaying() || audio.isLocalTtsPlaying()) {
                audio.stop(true);
            } else {
                await audio.handlePress(avatarMic);
            }
            updateMicState();
        });
        avatarMic.addEventListener('touchend', async () => {
            await audio.handleRelease(avatarMic, triggerSendWithText);
            updateMicState();
        });
        avatarMic.addEventListener('mouseleave', () => {
            if (audio.getRecState()) {
                setTimeout(() => {
                    if (audio.getRecState()) audio.handleRelease(avatarMic, triggerSendWithText);
                    updateMicState();
                }, 500);
            }
        });
        // Poll mic state while in expanded mode
        _micPoll = setInterval(updateMicState, 300);
    }

    // --- Chat overlay (WoW-style) ---
    const chatOverlay = container.querySelector('#avatar-chat');
    const chatLog = container.querySelector('#avatar-chat-log');
    const chatToggle = container.querySelector('#avatar-chat-toggle');
    const chatStop = container.querySelector('#avatar-chat-stop');
    const chatVolume = container.querySelector('#avatar-chat-volume');
    let _chatUnsubs = [];
    let _aiStreamEl = null;  // current streaming message element
    let _thinkOpen = false;  // true while inside <think> block
    const MAX_MESSAGES = 50;

    if (chatOverlay) {
        // Collapse/expand toggle
        chatToggle?.addEventListener('click', () => {
            chatOverlay.classList.toggle('collapsed');
            chatToggle.textContent = chatOverlay.classList.contains('collapsed') ? '\u25B2' : '\u25BC';
        });

        // Stop generation
        chatStop?.addEventListener('click', () => handleStop());

        // Volume slider
        if (chatVolume) {
            chatVolume.value = Math.round(audio.getVolume() * 100);
            chatVolume.addEventListener('input', () => {
                audio.setVolume(chatVolume.value / 100);
            });
        }

        function addChatMsg(role, text) {
            if (!chatLog) return;
            const el = document.createElement('div');
            el.className = `chat-msg chat-msg-${role}`;
            el.textContent = text;
            chatLog.appendChild(el);
            // Trim old messages
            while (chatLog.children.length > MAX_MESSAGES) chatLog.removeChild(chatLog.firstChild);
            chatLog.scrollTop = chatLog.scrollHeight;
            return el;
        }

        // User sent a message
        _chatUnsubs.push(eventBus.on(eventBus.Events.USER_SENT, (data) => {
            if (data?.text) addChatMsg('user', data.text);
        }));

        // AI streaming chunks
        // Backend sends think content wrapped in <think>...</think> as type:content chunks.
        // The </think> only arrives when the LLM finishes thinking (done event).
        // Strategy: track open/close tags, only display text outside think blocks.
        _chatUnsubs.push(eventBus.on(eventBus.Events.CHAT_CHUNK, (data) => {
            if (!data?.text) return;
            let text = data.text;

            // Check for think open/close tags in this chunk
            if (text.includes('<think>') || text.includes('<seed:think')) {
                _thinkOpen = true;
                // Strip everything from <think> onward in this chunk
                text = text.replace(/<(?:seed:)?think[^>]*>[\s\S]*/i, '');
            }
            if (_thinkOpen && (text.includes('</think>') || text.includes('</seed:think'))) {
                _thinkOpen = false;
                // Keep everything after the closing tag
                text = text.replace(/[\s\S]*<\/[^>]*think[^>]*>/i, '');
            }

            // If inside think block, skip entirely
            if (_thinkOpen) return;

            // Strip avatar tags
            text = text.replace(/<<avatar:\s*[a-zA-Z0-9_]+(?:\s+\d+(?:\.\d+)?s)?>>/g, '');
            // Strip any stray think tags (edge case: open+close in same chunk)
            text = text.replace(/<\/?(?:seed:)?think[^>]*>/gi, '');

            if (text && text.trim()) {
                if (!_aiStreamEl) _aiStreamEl = addChatMsg('ai', '');
                _aiStreamEl.textContent += text;
                chatLog.scrollTop = chatLog.scrollHeight;
            }
        }));

        // AI done typing — finalize stream element
        _chatUnsubs.push(eventBus.on(eventBus.Events.AI_TYPING_END, () => {
            _aiStreamEl = null;
            _thinkOpen = false;
        }));

        // Wakeword conversations bypass streaming — fetch messages when LLM finishes
        let _wakewordPending = false;
        _chatUnsubs.push(eventBus.on(eventBus.Events.WAKEWORD_DETECTED, () => {
            _wakewordPending = true;
        }));
        _chatUnsubs.push(eventBus.on(eventBus.Events.AI_TYPING_END, async () => {
            if (!_wakewordPending) return;
            _wakewordPending = false;
            try {
                const resp = await fetch('/api/history');
                if (!resp.ok) return;
                const data = await resp.json();
                const msgs = data.messages || [];
                const recent = msgs.slice(-2);
                for (const m of recent) {
                    if (m.role === 'user') {
                        const text = (m.content || '').trim();
                        if (text) addChatMsg('user', text);
                    } else if (m.role === 'assistant' && m.parts) {
                        // Assistant uses parts array
                        const text = m.parts
                            .filter(p => p.type === 'content')
                            .map(p => p.text || '')
                            .join('')
                            .replace(/<think>[\s\S]*?<\/think>/gi, '')
                            .trim();
                        if (text) addChatMsg('ai', text);
                    }
                }
            } catch (e) { /* silent */ }
        }));

        // AI starts — show stop button, reset stream state
        _chatUnsubs.push(eventBus.on(eventBus.Events.AI_TYPING_START, () => {
            _aiStreamEl = null;
            _thinkOpen = false;
            chatStop?.classList.add('visible');
        }));

        // AI done or error — hide stop button
        const hideStop = () => chatStop?.classList.remove('visible');
        _chatUnsubs.push(eventBus.on(eventBus.Events.AI_TYPING_END, hideStop));
        _chatUnsubs.push(eventBus.on(eventBus.Events.LLM_ERROR, hideStop));
        _chatUnsubs.push(eventBus.on(eventBus.Events.TTS_STOPPED, hideStop));
    }

    // --- Load config from backend ---
    let avatarConfig = {};
    let modelFile = 'sapphire.glb';
    let trackMap = { ...FALLBACK_TRACK_MAP };
    let idlePool = [...FALLBACK_IDLE_POOL];
    let greetingTrack = 'wave';
    let camPos = { ...FALLBACK_CAMERA };
    let camTarget = { ...FALLBACK_TARGET };
    let modelScale = 1.0;

    try {
        const resp = await fetch('/api/plugin/avatar/config');
        if (resp.ok) {
            avatarConfig = await resp.json();
            modelFile = avatarConfig.active_model || modelFile;
            const modelCfg = (avatarConfig.models || {})[modelFile];
            if (modelCfg) {
                if (modelCfg.track_map) trackMap = { ...FALLBACK_TRACK_MAP, ...modelCfg.track_map };
                if (modelCfg.idle_pool?.length) idlePool = modelCfg.idle_pool;
                if (modelCfg.greeting_track !== undefined) greetingTrack = modelCfg.greeting_track;
                if (modelCfg.camera) camPos = modelCfg.camera;
                if (modelCfg.target) camTarget = modelCfg.target;
                if (modelCfg.scale) modelScale = modelCfg.scale;
            }
        }
    } catch (e) { /* use fallbacks */ }

    const MODEL_URL = `/api/avatar/${modelFile}`;

    // Build oneshot set from idle pool config
    const ONESHOT_TRACKS = new Set(
        idlePool.filter(v => v.oneshot).map(v => v.track)
    );
    // Also treat known oneshot animations as oneshot
    for (const t of ['happy', 'wave', 'attention', 'attention2']) ONESHOT_TRACKS.add(t);

    // Weighted random idle pick
    function pickIdleVariant() {
        const total = idlePool.reduce((s, v) => s + v.weight, 0);
        let roll = Math.random() * total;
        for (const v of idlePool) {
            roll -= v.weight;
            if (roll <= 0) return v.track;
        }
        return trackMap.idle || 'idle';
    }

    // Dynamic imports (cached after first load)
    let THREE, GLTFLoader, OrbitControls;
    try {
        THREE = await import(THREE_CDN);
        const gltfMod = await import(GLTF_CDN);
        const orbitMod = await import(ORBIT_CDN);
        GLTFLoader = gltfMod.GLTFLoader;
        OrbitControls = orbitMod.OrbitControls;
    } catch (e) {
        console.error('[Avatar] Failed to load three.js:', e);
        canvas.style.display = 'none';
        container.innerHTML += '<div style="text-align:center;color:var(--text-muted);padding:8px;">Three.js failed to load</div>';
        return;
    }

    // Scene
    const scene = new THREE.Scene();
    const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;

    // Environment (procedural scene — only visible in expanded modes)
    const env = createEnvironment(scene, THREE, renderer);

    // Camera — from config
    const camera = new THREE.PerspectiveCamera(30, canvas.clientWidth / canvas.clientHeight, 0.1, 200);
    camera.position.set(camPos.x, camPos.y, camPos.z);

    // Orbit controls
    const controls = new OrbitControls(camera, canvas);
    controls.target.set(camTarget.x, camTarget.y, camTarget.z);
    controls.minDistance = 0.5;
    controls.maxDistance = 20;
    controls.enablePan = true;
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.rotateSpeed = 0.5;
    controls.panSpeed = 0.4;
    controls.maxPolarAngle = Math.PI * 0.85;
    controls.update();

    // Camera orbit system
    const orbitSystem = createCameraOrbitSystem(camera, controls, THREE);

    // Player controller (WASD + mouse look)
    const playerCtrl = createPlayerController(camera, controls, canvas, THREE);
    _isPlayerMode = () => playerCtrl.isEnabled();
    _setPlayerMode = setPlayerMode;

    // Orbit toggle button
    const btnOrbit = container.querySelector('#avatar-btn-orbit');
    if (btnOrbit) {
        btnOrbit.classList.add('orbit-active');  // on by default
        btnOrbit.addEventListener('click', () => {
            if (playerCtrl.isEnabled()) return;  // can't orbit in player mode
            const on = orbitSystem.toggle();
            btnOrbit.classList.toggle('orbit-active', on);
        });
    }

    // Player mode toggle
    const btnPlayer = container.querySelector('#avatar-btn-player');
    function setPlayerMode(on) {
        if (on === playerCtrl.isEnabled()) return;
        if (!on && _stopGame) _stopGame();
        playerCtrl.toggle();
        btnPlayer?.classList.toggle('player-active', on);
        displayEl.classList.toggle('player-mode', on);
        if (on && orbitSystem.isEnabled()) {
            orbitSystem.toggle();
            btnOrbit?.classList.remove('orbit-active');
        }
    }
    if (btnPlayer) {
        btnPlayer.addEventListener('click', () => setPlayerMode(!playerCtrl.isEnabled()));
    }

    // Auto-activate player mode on WASD/Space in fullwindow
    const PLAYER_TRIGGER_KEYS = new Set(['KeyW', 'KeyA', 'KeyS', 'KeyD', 'Space']);
    const _onPlayerAutoActivate = (e) => {
        if (displayMode === 'sidebar') return;
        if (playerCtrl.isEnabled()) return;
        if (!PLAYER_TRIGGER_KEYS.has(e.code)) return;
        e.preventDefault();
        setPlayerMode(true);
    };
    document.addEventListener('keydown', _onPlayerAutoActivate);

    // Missile Command game
    const missileGame = createMissileCommand(scene, THREE, camera, canvas, eventBus.dispatch);
    _stopGame = () => { if (missileGame.isActive()) missileGame.stop(); };

    const _onGameKey = (e) => {
        if (e.code !== 'KeyG') return;
        if (displayMode === 'sidebar') return;
        if (!playerCtrl.isEnabled()) return;  // must be in player mode
        e.preventDefault();
        if (missileGame.isActive()) {
            missileGame.stop();
        } else {
            missileGame.start(displayEl);
        }
    };
    document.addEventListener('keydown', _onGameKey);

    // Double-click to reset camera (only in orbit mode)
    canvas.addEventListener('dblclick', () => {
        if (playerCtrl.isEnabled()) return;
        camera.position.set(camPos.x, camPos.y, camPos.z);
        controls.target.set(camTarget.x, camTarget.y, camTarget.z);
        controls.update();
    });

    // Lighting (default — used in sidebar mode, hidden when environment is active)
    const defaultAmbient = new THREE.AmbientLight(0xffffff, 0.7);
    scene.add(defaultAmbient);
    const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
    dirLight.position.set(2, 3, 2);
    scene.add(dirLight);
    const rimLight = new THREE.DirectionalLight(0x4a9eff, 0.4);
    rimLight.position.set(-1, 2, -2);
    scene.add(rimLight);

    // Resize
    function resize() {
        const w = canvas.clientWidth;
        const h = canvas.clientHeight;
        if (canvas.width !== w || canvas.height !== h) {
            renderer.setSize(w, h, false);
            camera.aspect = w / h;
            camera.updateProjectionMatrix();
        }
    }
    resize();

    // Load model
    const loader = new GLTFLoader();
    let mixer, actions = {}, currentAction = null;

    // State vars must exist before the greeting flow inside the try-block
    // calls setState('idle'). Function declarations hoist; `let` doesn't.
    let idleTimer = null;
    let current = greetingTrack ? 'wave' : 'idle';

    try {
        const gltf = await new Promise((resolve, reject) => {
            loader.load(MODEL_URL, resolve, undefined, reject);
        });

        scene.add(gltf.scene);

        // Apply scale from config (user-controlled, default 1.0)
        if (modelScale !== 1.0) {
            gltf.scene.scale.multiplyScalar(modelScale);
        }

        // Frame camera on model center after scaling
        const box = new THREE.Box3().setFromObject(gltf.scene);
        const size = box.getSize(new THREE.Vector3());
        const center = box.getCenter(new THREE.Vector3());

        // If no custom camera was saved, auto-frame based on model bounds
        const hasCustomCamera = (avatarConfig.models || {})[modelFile]?.camera;
        if (!hasCustomCamera) {
            const dist = Math.max(size.y, size.x) * 2.5;
            camTarget = { x: center.x, y: center.y, z: center.z };
            camPos = { x: center.x, y: center.y + size.y * 0.1, z: center.z + dist };
            camera.position.set(camPos.x, camPos.y, camPos.z);
            controls.target.set(camTarget.x, camTarget.y, camTarget.z);
            controls.update();
        }

        // Dynamic zoom limits based on model size
        const maxDim = Math.max(size.x, size.y, size.z);
        controls.minDistance = maxDim * 0.3;
        controls.maxDistance = maxDim * 10;

        // Feed model info to orbit system and randomize start position
        orbitSystem.setModelInfo(center, size.y);
        orbitSystem.randomStart();

        mixer = new THREE.AnimationMixer(gltf.scene);

        for (const clip of gltf.animations) {
            const action = mixer.clipAction(clip);
            action.clampWhenFinished = true;
            actions[clip.name] = action;
        }

        // Start with greeting track, then idle. Go through setState('idle')
        // after the greeting so the variety picker actually arms — going
        // through crossfadeTo directly skipped scheduleIdleVariant and the
        // pool wouldn't start cycling until the first SSE event landed.
        const greetAction = greetingTrack ? actions[greetingTrack] : null;
        if (greetAction) {
            greetAction.setLoop(THREE.LoopOnce);
            greetAction.play();
            currentAction = greetAction;
            mixer.addEventListener('finished', function onGreetDone(e) {
                if (e.action === greetAction) {
                    mixer.removeEventListener('finished', onGreetDone);
                    setState('idle');
                }
            });
        } else {
            setState('idle');
        }

        // Enable avatar shadow casting for environment
        env.enableAvatarShadows(gltf.scene);

        // Load configured location + populate selector
        const configuredLocation = avatarConfig.active_location || 'cabin';
        env.setLocation(configuredLocation);

        const locationSelect = container.querySelector('#avatar-location-select');
        if (locationSelect) {
            for (const loc of env.listLocations()) {
                const opt = document.createElement('option');
                opt.value = loc.name;
                opt.textContent = loc.name[0].toUpperCase() + loc.name.slice(1);
                if (loc.name === configuredLocation) opt.selected = true;
                locationSelect.appendChild(opt);
            }
            locationSelect.addEventListener('change', async () => {
                const name = locationSelect.value;
                await env.setLocation(name);
                // Save to config
                fetch('/api/plugin/avatar/config', {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ active_location: name }),
                });
            });
        }

        // Wire environment toggle to display mode changes
        _onDisplayModeChange = (mode) => {
            const expanded = mode !== 'sidebar';
            env.setVisible(expanded);
            defaultAmbient.visible = !expanded;
            dirLight.visible = !expanded;
            // Rim light stays on always — it's her signature
            // Reset camera to default framing when returning to sidebar
            if (!expanded) {
                camera.position.set(camPos.x, camPos.y, camPos.z);
                controls.target.set(camTarget.x, camTarget.y, camTarget.z);
                controls.update();
            }
        };

        // Expand zoom limits for environment exploration
        controls.maxDistance = 35;

    } catch (e) {
        console.error('[Avatar] Failed to load model:', e);
        canvas.style.display = 'none';
        container.innerHTML += '<div style="text-align:center;color:var(--text-muted);padding:8px;">Model failed to load</div>';
        return;
    }

    // --- Animation crossfade ---
    function crossfadeTo(stateName) {
        const trackName = trackMap[stateName] || stateName;  // allow raw track names for idle variety
        const action = actions[trackName];
        if (!action || currentAction === action) return;

        action.reset();
        action.setLoop(ONESHOT_TRACKS.has(trackName) ? THREE.LoopOnce : THREE.LoopRepeat);
        action.clampWhenFinished = true;

        if (currentAction) {
            action.crossFadeFrom(currentAction, CROSSFADE_MS / 1000, true);
        }
        action.play();
        currentAction = action;

        // Oneshot tracks return to idle when the clip finishes. setState('idle')
        // re-arms the variety scheduler so the picker keeps cycling.
        if (ONESHOT_TRACKS.has(trackName)) {
            mixer.addEventListener('finished', function onDone(e) {
                if (e.action !== action) return;
                mixer.removeEventListener('finished', onDone);
                if (currentAction === action) setState('idle');
            });
        }
    }

    // --- Idle variety system ---
    function scheduleIdleVariant() {
        clearTimeout(idleTimer);
        const delay = 8000 + Math.random() * 12000;
        idleTimer = setTimeout(() => {
            if (current !== 'idle') return;
            const track = pickIdleVariant();
            crossfadeTo(track);
            if (!ONESHOT_TRACKS.has(track)) {
                scheduleIdleVariant();
            }
        }, delay);
    }

    // --- State (flat, no priorities) ---
    // The rail system replaces priority/persist/duration. For now setState
    // just records the latest event-mapped state and crossfades; the
    // ONESHOT_TRACKS handling inside crossfadeTo returns to idle when a
    // transient clip finishes. (`current` is declared earlier so the
    // greeting flow inside the model-load try-block can call setState.)
    function setState(name) {
        clearTimeout(idleTimer);
        current = name;
        crossfadeTo(name);
        if (statusEl) statusEl.textContent = name === 'idle' ? '' : name;
        if (name === 'idle') scheduleIdleVariant();
    }

    // Wire SSE events
    const unsubs = [];
    for (const [event, stateName] of Object.entries(TRANSITIONS)) {
        const unsub = eventBus.on(event, () => setState(stateName));
        if (unsub) unsubs.push(unsub);
    }

    // Sapph-triggered animations: <<avatar: trackname>> in chat responses.
    // Strip-phase version: plain oneshot. The rail system being built next
    // will own this handler properly (rails 1/2, mode=once|loop, wind-down).
    const avatarUnsub = eventBus.on('avatar_animate', (data) => {
        const { track } = data || {};
        const action = actions[track];
        if (!action) {
            console.warn(`[Avatar] Track "${track}" not found in model`);
            return;
        }

        action.reset();
        action.setLoop(THREE.LoopOnce);
        action.clampWhenFinished = true;
        if (currentAction) action.crossFadeFrom(currentAction, CROSSFADE_MS / 1000, true);
        action.play();
        currentAction = action;

        // Return to the latest SSE-mapped state when the clip finishes.
        mixer.addEventListener('finished', function onDone(e) {
            if (e.action !== action) return;
            mixer.removeEventListener('finished', onDone);
            if (currentAction === action) crossfadeTo(current);
        });
    });
    if (avatarUnsub) unsubs.push(avatarUnsub);

    // Render loop
    const clock = new THREE.Clock();
    let running = true;

    function animate() {
        if (!running) return;
        requestAnimationFrame(animate);
        const delta = clock.getDelta();
        if (mixer) mixer.update(delta);
        env.update(delta);
        if (playerCtrl.isEnabled()) {
            playerCtrl.update(delta);
        } else {
            orbitSystem.update(delta);
            controls.update();
        }
        missileGame.update(delta);
        resize();
        renderer.render(scene, camera);
    }
    animate();

    // Cleanup
    const _thisCleanup = () => {
        running = false;
        clearTimeout(idleTimer);
        clearInterval(_micPoll);
        unsubs.forEach(fn => fn());
        _chatUnsubs.forEach(fn => fn());
        document.removeEventListener('keydown', _onPlayerAutoActivate);
        document.removeEventListener('keydown', _onGameKey);
        missileGame.cleanup();
        playerCtrl.cleanup();
        orbitSystem.cleanup();
        controls.dispose();
        renderer.dispose();
        document.removeEventListener('keydown', _onEscKey);
        document.removeEventListener('fullscreenchange', _onFsChange);
        if (displayMode !== 'sidebar') {
            displayEl.classList.remove('avatar-fullwindow');
            canvas.style.height = '280px';
        }
        if (_cleanup === _thisCleanup) _cleanup = null;
    };

    // Stale-init guard: if another init() entered while this one was building
    // (rapid sidebar reload bursts), abandon THIS instance — teardown the
    // resources we just created and don't overwrite the newer instance's
    // _cleanup. Without this, both instances run concurrently, both register
    // observers + RAF loops, and the earlier one leaks. 2026-05-14.
    if (myToken !== _initToken) {
        _thisCleanup();
        return;
    }
    _cleanup = _thisCleanup;

    // Debounced disposal — a brief detach during framework DOM rebuild
    // (sidebar accordion refresh, view switch) should NOT permanently dispose
    // the WebGL context. Only treat detachment as final after 1s. Previously
    // any momentary detach killed the avatar irreversibly. 2026-05-13.
    let _detachTimer = null;
    const observer = new MutationObserver(() => {
        if (!document.contains(canvas)) {
            if (!_detachTimer) {
                _detachTimer = setTimeout(() => {
                    _detachTimer = null;
                    if (!document.contains(canvas)) {
                        if (_cleanup) _cleanup();
                        observer.disconnect();
                    }
                }, 1000);
            }
        } else if (_detachTimer) {
            clearTimeout(_detachTimer);
            _detachTimer = null;
        }
    });
    observer.observe(container.parentElement || document.body, { childList: true, subtree: true });
}
