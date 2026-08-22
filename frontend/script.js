const chatBox = document.getElementById('chat-box');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const logBox = document.getElementById('log-box');
const fileInput = document.getElementById('file-input');
const selectedFiles = document.getElementById('selected-files');
const sessionList = document.getElementById('session-list');
const newSessionBtn = document.getElementById('new-session-btn');
const currentSessionTitle = document.getElementById('current-session-title');
const researchMode = document.getElementById('research-mode');

let sessions = [];
let sessionId = localStorage.getItem('research_agent_session_id') || '';
const userNamespace = localStorage.getItem('research_agent_user_id') || 'default_user';
let streamingMessage = null;
let streamingText = '';
let activeStreamId = '';
let streamBuffer = '';
let streamFrameId = null;
let pendingStreamFinal = null;

function apiFetch(url, options = {}) {
    const headers = new Headers(options.headers || {});
    headers.set('X-User-ID', userNamespace);
    return fetch(url, { ...options, headers });
}

function formatSessionTime(value) {
    if (!value) {
        return '';
    }
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
        return value.replace('T', ' ');
    }
    return date.toLocaleString('zh-CN', {
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit'
    });
}

function updateSelectedFiles() {
    const files = Array.from(fileInput.files || []);
    selectedFiles.textContent = files.length
        ? files.map(file => file.name).join(' | ')
        : '未选择附件';
}

function appendLog(text, type = 'normal') {
    const item = document.createElement('div');
    item.classList.add('log-item');

    if (text.includes('异常') || text.includes('失败') || text.includes('[WARN]')) {
        type = 'error';
    } else if (text.includes('[UPLOAD]') || text.includes('[VISION]') || text.includes('[Query Rewrite]')) {
        type = 'success';
    } else if (text.startsWith('[')) {
        type = 'highlight';
    }

    if (type !== 'normal') {
        item.classList.add(type);
    }

    item.textContent = text;
    logBox.appendChild(item);
    logBox.scrollTop = logBox.scrollHeight;
}

function appendMessage(text, sender, isMarkdown = false) {
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('message', sender);

    const bubble = document.createElement('div');
    bubble.classList.add('bubble');
    if (isMarkdown && sender === 'system') {
        bubble.innerHTML = marked.parse(text);
    } else {
        bubble.textContent = text;
    }

    msgDiv.appendChild(bubble);
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function ensureStreamingMessage(streamId) {
    removeLoading();
    if (!streamingMessage) {
        streamingMessage = document.createElement('div');
        streamingMessage.classList.add('message', 'system');
        const bubble = document.createElement('div');
        bubble.classList.add('bubble', 'streaming');
        streamingMessage.appendChild(bubble);
        chatBox.appendChild(streamingMessage);
    }

    const nextStreamId = streamId || activeStreamId || 'answer';
    if (activeStreamId && activeStreamId !== nextStreamId) {
        streamingText = '';
        streamBuffer = '';
    }
    activeStreamId = nextStreamId;
    return streamingMessage.querySelector('.bubble');
}

function finishStreamRendering() {
    if (!pendingStreamFinal || !streamingMessage) {
        return;
    }

    const { text, resolve } = pendingStreamFinal;
    const bubble = streamingMessage.querySelector('.bubble');
    bubble.classList.remove('streaming');
    bubble.innerHTML = marked.parse(text);

    pendingStreamFinal = null;
    streamingMessage = null;
    streamingText = '';
    streamBuffer = '';
    activeStreamId = '';
    chatBox.scrollTop = chatBox.scrollHeight;
    resolve();
}

function paintStreamFrame() {
    streamFrameId = null;
    if (!streamingMessage) {
        return;
    }

    if (streamBuffer) {
        const chunkSize = Math.max(1, Math.ceil(streamBuffer.length / 36));
        streamingText += streamBuffer.slice(0, chunkSize);
        streamBuffer = streamBuffer.slice(chunkSize);
        streamingMessage.querySelector('.bubble').innerHTML = marked.parse(streamingText);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    if (streamBuffer) {
        streamFrameId = requestAnimationFrame(paintStreamFrame);
    } else if (pendingStreamFinal) {
        finishStreamRendering();
    }
}

function scheduleStreamFrame() {
    if (streamFrameId === null) {
        streamFrameId = requestAnimationFrame(paintStreamFrame);
    }
}

function appendStreamToken(text, streamId) {
    if (!text) {
        return;
    }
    ensureStreamingMessage(streamId);
    streamBuffer += text;
    scheduleStreamFrame();
}

function finalizeStreamMessage(text) {
    removeLoading();
    return new Promise(resolve => {
        if (!streamingMessage) {
            ensureStreamingMessage('final');
            streamBuffer = text;
        }
        pendingStreamFinal = { text, resolve };
        scheduleStreamFrame();
    });
}

function discardStreamMessage() {
    if (streamFrameId !== null) {
        cancelAnimationFrame(streamFrameId);
    }
    if (pendingStreamFinal) {
        pendingStreamFinal.resolve();
    }
    if (streamingMessage) {
        streamingMessage.remove();
    }
    streamingMessage = null;
    streamingText = '';
    streamBuffer = '';
    streamFrameId = null;
    pendingStreamFinal = null;
    activeStreamId = '';
}

function resetChatBox() {
    chatBox.innerHTML = '';
    appendMessage('你可以输入学术问题，也可以附带图片或 PDF。系统会把图片转成文本，再进入 Research Agent 流程。', 'system');
}

function showLoading() {
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('message', 'system');
    msgDiv.id = 'loading-indicator';

    const bubble = document.createElement('div');
    bubble.classList.add('bubble', 'typing');
    bubble.innerHTML = '<div class="dot"></div><div class="dot"></div><div class="dot"></div>';

    msgDiv.appendChild(bubble);
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function removeLoading() {
    const loading = document.getElementById('loading-indicator');
    if (loading) {
        loading.remove();
    }
}

function renderSessions() {
    sessionList.innerHTML = '';

    if (!sessions.length) {
        const empty = document.createElement('div');
        empty.className = 'session-empty';
        empty.textContent = '暂无历史会话';
        sessionList.appendChild(empty);
        return;
    }

    for (const session of sessions) {
        const item = document.createElement('div');
        item.className = 'session-item';
        if (session.session_id === sessionId) {
            item.classList.add('active');
        }

        const main = document.createElement('button');
        main.type = 'button';
        main.className = 'session-main';

        const title = document.createElement('span');
        title.className = 'session-title';
        title.textContent = session.title || session.session_id;

        const time = document.createElement('span');
        time.className = 'session-time';
        time.textContent = formatSessionTime(session.updated_at);

        main.appendChild(title);
        main.appendChild(time);
        main.addEventListener('click', async () => {
            if (session.session_id === sessionId) {
                return;
            }
            sessionId = session.session_id;
            localStorage.setItem('research_agent_session_id', sessionId);
            renderSessions();
            await loadSessionMessages(sessionId);
        });

        const actions = document.createElement('div');
        actions.className = 'session-actions';

        const renameBtn = document.createElement('button');
        renameBtn.type = 'button';
        renameBtn.className = 'session-action-btn';
        renameBtn.textContent = '改名';
        renameBtn.title = '重命名会话';
        renameBtn.addEventListener('click', async event => {
            event.stopPropagation();
            await renameSession(session);
        });

        const deleteBtn = document.createElement('button');
        deleteBtn.type = 'button';
        deleteBtn.className = 'session-action-btn danger';
        deleteBtn.textContent = '删除';
        deleteBtn.title = '删除会话';
        deleteBtn.addEventListener('click', async event => {
            event.stopPropagation();
            await deleteSession(session);
        });

        actions.appendChild(renameBtn);
        actions.appendChild(deleteBtn);
        item.appendChild(main);
        item.appendChild(actions);
        sessionList.appendChild(item);
    }

    const current = sessions.find(session => session.session_id === sessionId);
    currentSessionTitle.textContent = current?.title || 'Academic Research Copilot';
}

async function loadSessions(options = {}) {
    const response = await apiFetch('/api/sessions');
    const data = await response.json();
    sessions = data.sessions || [];

    if (!sessionId || !sessions.some(session => session.session_id === sessionId)) {
        if (sessions.length) {
            sessionId = sessions[0].session_id;
        } else {
            await createNewSession();
            return;
        }
    }

    localStorage.setItem('research_agent_session_id', sessionId);
    renderSessions();

    if (options.reloadMessages !== false) {
        await loadSessionMessages(sessionId);
    }
}

async function createNewSession() {
    const response = await apiFetch('/api/sessions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: 'New research session' })
    });
    const session = await response.json();
    sessionId = session.session_id;
    localStorage.setItem('research_agent_session_id', sessionId);
    await loadSessions();
}

async function renameSession(session) {
    const currentTitle = session.title || '';
    const title = window.prompt('请输入新的会话名称：', currentTitle);
    if (title === null) {
        return;
    }

    const nextTitle = title.trim();
    if (!nextTitle || nextTitle === currentTitle) {
        return;
    }

    const response = await apiFetch(`/api/sessions/${session.session_id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: nextTitle })
    });
    if (!response.ok) {
        appendLog(`会话重命名失败: HTTP ${response.status}`, 'error');
        return;
    }
    await loadSessions({ reloadMessages: false });
}

async function deleteSession(session) {
    const title = session.title || session.session_id;
    const message = `确定删除会话“${title}”吗？该会话的历史消息和摘要会从 SQLite 中删除。`;
    if (!window.confirm(message)) {
        return;
    }

    const deletingCurrent = session.session_id === sessionId;
    const response = await apiFetch(`/api/sessions/${session.session_id}`, { method: 'DELETE' });
    if (!response.ok) {
        appendLog(`会话删除失败: HTTP ${response.status}`, 'error');
        return;
    }

    if (deletingCurrent) {
        localStorage.removeItem('research_agent_session_id');
        sessionId = '';
    }
    await loadSessions();
}

async function loadSessionMessages(targetSessionId) {
    resetChatBox();
    const response = await apiFetch(`/api/sessions/${targetSessionId}/messages`);
    const data = await response.json();
    for (const message of data.messages || []) {
        appendMessage(message.content, message.role === 'user' ? 'user' : 'system', message.role !== 'user');
    }

    const current = sessions.find(session => session.session_id === targetSessionId);
    currentSessionTitle.textContent = current?.title || 'Academic Research Copilot';
}

async function sendMessage() {
    const text = userInput.value.trim();
    const files = Array.from(fileInput.files || []);
    if (!text && !files.length) {
        return;
    }

    appendMessage(text || `已上传 ${files.length} 个附件`, 'user');
    userInput.value = '';
    fileInput.value = '';
    updateSelectedFiles();
    sendBtn.disabled = true;
    showLoading();

    logBox.innerHTML = '';
    appendLog('发起新任务...', 'highlight');

    let receivedFinal = false;
    try {
        let response;
        if (files.length) {
            const formData = new FormData();
            formData.append('message', text);
            formData.append('session_id', sessionId);
            formData.append('mode', researchMode.value);
            files.forEach(file => formData.append('files', file));
            response = await apiFetch('/api/chat', { method: 'POST', body: formData });
        } else {
            response = await apiFetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: text, session_id: sessionId, mode: researchMode.value })
            });
        }

        if (!response.ok || !response.body) {
            throw new Error(`HTTP ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) {
                break;
            }

            buffer += decoder.decode(value, { stream: true });
            const chunks = buffer.split('\n\n');
            buffer = chunks.pop();

            for (const chunk of chunks) {
                if (!chunk.startsWith('data: ')) {
                    continue;
                }

                const data = JSON.parse(chunk.slice(6));
                if (data.type === 'log') {
                    appendLog(data.content);
                } else if (data.type === 'token') {
                    appendStreamToken(data.content, data.stream_id);
                } else if (data.type === 'final') {
                    receivedFinal = true;
                    await finalizeStreamMessage(data.content);
                    appendLog('任务执行完毕', 'highlight');
                    await loadSessions({ reloadMessages: false });
                }
            }
        }
        if (!receivedFinal) {
            removeLoading();
            discardStreamMessage();
            appendMessage('连接已断开，任务仍在服务器后台执行；完成后刷新当前会话即可查看结果。', 'system');
            appendLog('SSE 已断开，后台任务继续执行并会写入会话历史。', 'highlight');
        }
    } catch (error) {
        removeLoading();
        discardStreamMessage();
        appendMessage(`请求失败: ${error.message}`, 'system');
        appendLog(`连接中断: ${error.message}`, 'error');
    } finally {
        sendBtn.disabled = false;
        userInput.focus();
    }
}

sendBtn.addEventListener('click', sendMessage);
fileInput.addEventListener('change', updateSelectedFiles);
newSessionBtn.addEventListener('click', createNewSession);

userInput.addEventListener('keydown', event => {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
});

loadSessions().catch(error => {
    resetChatBox();
    appendLog(`会话加载失败: ${error.message}`, 'error');
});
