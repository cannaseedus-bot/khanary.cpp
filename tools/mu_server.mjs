#!/usr/bin/env node
// OpenAI-compatible /v1/chat/completions server for CHAT-µ and TOOL-µ.
// Uses node-llama-cpp — no API token required.
// Keeps a persistent context and queues requests to avoid KV-cache re-allocation.
// Usage: node mu_server.mjs --model <path.gguf> --port <port> [--ctx-size 2048] [--threads N]

import { getLlama, LlamaChatSession } from 'node-llama-cpp';
import http from 'http';
import { parseArgs } from 'util';

const { values: args } = parseArgs({
    options: {
        model:      { type: 'string' },
        port:       { type: 'string', default: '17889' },
        'ctx-size': { type: 'string', default: '2048' },
        threads:    { type: 'string', default: '0' },
    },
    strict: false,
});

if (!args.model) {
    process.stderr.write('Usage: node mu_server.mjs --model <path.gguf> [--port 17889] [--ctx-size 2048] [--threads N]\n');
    process.exit(1);
}

const PORT      = parseInt(args.port, 10);
const CTX_SIZE  = parseInt(args['ctx-size'], 10);
const THREADS   = parseInt(args.threads, 10);
const MODEL_ID  = args.model.replace(/.*[\\/]/, '').replace(/\.gguf$/i, '');

process.stdout.write(`[mu-server] Loading ${MODEL_ID}...\n`);

const llamaOpts = { gpu: 'auto' };
if (THREADS > 0) llamaOpts.cpuThreads = THREADS;

const llama   = await getLlama(llamaOpts);
const model   = await llama.loadModel({ modelPath: args.model });
const context = await model.createContext({ contextSize: CTX_SIZE });
process.stdout.write(`[mu-server] Context ready (${CTX_SIZE} tokens). Listening on http://127.0.0.1:${PORT}\n`);

// Serialise requests through a simple promise chain so the context is never used concurrently.
let queue = Promise.resolve();

function enqueue(fn) {
    const next = queue.then(fn);
    queue = next.catch(() => {});
    return next;
}

async function complete(body) {
    const messages    = Array.isArray(body.messages) ? body.messages : [];
    const temperature = typeof body.temperature === 'number' ? body.temperature : 0.3;
    const maxTokens   = typeof body.max_tokens   === 'number' ? body.max_tokens  : 512;

    let systemPrompt;
    let userText = '';
    for (const m of messages) {
        if (m.role === 'system')    systemPrompt = String(m.content ?? '');
        else if (m.role === 'user') userText = String(m.content ?? '');
    }
    if (!userText) return '';

    // Reuse the shared context; explicitly dispose the sequence so it returns to the pool.
    const sequence = context.getSequence();
    const session  = new LlamaChatSession({
        contextSequence: sequence,
        systemPrompt,
        autoDisposeSequence: false,
    });

    try {
        return await session.prompt(userText, { temperature, maxTokens });
    } finally {
        sequence.dispose();
    }
}

const server = http.createServer((req, res) => {
    if (req.method === 'GET' && req.url === '/v1/models') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
            object: 'list',
            data: [{ id: MODEL_ID, object: 'model', owned_by: 'local' }],
        }));
        return;
    }

    if (req.method === 'POST' && req.url === '/v1/chat/completions') {
        let raw = '';
        req.on('data', c => { raw += c; });
        req.on('end', () => {
            enqueue(async () => {
                try {
                    const body = JSON.parse(raw);
                    const text = await complete(body);
                    const resp = {
                        id:      `chatcmpl-${Date.now()}`,
                        object:  'chat.completion',
                        created: Math.floor(Date.now() / 1000),
                        model:   body.model ?? MODEL_ID,
                        choices: [{
                            index:         0,
                            message:       { role: 'assistant', content: text },
                            finish_reason: 'stop',
                        }],
                        usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
                    };
                    res.writeHead(200, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify(resp));
                } catch (err) {
                    process.stderr.write(`[mu-server] error: ${err.message}\n`);
                    res.writeHead(500, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({ error: { message: err.message, type: 'server_error' } }));
                }
            });
        });
        return;
    }

    res.writeHead(404);
    res.end();
});

server.listen(PORT, '127.0.0.1');
server.on('error', err => {
    process.stderr.write(`[mu-server] listen error: ${err.message}\n`);
    process.exit(1);
});
