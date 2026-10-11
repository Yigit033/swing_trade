// Offline rendered-component and API transport contracts; not browser E2E.
// Run: node --test integration-audit.test.cjs
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const ts = require('typescript');
const React = require('react');
const { renderToStaticMarkup } = require('react-dom/server');

const empty = () => null;
const icons = new Proxy({}, { get: () => empty });
function load(relative, mocks = {}, expose = '') {
    const filename = path.join(__dirname, relative);
    const compiled = ts.transpileModule(fs.readFileSync(filename, 'utf8') + expose, {
        compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX,
            target: ts.ScriptTarget.ES2020, esModuleInterop: true },
    }).outputText;
    const module = { exports: {} };
    const imports = { 'lucide-react': icons, recharts: icons, '@/lib/api': {}, ...mocks };
    new Function('require', 'module', 'exports', compiled)(
        name => Object.hasOwn(imports, name) ? imports[name] : require(name), module, module.exports);
    return module.exports;
}
const invalidators = { invalidateTrades: empty, invalidatePerformance: empty };
function renderPage(name, hooks) {
    const Page = load(`src/app/${name}/page.tsx`, {
        '@/hooks/useApi': { useInvalidateQueries: () => invalidators, ...hooks },
    }).default;
    return renderToStaticMarkup(React.createElement(Page));
}
function tradeHooks(trades, overrides = {}) {
    return {
        useTrades: () => ({ data: trades, isLoading: false, isError: false, refetch: empty }),
        useTradesLastUpdate: () => ({}),
        useUpdatePrices: () => ({ mutate: empty }),
        ...overrides,
    };
}

test('Trades and Performance API failures render alerts, not empty-result claims', () => {
    const trades = renderPage('trades', tradeHooks([], {
        useTrades: () => ({ isError: true, refetch: empty }),
    }));
    const performance = renderPage('performance', {
        usePerformance: () => ({ isError: true, refetch: empty }),
    });
    assert.match(trades, /role="alert"/);
    assert.match(performance, /role="alert"/);
    assert.doesNotMatch(performance, /No closed trades yet/);
    assert.doesNotMatch(trades, /0 total/);
});

test('Partial trade uses backend PnL and never invents full-size fallback', () => {
    const row = { id: 1, ticker: 'AAA', status: 'OPEN', entry_price: 100,
        current_price: 105, position_size: 100, partial_exit_price: 110, partial_exit_pct: 33 };
    const unknown = renderPage('trades', tradeHooks([row]));
    assert.doesNotMatch(unknown, /\+\$500\.00/);
    const known = renderPage('trades', tradeHooks([{ ...row, unrealized_pnl: 665, unrealized_pnl_pct: 6.65 }]));
    assert.match(known, /\+\$665\.00/);
    assert.match(known, /\+6\.65%/);
});

test('Execution REVIEW_REQUIRED is visible after price refresh', () => {
    const html = renderPage('trades', tradeHooks([], {
        useUpdatePrices: () => ({ mutate: empty, data: { trades: [
            { id: 1, ticker: 'AAA', execution_warning: 'REVIEW_REQUIRED: INTRADAY_ENTRY' },
        ] } }),
    }));
    assert.match(html, /REVIEW_REQUIRED: INTRADAY_ENTRY/);
    assert.match(html, /role="alert"/);
});

test('Performance preserves zero best return and labels limited non-portfolio results', () => {
    const html = renderPage('performance', { usePerformance: () => ({ data: {
        recent_closed: [
            { id: 1, ticker: 'ZERO', status: 'MANUAL', realized_pnl: 0, realized_pnl_pct: 0 },
            { id: 2, ticker: 'LOSS', status: 'STOPPED', realized_pnl: -100, realized_pnl_pct: -5 },
            { id: 3, ticker: 'REJECT', status: 'REJECTED', realized_pnl: 999, realized_pnl_pct: 99 },
        ],
    } }) });
    assert.match(html, /Best Trade[\s\S]*?ZERO/);
    assert.doesNotMatch(html, /REJECT/);
    assert.match(html, /portföy getirisi değil/);
    assert.match(html, /son 30/);
});

function findNode(node, predicate) {
    if (!node || typeof node !== 'object') return null;
    if (predicate(node)) return node;
    for (const child of React.Children.toArray(node.props?.children)) {
        const found = findNode(child, predicate);
        if (found) return found;
    }
    return null;
}

test('Lookup passes bar date, portfolio, ATR, size and hold period into tracking', async () => {
    const result = { ticker: 'AAA', swing_ready: true, data_as_of: '2026-10-09',
        entry_price: 100, stop_loss: 90, target_1: 110, target_2: 120,
        atr: 4.25, position_size: 22, hold_days: [2, 15], quality_score: 81 };
    const states = ['', 25000, false, [result], '', null, true, ''];
    let index = 0, sent;
    const fakeReact = { ...React, useState: () => [states[index++], empty] };
    const Page = load('src/app/lookup/page.tsx', {
        react: fakeReact,
        '@/lib/api': { trackSignal: async (...args) => { sent = args; return { status: 'added' }; } },
    }).default;
    const tree = Page();
    const card = findNode(tree, n => n.props?.r?.ticker === 'AAA' && typeof n.props.onAdd === 'function');
    assert.ok(card, 'lookup result must have an Add handler');
    await card.props.onAdd(result);
    assert.equal(sent[0].date, '2026-10-09');
    assert.equal(sent[0].atr, 4.25);
    assert.equal(sent[0].position_size, 22);
    assert.equal(sent[0].hold_days_max, 15);
    assert.equal(sent[1], 25000);
});

test('Real trackSignal serializer retains lookup inputs', async () => {
    let sent;
    const client = { interceptors: { request: { use: empty }, response: { use: empty } },
        post: async (url, body) => { sent = { url, body }; return { data: { status: 'added' } }; } };
    const api = load('src/lib/api.ts', { axios: { create: () => client } });
    await api.trackSignal({ ticker: 'AAA', date: '2026-10-09', entry_price: 100,
        stop_loss: 90, target_1: 110, target_2: 120, atr: 4.25,
        position_size: 22, hold_days_max: 15, quality_score: 81 }, 25000);
    assert.equal(sent.url, '/api/scanner/track');
    assert.equal(sent.body.date, '2026-10-09');
    assert.equal(sent.body.portfolio_value, 25000);
    assert.equal(sent.body.atr, 4.25);
    assert.equal(sent.body.hold_days_max, 15);
});

test('Price refresh invalidates Trades, Performance and Pending caches', () => {
    const keys = [];
    const hooks = load('src/hooks/useApi.ts', {
        '@tanstack/react-query': {
            useQueryClient: () => ({ invalidateQueries: ({ queryKey }) => keys.push(queryKey[0]) }),
            useMutation: config => config,
        },
    });
    hooks.useUpdatePrices().onSuccess();
    assert.ok(keys.includes('trades'));
    assert.ok(keys.includes('performance'));
    assert.ok(keys.includes('pending'));
});

test('Scanner renders T1/T2 without inventing a fixed partial percentage', () => {
    const { SignalCard } = load('src/app/scanner/page.tsx', {
        react: { ...React, useState: () => [true, empty] },
        '@/providers/ScannerJobProvider': {},
        'next/link': empty,
    }, '\nexport { SignalCard };');
    const html = renderToStaticMarkup(React.createElement(SignalCard, {
        s: { ticker: 'AAA', quality_score: 81, entry_price: 100, stop_loss: 90,
            target_1: 110, target_2: 120, expected_hold_min: 2, expected_hold_max: 15 },
        onTrack: empty, tracking: false,
    }));
    assert.match(html, /110\.00/);
    assert.match(html, /120\.00/);
    assert.match(html, /ayarlardaki kısmi çıkış oranı/);
    assert.doesNotMatch(html, /%50/);
});

test('Edge renders policy cohort, missing outcomes and overlapping-date warning', () => {
    const side = { n: 0, mean: null, win_rate: null };
    const data = { n_tracked: 1, n_complete: 0, threshold_split: { reference: 80, above: side, below: side },
        policy_split: { eligible: side, excluded: side, floors: { BULL: 78 }, legacy_inferred_rows: 1 },
        matched_edge: { ...side, matched_dates: 2, mean: 1.5 },
        quality_buckets: [], aggregates: {}, signals: [],
    };
    const states = [data, false, false, '', false, 'all', 'date'];
    let index = 0;
    const Page = load('src/app/edge/page.tsx', {
        react: { ...React, useState: () => [states[index++], empty], useEffect: empty,
            useMemo: fn => fn() },
    }).default;
    const html = renderToStaticMarkup(React.createElement(Page));
    assert.match(html, /Politikaya uygun sinyal/);
    assert.match(html, /outcome pencereleri örtüşebilir/);
    assert.doesNotMatch(html, /2 bağımsız tarih/);
    assert.doesNotMatch(html, /NaN|undefined/);
});

test('Settings API error renders an error rather than defaults as saved settings', () => {
    const Page = load('src/app/settings/page.tsx', {
        'next/link': empty,
        '@tanstack/react-query': {
            useQueryClient: () => ({}),
            useQuery: () => ({ isError: true, error: new Error('contract-test-failure'), refetch: empty }),
            useMutation: () => ({}),
        },
    }).default;
    const html = renderToStaticMarkup(React.createElement(Page));
    assert.match(html, /contract-test-failure/);
});
