// No dependency install: arguments are the TypeScript module path and patched Studio UI root.
// These checks execute the staged TSX with mocked UI/runtime callbacks, never hardware.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');
const ts = require(process.argv[2] || 'typescript');
if (!process.argv[3]) throw new Error('Usage: node test-recording-controls.cjs TYPESCRIPT_MODULE STUDIO_UI_ROOT');
const uiRoot = path.resolve(process.argv[3]);
const viewerSource = fs.readFileSync(path.join(uiRoot, 'src/routes/datasets/record/recording-viewer.tsx'), 'utf8');
const providerSource = fs.readFileSync(path.join(uiRoot, 'src/features/robots/runtime-session-provider.tsx'), 'utf8');
const compile = (source) => {
    const result = ts.transpileModule(source, {
        compilerOptions: { jsx: ts.JsxEmit.ReactJSX, module: ts.ModuleKind.CommonJS },
        reportDiagnostics: true,
    });
    assert.deepEqual(result.diagnostics, []);
    return result.outputText;
};

function render(overrides = {}) {
    const calls = [];
    const effects = [];
    const listeners = {};
    const mutation = (name) => ({ isPending: false, mutate: (...args) => calls.push([name, ...args]) });
    const session = {
        dataset: { default_task: 'Move object', id: 'dataset', project_id: 'project' },
        state: { connected: true, dataset_loaded: true, is_recording: false, follower_source: 'teleop' },
        readyForRecording: true,
        setFollowerSource: mutation('source'),
        startEpisode: mutation('start'),
        saveEpisode: mutation('save'),
        discardEpisode: mutation('discard'),
        observation: { current: {} },
        actions: { current: {} },
        ...overrides,
    };
    session.state = {
        connected: true, dataset_loaded: true, is_recording: false, follower_source: 'teleop', ...overrides.state,
    };
    const jsx = (type, props) => ({ type, props });
    const context = {
        exports: {},
        window: {
            addEventListener: (name, fn) => { listeners[name] = fn; },
            removeEventListener: () => {},
        },
        require: (name) => {
            if (name === 'react') return { useState: (value) => [value, () => {}], useEffect: (fn) => effects.push(fn) };
            if (name === 'react/jsx-runtime') return { jsx, jsxs: jsx };
            if (name === '@geti-ui/ui') return new Proxy({}, { get: (_, key) => key });
            if (name.endsWith('runtime-session-provider')) return { useRuntimeSession: () => session };
            if (name.endsWith('robot-control-view')) return { RobotControlView: 'RobotControlView' };
            if (name.endsWith('robot-models-context')) return { RobotModelsProvider: 'RobotModelsProvider' };
            if (name.endsWith('router')) return { paths: { project: { datasets: { show: () => '/dataset' } } } };
            if (name.endsWith('.css')) return { default: {} };
            throw new Error(`Unexpected import ${name}`);
        },
    };
    vm.runInNewContext(compile(viewerSource), context);
    const tree = context.exports.RecordingViewer();
    effects.forEach((effect) => effect());
    const nodes = [];
    const visit = (node) => {
        if (Array.isArray(node)) return node.forEach(visit);
        if (!node || typeof node !== 'object') return;
        nodes.push(node);
        visit(node.props?.children);
    };
    visit(tree);
    const text = (node) => typeof node === 'string' ? node : Array.isArray(node)
        ? node.map(text).join('') : node?.props ? text(node.props.children) : '';
    const button = (label) => nodes.find((node) => node.type === 'Button' && text(node) === label);
    return { calls, session, nodes, button, text: text(tree), key: (key) => listeners.keydown({ key }) };
}

test('connected initialization offers Pause and sends hold only when pressed', () => {
    const view = render({ readyForRecording: false, state: { dataset_loaded: false } });
    assert.deepEqual(view.calls, []);
    assert.equal(view.button('Pause following').props.isDisabled, false);
    view.button('Pause following').props.onPress();
    assert.deepEqual(view.calls, [['source', 'hold']]);
    view.key('ArrowRight');
    assert.equal(view.calls.length, 1);
});

test('holding blocks Start button, form submit, and hotkey; resume requires a click', () => {
    const view = render({ state: { follower_source: 'hold' } });
    assert.match(view.text, /Holding position \(torque on\)/);
    assert.match(view.text, /not an emergency stop/);
    assert.equal(view.button('Start episode→').props.isDisabled, true);
    assert.deepEqual(view.calls, []);
    view.key('ArrowRight');
    view.nodes.find((node) => node.type === 'Form').props.onSubmit({ preventDefault() {} });
    assert.deepEqual(view.calls, []);
    view.button('Resume following').props.onPress();
    assert.deepEqual(view.calls, [['source', 'teleop']]);
});

test('initialization cannot resume until ready, disconnected controls are disabled', () => {
    assert.equal(render({ readyForRecording: false, state: { follower_source: 'hold' } })
        .button('Resume following').props.isDisabled, true);
    const disconnected = render({ readyForRecording: false, state: { connected: false } });
    assert.equal(disconnected.button('Pause following').props.isDisabled, true);
    assert.match(disconnected.text, /Environment disconnected/);
});

test('paused active episode can Accept by click or hotkey and is not discarded automatically', () => {
    const view = render({ state: { follower_source: 'hold', is_recording: true } });
    assert.deepEqual(view.calls, []);
    view.button('Accept→').props.onPress();
    view.key('ArrowRight');
    assert.deepEqual(view.calls, [['save'], ['save']]);
});

test('following enables Start and hotkey without sending motion source commands', () => {
    const view = render();
    assert.match(view.text, /Following leader/);
    assert.equal(view.button('Start episode→').props.isDisabled, false);
    view.key('ArrowRight');
    assert.deepEqual(view.calls, [['start', 'Move object']]);
});

test('pending follower command blocks episode start and another source command', () => {
    const view = render({ setFollowerSource: { isPending: true, mutate: () => assert.fail('unexpected source') } });
    assert.equal(view.button('Pause following').props.isDisabled, true);
    assert.equal(view.button('Start episode→').props.isDisabled, true);
    view.key('ArrowRight');
    assert.deepEqual(view.calls, []);
});

function runOpen(props, count) {
    const file = ts.createSourceFile('provider.tsx', providerSource, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
    let initializer;
    const visit = (node) => {
        if (ts.isVariableDeclaration(node) && node.name.getText(file) === 'onOpen') initializer = node.initializer;
        ts.forEachChild(node, visit);
    };
    visit(file);
    assert.ok(initializer, 'provider retains an onOpen callback');
    const calls = [];
    const context = {
        props, devices: {},
        socket: { sendJsonMessage: () => calls.push(['handshake']) },
        loadDataset: { mutate: (dataset) => calls.push(['dataset', dataset]) },
        loadModel: { mutate: (model) => calls.push(['model', model]) },
        setFollowerSource: { mutate: (source) => calls.push(['source', source]) },
    };
    vm.runInNewContext(compile(`const onOpen = ${initializer.getText(file)};\nfor (let i=0; i<${count}; i++) onOpen();`), context);
    return calls;
}

test('recording open and reconnect always request hold, never teleop', () => {
    const dataset = { id: 'dataset' };
    assert.deepEqual(runOpen({ dataset }, 2), [
        ['handshake'], ['dataset', dataset], ['source', 'hold'],
        ['handshake'], ['dataset', dataset], ['source', 'hold'],
    ]);
});

test('inference open still loads model without adding follower-source commands', () => {
    const calls = runOpen({ model: 'model', inferenceDevice: 'CPU' }, 1);
    assert.equal(calls[0][0], 'handshake');
    assert.equal(calls[1][0], 'model');
    assert.equal(calls[1][1].model, 'model');
    assert.equal(calls[1][1].inference_device, 'CPU');
    assert.equal(calls.length, 2);
});
