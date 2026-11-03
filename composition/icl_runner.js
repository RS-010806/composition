// In-context compositional generalization with invented words (browser runner).
//
// Paste into the DevTools console of any https page (the released data was collected on
// aistudio.google.com), after setting sessionStorage.__PK to a Gemini API key. It renders
// synthetic colored-shape images with <canvas>, sends them as inline PNGs to the Gemini API
// `generateContent` endpoint, and stores one record per trial in window.ICL.records.
// Trials are seeded, so every model sees identical demos and test images.
//
//   ICL.add(model, design, order, binding, n)   queue n trials
//   ICL.start()                                  run with per-model pacing
//   ICL.records                                  results (images are not stored)
(() => {
  const COLORS = ['red', 'orange', 'yellow', 'green', 'blue', 'purple'];
  const RGB = { red: [215, 40, 40], orange: [240, 135, 20], yellow: [225, 205, 30], green: [40, 165, 60], blue: [40, 90, 220], purple: [140, 50, 180] };
  const SHAPES = ['circle', 'square', 'triangle', 'diamond', 'star', 'plus'];
  const WORDS = ['dax', 'wug', 'fep', 'lun', 'tiv', 'zor', 'kib', 'mup', 'gol', 'sef', 'nim', 'pav', 'rux', 'bek', 'jom', 'yil'];
  const N = 6;
  const API = 'https://generativelanguage.googleapis.com/v1beta/models/';

  function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  const randint = (rng, n) => Math.floor(rng() * n);
  const pick = (rng, arr) => arr[randint(rng, arr.length)];
  const shuffle = (rng, arr) => { const a = arr.slice(); for (let i = a.length - 1; i > 0; i--) { const j = randint(rng, i + 1); [a[i], a[j]] = [a[j], a[i]]; } return a; };

  function polygon(shape, cx, cy, r, rot) {
    const R = (pts) => pts.map(([x, y]) => [cx + x * Math.cos(rot) - y * Math.sin(rot), cy + x * Math.sin(rot) + y * Math.cos(rot)]);
    if (shape === 'square') { const a = r * 0.82; return R([[-a, -a], [a, -a], [a, a], [-a, a]]); }
    if (shape === 'diamond') return R([[0, -r], [r * 0.7, 0], [0, r], [-r * 0.7, 0]]);
    if (shape === 'triangle') return R([-Math.PI / 2, Math.PI / 6, 5 * Math.PI / 6].map(t => [r * Math.cos(t), r * Math.sin(t)]));
    if (shape === 'star') { const p = []; for (let k = 0; k < 10; k++) { const t = -Math.PI / 2 + k * Math.PI / 5, rr = k % 2 ? r * 0.45 : r; p.push([rr * Math.cos(t), rr * Math.sin(t)]); } return R(p); }
    if (shape === 'plus') { const a = r, b = r * 0.34; return R([[-b, -a], [b, -a], [b, -b], [a, -b], [a, b], [b, b], [b, a], [-b, a], [-b, b], [-a, b], [-a, -b], [-b, -b]]); }
    throw new Error(shape);
  }

  // objs: [{c, s, cx, cy, r}] on a 128 x 128 canvas. Returns base64 PNG.
  function render(objs, rng) {
    const W = 128, cv = document.createElement('canvas'); cv.width = W; cv.height = W;
    const g = cv.getContext('2d'); const bg = 200 + randint(rng, 46);
    g.fillStyle = `rgb(${bg},${bg},${bg})`; g.fillRect(0, 0, W, W);
    for (const o of objs) {
      const f = 0.85 + rng() * 0.23;
      const [r0, g0, b0] = RGB[COLORS[o.c]].map(v => Math.max(0, Math.min(255, Math.round(v * f + (rng() - 0.5) * 16))));
      g.fillStyle = `rgb(${r0},${g0},${b0})`;
      const shape = SHAPES[o.s];
      if (shape === 'circle') { g.beginPath(); g.arc(o.cx, o.cy, o.r, 0, 2 * Math.PI); g.fill(); continue; }
      const rot = (shape === 'square' || shape === 'diamond') ? 0 : (rng() - 0.5) * 0.6;
      const p = polygon(shape, o.cx, o.cy, o.r, rot);
      g.beginPath(); g.moveTo(p[0][0], p[0][1]); for (const [x, y] of p.slice(1)) g.lineTo(x, y); g.closePath(); g.fill();
    }
    const im = g.getImageData(0, 0, W, W), d = im.data;
    for (let i = 0; i < d.length; i += 4) { const n = (rng() - 0.5) * 10; d[i] += n; d[i + 1] += n; d[i + 2] += n; }
    g.putImageData(im, 0, 0);
    return cv.toDataURL('image/png').split(',')[1];
  }

  const single = (c, s, rng) => { const r = 26 + rng() * 12; return [{ c, s, r, cx: r + 4 + rng() * (128 - 2 * r - 8), cy: r + 4 + rng() * (128 - 2 * r - 8) }]; };
  const pair = (left, right, rng) => { const r = 18 + rng() * 6; return [{ c: left[0], s: left[1], r, cx: 32 + (rng() - 0.5) * 8, cy: 64 + (rng() - 0.5) * 30 }, { c: right[0], s: right[1], r, cx: 96 + (rng() - 0.5) * 8, cy: 64 + (rng() - 0.5) * 30 }]; };
  const ALL = []; for (let c = 0; c < N; c++) for (let s = 0; s < N; s++) ALL.push([c, s]);
  const key = ([c, s]) => c + ',' + s;

  // design: 'cross' | 'confounded' | 'random'; order: 'adj' (color word first) | 'noun' (shape word first)
  function makeTrial(design, order, binding, seed) {
    const rng = mulberry32(seed);
    const words = shuffle(rng, WORDS).slice(0, 2 * N);
    const cw = words.slice(0, N), sw = words.slice(N);
    let demos, test;
    if (design === 'cross') {             // one full row and one full column: 11 demos, deconfounded
      const pc = randint(rng, N), ps = randint(rng, N);
      demos = []; for (let c = 0; c < N; c++) demos.push([c, ps]); for (let s = 0; s < N; s++) if (s !== ps) demos.push([pc, s]);
      test = pick(rng, ALL.filter(([c, s]) => c !== pc && s !== ps));
    } else if (design === 'confounded') { // each color always with its own shape: 11 demos, every word seen, fully confounded
      const home = shuffle(rng, [...Array(N).keys()]);
      const diag = [...Array(N).keys()].map(c => [c, home[c]]);
      demos = diag.concat(shuffle(rng, diag).slice(0, N - 1));
      test = pick(rng, ALL.filter(([c, s]) => home[c] !== s));
    } else if (design === 'random') {     // 11 random compounds; test compound's color and shape both appear
      for (;;) {
        demos = shuffle(rng, ALL).slice(0, 2 * N - 1);
        const ks = new Set(demos.map(key)), cs = new Set(demos.map(d => d[0])), ss = new Set(demos.map(d => d[1]));
        const cand = ALL.filter(x => !ks.has(key(x)) && cs.has(x[0]) && ss.has(x[1]));
        if (cand.length) { test = pick(rng, cand); break; }
      }
    } else throw new Error(design);
    demos = shuffle(rng, demos);
    const label = ([c, s]) => order === 'adj' ? `${cw[c]} ${sw[s]}` : `${sw[s]} ${cw[c]}`;
    let distractor = null, side = null;
    if (binding) { distractor = pick(rng, ALL.filter(([c, s]) => c !== test[0] && s !== test[1])); side = rng() < 0.5 ? 'left' : 'right'; }
    const images = demos.map(d => render(single(d[0], d[1], rng), rng));
    const testImage = binding ? render(side === 'left' ? pair(test, distractor, rng) : pair(distractor, test, rng), rng) : render(single(test[0], test[1], rng), rng);
    return { design, order, binding, seed, cw, sw, demos, test, distractor, side, expected: label(test), labels: demos.map(label), images, testImage };
  }

  function contents(t) {
    const parts = [{ text: 'Here is a small made-up language for naming pictures. Each picture shows an object, and its name is two made-up words. Study the examples, then name the new picture.' }];
    t.images.forEach((img, i) => { parts.push({ text: `Example ${i + 1}:` }, { inline_data: { mime_type: 'image/png', data: img } }, { text: `Name: ${t.labels[i]}` }); });
    if (t.binding) parts.push({ text: 'New picture (it shows two objects):' }, { inline_data: { mime_type: 'image/png', data: t.testImage } }, { text: `What is the name of the object on the ${t.side}? Reply with the two words only.` });
    else parts.push({ text: 'New picture:' }, { inline_data: { mime_type: 'image/png', data: t.testImage } }, { text: 'What is its name? Reply with the two words only.' });
    return [{ role: 'user', parts }];
  }

  function score(t, text) {
    const vocab = new Set(t.cw.concat(t.sw));
    const toks = (text.toLowerCase().match(/[a-z]+/g) || []).filter(w => vocab.has(w)).slice(0, 2);
    const [c, s] = t.test;
    const out = { pred: toks.join(' '), exact: toks.join(' ') === t.expected, colorWord: toks.includes(t.cw[c]), shapeWord: toks.includes(t.sw[s]) };
    if (t.binding) { const [dc, ds] = t.distractor; out.distractorWord = toks.includes(t.cw[dc]) || toks.includes(t.sw[ds]); }
    return out;
  }

  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const ICL = window.ICL = window.ICL || { records: [], jobs: [], active: {}, log: [], conc: {}, pace: {} };
  ICL.makeTrial = makeTrial; ICL.contents = contents; ICL.score = score;

  ICL.call = async (model, t, generationConfig) => {
    const KEY = sessionStorage.getItem('__PK');
    const body = { contents: contents(t) };
    if (generationConfig) body.generationConfig = generationConfig;
    let last = null;
    for (let attempt = 1; attempt <= 10; attempt++) {
      try {
        const r = await fetch(API + model + ':generateContent', { method: 'POST', headers: { 'Content-Type': 'application/json', 'x-goog-api-key': KEY }, body: JSON.stringify(body) });
        const j = await r.json();
        if (r.ok && j.candidates && j.candidates.length) {
          const c = j.candidates[0];
          const text = ((c.content || {}).parts || []).filter(p => !p.thought).map(p => p.text || '').join('');
          return { ok: true, text, modelVersion: j.modelVersion, finishReason: c.finishReason, usage: j.usageMetadata, attempts: attempt };
        }
        last = { status: r.status, err: j.error ? (j.error.status + ': ' + (j.error.message || '')).slice(0, 200) : '' };
        if ([400, 402, 403, 404].includes(r.status)) break;
        ICL.log.push({ t: Date.now(), model, attempt, last });
        await sleep(r.status === 429 ? 20000 + Math.random() * 10000 : 5000 * attempt);
      } catch (e) { last = { fetchError: String(e) }; await sleep(5000 * attempt); }
    }
    return { ok: false, error: last };
  };

  // Seeds depend only on (design, order, binding, i), so all models get identical trials.
  const seedOf = (design, order, binding, i) => 1000003 * ['cross', 'confounded', 'random'].indexOf(design) + 7919 * (order === 'adj' ? 1 : 2) + (binding ? 500000 : 0) + i;
  ICL.add = (model, design, order, binding, n, generationConfig) => {
    for (let i = 0; i < n; i++) {
      const id = [model, design, order, binding ? 'bind' : 'single', i].join('|');
      if (!ICL.jobs.find(j => j.id === id)) ICL.jobs.push({ id, model, design, order, binding, i, seed: seedOf(design, order, binding, i), generationConfig, state: 'pending' });
    }
  };
  ICL.launch = (model) => {
    if ((ICL.active[model] || 0) >= (ICL.conc[model] || 2)) return;
    const job = ICL.jobs.find(j => j.model === model && j.state === 'pending');
    if (!job) return;
    job.state = 'running'; ICL.active[model] = (ICL.active[model] || 0) + 1;
    const t = makeTrial(job.design, job.order, job.binding, job.seed);
    ICL.call(model, t, job.generationConfig).then(res => {
      const rec = { id: job.id, model, design: job.design, order: job.order, binding: job.binding, i: job.i, seed: job.seed, test: t.test, distractor: t.distractor, side: t.side, expected: t.expected, ok: res.ok, ts: new Date().toISOString() };
      if (res.ok) Object.assign(rec, score(t, res.text), { raw: res.text.slice(0, 120), modelVersion: res.modelVersion, finishReason: res.finishReason, promptTokens: res.usage && res.usage.promptTokenCount, outputTokens: res.usage && res.usage.candidatesTokenCount });
      else rec.error = res.error;
      ICL.records.push(rec); job.state = res.ok ? 'done' : 'failed'; ICL.active[model]--;
    });
  };
  ICL.start = () => {
    (ICL.timers || []).forEach(clearInterval);
    const models = [...new Set(ICL.jobs.map(j => j.model))];
    ICL.timers = models.map(m => setInterval(() => ICL.launch(m), ICL.pace[m] || 4500));
  };
  ICL.status = () => { const o = {}; ICL.jobs.forEach(j => { o[j.model] = o[j.model] || {}; o[j.model][j.state] = (o[j.model][j.state] || 0) + 1; }); return o; };
  return 'ICL runner ready';
})();
