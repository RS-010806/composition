// Post-training on a frozen large pretrained model (browser runner).
//
// Renders the same 6 x 6 colored-shape images as icl_runner.js, embeds them with the
// Gemini API image embedding model (gemini-embedding-2, 768 dims), then trains light linear
// heads (ridge classifiers) on small post-training sets chosen by design, and scores them on
// the 10 compounds that no design ever includes. Requires sessionStorage.__PK (API key).
//
//   await PP.embedAll()        render and embed 784 images (needed once)
//   await PP.runAll(20)        20 seeds per design; summary in PP.summary
(() => {
  const COLORS = ['red', 'orange', 'yellow', 'green', 'blue', 'purple'];
  const RGB = { red: [215, 40, 40], orange: [240, 135, 20], yellow: [225, 205, 30], green: [40, 165, 60], blue: [40, 90, 220], purple: [140, 50, 180] };
  const SHAPES = ['circle', 'square', 'triangle', 'diamond', 'star', 'plus'];
  const N = 6;
  const MODEL = 'gemini-embedding-2', DIM = 768;
  // Same held-out compounds and cross as composition/shapes.py
  const HELD = [[1, 2], [1, 3], [2, 3], [2, 4], [3, 4], [3, 5], [4, 1], [4, 5], [5, 1], [5, 2]];
  const isHeld = (c, s) => HELD.some(([a, b]) => a === c && b === s);
  const ALL = []; for (let c = 0; c < N; c++) for (let s = 0; s < N; s++) ALL.push([c, s]);
  const SEEN = ALL.filter(([c, s]) => !isHeld(c, s));
  const CROSS = ALL.filter(([c, s]) => c === 0 || s === 0);
  const DIAG = [...Array(N).keys()].map(c => [c, c]);   // each color with its "home" shape
  const TRAIN_PER = 18, EVAL_PER = 6, HELD_EVAL = 16;

  function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  const randint = (rng, n) => Math.floor(rng() * n);
  const shuffle = (rng, arr) => { const a = arr.slice(); for (let i = a.length - 1; i > 0; i--) { const j = randint(rng, i + 1); [a[i], a[j]] = [a[j], a[i]]; } return a; };

  // Renderer identical to icl_runner.js
  function polygon(shape, cx, cy, r, rot) {
    const R = (pts) => pts.map(([x, y]) => [cx + x * Math.cos(rot) - y * Math.sin(rot), cy + x * Math.sin(rot) + y * Math.cos(rot)]);
    if (shape === 'square') { const a = r * 0.82; return R([[-a, -a], [a, -a], [a, a], [-a, a]]); }
    if (shape === 'diamond') return R([[0, -r], [r * 0.7, 0], [0, r], [-r * 0.7, 0]]);
    if (shape === 'triangle') return R([-Math.PI / 2, Math.PI / 6, 5 * Math.PI / 6].map(t => [r * Math.cos(t), r * Math.sin(t)]));
    if (shape === 'star') { const p = []; for (let k = 0; k < 10; k++) { const t = -Math.PI / 2 + k * Math.PI / 5, rr = k % 2 ? r * 0.45 : r; p.push([rr * Math.cos(t), rr * Math.sin(t)]); } return R(p); }
    if (shape === 'plus') { const a = r, b = r * 0.34; return R([[-b, -a], [b, -a], [b, -b], [a, -b], [a, b], [b, b], [b, a], [-b, a], [-b, b], [-a, b], [-a, -b], [-b, -b]]); }
    throw new Error(shape);
  }
  function render(c, s, rng) {
    const W = 128, cv = document.createElement('canvas'); cv.width = W; cv.height = W;
    const g = cv.getContext('2d'); const bg = 200 + randint(rng, 46);
    g.fillStyle = `rgb(${bg},${bg},${bg})`; g.fillRect(0, 0, W, W);
    const r = 26 + rng() * 12, cx = r + 4 + rng() * (W - 2 * r - 8), cy = r + 4 + rng() * (W - 2 * r - 8);
    const f = 0.85 + rng() * 0.23;
    const [r0, g0, b0] = RGB[COLORS[c]].map(v => Math.max(0, Math.min(255, Math.round(v * f + (rng() - 0.5) * 16))));
    g.fillStyle = `rgb(${r0},${g0},${b0})`;
    const shape = SHAPES[s];
    if (shape === 'circle') { g.beginPath(); g.arc(cx, cy, r, 0, 2 * Math.PI); g.fill(); }
    else {
      const rot = (shape === 'square' || shape === 'diamond') ? 0 : (rng() - 0.5) * 0.6;
      const p = polygon(shape, cx, cy, r, rot);
      g.beginPath(); g.moveTo(p[0][0], p[0][1]); for (const [x, y] of p.slice(1)) g.lineTo(x, y); g.closePath(); g.fill();
    }
    const im = g.getImageData(0, 0, W, W), d = im.data;
    for (let i = 0; i < d.length; i += 4) { const n = (rng() - 0.5) * 10; d[i] += n; d[i + 1] += n; d[i + 2] += n; }
    g.putImageData(im, 0, 0);
    return cv.toDataURL('image/png').split(',')[1];
  }

  const PP = window.PP = window.PP || { emb: {}, log: [] };
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const k = (c, s) => c + ',' + s;

  PP.embedAll = async () => {
    const KEY = sessionStorage.getItem('__PK');
    const items = [];
    for (const [c, s] of ALL) {
      const n = isHeld(c, s) ? HELD_EVAL : TRAIN_PER + EVAL_PER;
      const rng = mulberry32(90001 + 101 * c + 7 * s);
      for (let i = 0; i < n; i++) items.push({ c, s, i, data: render(c, s, rng) });
    }
    for (let b = 0; b < items.length; b += 25) {
      const batch = items.slice(b, b + 25).filter(it => !(PP.emb[k(it.c, it.s)] && PP.emb[k(it.c, it.s)][it.i]));
      if (!batch.length) continue;
      const body = { requests: batch.map(it => ({ model: 'models/' + MODEL, content: { parts: [{ inline_data: { mime_type: 'image/png', data: it.data } }] }, outputDimensionality: DIM })) };
      for (let attempt = 1; attempt <= 10; attempt++) {
        const r = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${MODEL}:batchEmbedContents`, { method: 'POST', headers: { 'Content-Type': 'application/json', 'x-goog-api-key': KEY }, body: JSON.stringify(body) });
        const j = await r.json();
        if (r.ok && j.embeddings) {
          batch.forEach((it, q) => { (PP.emb[k(it.c, it.s)] = PP.emb[k(it.c, it.s)] || [])[it.i] = Float32Array.from(j.embeddings[q].values); });
          break;
        }
        PP.log.push({ t: Date.now(), b, attempt, status: r.status, err: j.error && j.error.message.slice(0, 120) });
        await sleep(r.status === 429 ? 20000 : 3000 * attempt);
      }
    }
    return Object.values(PP.emb).reduce((a, v) => a + v.filter(Boolean).length, 0);
  };

  // Ridge classifier in dual form: alpha = (K + lambda I)^-1 Y, scores = X_test X^T alpha.
  function cholSolve(A, n, B, m) {   // A: n x n (Float64, row-major, overwritten), B: n x m
    for (let j = 0; j < n; j++) {
      let s = A[j * n + j];
      for (let p = 0; p < j; p++) s -= A[j * n + p] * A[j * n + p];
      const d = Math.sqrt(s); A[j * n + j] = d;
      for (let i = j + 1; i < n; i++) { let t = A[i * n + j]; for (let p = 0; p < j; p++) t -= A[i * n + p] * A[j * n + p]; A[i * n + j] = t / d; }
    }
    for (let col = 0; col < m; col++) {
      const y = new Float64Array(n);
      for (let i = 0; i < n; i++) { let t = B[i * m + col]; for (let p = 0; p < i; p++) t -= A[i * n + p] * y[p]; y[i] = t / A[i * n + i]; }
      for (let i = n - 1; i >= 0; i--) { let t = y[i]; for (let p = i + 1; p < n; p++) t -= A[p * n + i] * y[p]; y[i] = t / A[i * n + i]; }
      for (let i = 0; i < n; i++) B[i * m + col] = y[i];
    }
    return B;
  }
  function ridge(train, heads) {   // train: [{x, c, s}]; heads: {name: [labelFn, nClasses]}
    const n = train.length, mean = new Float64Array(DIM);
    for (const t of train) for (let q = 0; q < DIM; q++) mean[q] += t.x[q] / n;
    const X = train.map(t => Float64Array.from(t.x, (v, q) => v - mean[q]));
    const K = new Float64Array(n * n);
    for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) { let s = 0; const a = X[i], b = X[j]; for (let q = 0; q < DIM; q++) s += a[q] * b[q]; K[i * n + j] = s; K[j * n + i] = s; }
    let tr = 0; for (let i = 0; i < n; i++) tr += K[i * n + i];
    const lam = 0.1 * tr / n;
    const out = { mean, X, alphas: {} };
    for (const [name, [labelFn, m]] of Object.entries(heads)) {
      const A = Float64Array.from(K); for (let i = 0; i < n; i++) A[i * n + i] += lam;
      const Y = new Float64Array(n * m); train.forEach((t, i) => { Y[i * m + labelFn(t)] = 1; });
      out.alphas[name] = { alpha: cholSolve(A, n, Y, m), m };
    }
    return out;
  }
  function predict(model, x, name) {
    const { alpha, m } = model.alphas[name], n = model.X.length;
    const xc = Float64Array.from(x, (v, q) => v - model.mean[q]);
    const kx = new Float64Array(n); for (let i = 0; i < n; i++) { let s = 0; const a = model.X[i]; for (let q = 0; q < DIM; q++) s += a[q] * xc[q]; kx[i] = s; }
    let best = -1, bv = -Infinity;
    for (let col = 0; col < m; col++) { let s = 0; for (let i = 0; i < n; i++) s += kx[i] * alpha[i * m + col]; if (s > bv) { bv = s; best = col; } }
    return best;
  }

  function trainSet(design, rng) {
    const pickRenders = (c, s, n) => shuffle(rng, [...Array(TRAIN_PER).keys()]).slice(0, n).map(i => ({ x: PP.emb[k(c, s)][i], c, s }));
    if (design === 'cross') return CROSS.flatMap(([c, s]) => pickRenders(c, s, 10));
    if (design === 'random11') return shuffle(rng, SEEN).slice(0, 11).flatMap(([c, s]) => pickRenders(c, s, 10));
    if (design === 'confounded') return shuffle(rng, DIAG.flatMap(([c, s]) => pickRenders(c, s, TRAIN_PER))).slice(0, 110);
    if (design === 'all26') return shuffle(rng, SEEN.flatMap(([c, s]) => pickRenders(c, s, TRAIN_PER))).slice(0, 110);
    if (design === 'all26_full') return SEEN.flatMap(([c, s]) => pickRenders(c, s, TRAIN_PER));
    throw new Error(design);
  }

  function evaluate(model, joint) {
    const res = { held: { n: 0, compound: 0, color: 0, shape: 0, shortcut: 0 }, seen: { n: 0, compound: 0 } };
    for (const [c, s] of ALL) {
      const held = isHeld(c, s), from = held ? 0 : TRAIN_PER, to = held ? HELD_EVAL : TRAIN_PER + EVAL_PER;
      for (let i = from; i < to; i++) {
        const x = PP.emb[k(c, s)][i];
        let pc, ps;
        if (joint) { const j = predict(model, x, 'joint'); pc = Math.floor(j / N); ps = j % N; }
        else { pc = predict(model, x, 'color'); ps = predict(model, x, 'shape'); }
        const r = held ? res.held : res.seen;
        r.n++; r.compound += (pc === c && ps === s) ? 1 : 0;
        if (held) { r.color += pc === c ? 1 : 0; r.shape += ps === s ? 1 : 0; r.shortcut += ps === c ? 1 : 0; }
      }
    }
    for (const r of [res.held, res.seen]) for (const key of Object.keys(r)) if (key !== 'n') r[key] = r[key] / r.n;
    return res;
  }

  PP.runAll = async (seeds = 20) => {
    const designs = ['cross', 'confounded', 'random11', 'all26', 'all26_full'];
    PP.runs = [];
    for (let seed = 0; seed < seeds; seed++) {
      for (const design of designs) {
        const rng = mulberry32(5000 + 97 * seed + designs.indexOf(design));
        const train = trainSet(design, rng);
        const fact = ridge(train, { color: [t => t.c, N], shape: [t => t.s, N] });
        PP.runs.push({ seed, design, head: 'factorized', n: train.length, compounds: new Set(train.map(t => k(t.c, t.s))).size, ...flat(evaluate(fact, false)) });
        if (design === 'cross' || design === 'all26_full') {
          const joint = ridge(train, { joint: [t => t.c * N + t.s, N * N] });
          PP.runs.push({ seed, design, head: 'joint', n: train.length, compounds: new Set(train.map(t => k(t.c, t.s))).size, ...flat(evaluate(joint, true)) });
        }
        await sleep(0);
      }
    }
    const groups = {};
    for (const r of PP.runs) { const g = r.design + '|' + r.head; (groups[g] = groups[g] || []).push(r); }
    PP.summary = Object.fromEntries(Object.entries(groups).map(([g, rs]) => {
      const mean = key => rs.reduce((a, r) => a + r[key], 0) / rs.length;
      const sd = key => { const m = mean(key); return Math.sqrt(rs.reduce((a, r) => a + (r[key] - m) ** 2, 0) / (rs.length - 1)); };
      return [g, { seeds: rs.length, n_train: rs[0].n, compounds: rs[0].compounds, held_compound: mean('held_compound'), held_compound_sd: sd('held_compound'), held_color: mean('held_color'), held_shape: mean('held_shape'), held_shortcut: mean('held_shortcut'), seen_compound: mean('seen_compound') }];
    }));
    return PP.summary;
  };
  const flat = (res) => ({ held_compound: res.held.compound, held_color: res.held.color, held_shape: res.held.shape, held_shortcut: res.held.shortcut, seen_compound: res.seen.compound });
  return 'post-training probe ready';
})();
