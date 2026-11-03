// In-context labels as post-training data (browser runner).
//
// Post-training on confounded examples (each color always with the same shape) gives linear
// heads no way to tell color from shape, so they name 0% of new combinations correctly.
// This pilot asks whether the base model, used in context, can supply the missing labels:
//   1. Labeled set: the 6 confounded compounds (c, c), 18 images each, named in made-up words.
//   2. Unlabeled set: images of the other 20 shown compounds (none of the 10 held-out ones).
//   3. Teacher: Gemini 3.5 Flash-Lite sees 11 labeled confounded examples in context and names
//      each unlabeled image. Its names are parsed by vocabulary into (color, shape) labels.
//   4. Student: ridge heads on frozen Gemini image embeddings, trained on the labeled set plus
//      the teacher-labeled images, scored on the 10 held-out compounds (never labeled, never
//      shown to the teacher).
// Run after posttrain_probe.js (needs PP.emb) and icl_runner.js (needs ICL.call).
//
//   await IP.check()            re-render one image and confirm it matches its stored embedding
//   IP.oracle([1, 2])           student trained on true labels for 1 or 2 images per compound
//   await IP.teach(order, per)  query the teacher; order 'adj' (color word first) or 'noun'
//   IP.report()                 teacher accuracy and student accuracy per order
(() => {
  const N = 6, DIM = 768, MODEL = 'gemini-3.5-flash-lite';
  const WORDS = ['dax', 'wug', 'fep', 'lun', 'tiv', 'zor', 'kib', 'mup', 'gol', 'sef', 'nim', 'pav', 'rux', 'bek', 'jom', 'yil'];
  const COLORS = ['red', 'orange', 'yellow', 'green', 'blue', 'purple'];
  const RGB = { red: [215, 40, 40], orange: [240, 135, 20], yellow: [225, 205, 30], green: [40, 165, 60], blue: [40, 90, 220], purple: [140, 50, 180] };
  const SHAPES = ['circle', 'square', 'triangle', 'diamond', 'star', 'plus'];
  const HELD = [[1, 2], [1, 3], [2, 3], [2, 4], [3, 4], [3, 5], [4, 1], [4, 5], [5, 1], [5, 2]];
  const isHeld = (c, s) => HELD.some(([a, b]) => a === c && b === s);
  const ALL = []; for (let c = 0; c < N; c++) for (let s = 0; s < N; s++) ALL.push([c, s]);
  const DIAG = [...Array(N).keys()].map(c => [c, c]);
  const OFF = ALL.filter(([c, s]) => !isHeld(c, s) && c !== s);   // 20 shown, non-confounded compounds
  const TRAIN_PER = 18, HELD_EVAL = 16;
  const k = (c, s) => c + ',' + s;

  function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  const randint = (rng, n) => Math.floor(rng() * n);
  const shuffle = (rng, arr) => { const a = arr.slice(); for (let i = a.length - 1; i > 0; i--) { const j = randint(rng, i + 1); [a[i], a[j]] = [a[j], a[i]]; } return a; };

  // Renderer and image seeds identical to posttrain_probe.js, so image i of compound (c, s)
  // is exactly the image whose embedding is PP.emb[k(c, s)][i].
  function polygon(shape, cx, cy, r, rot) {
    const R = (pts) => pts.map(([x, y]) => [cx + x * Math.cos(rot) - y * Math.sin(rot), cy + x * Math.sin(rot) + y * Math.cos(rot)]);
    if (shape === 'square') { const a = r * 0.82; return R([[-a, -a], [a, -a], [a, a], [-a, a]]); }
    if (shape === 'diamond') return R([[0, -r], [r * 0.7, 0], [0, r], [-r * 0.7, 0]]);
    if (shape === 'triangle') return R([-Math.PI / 2, Math.PI / 6, 5 * Math.PI / 6].map(t => [r * Math.cos(t), r * Math.sin(t)]));
    if (shape === 'star') { const p = []; for (let q = 0; q < 10; q++) { const t = -Math.PI / 2 + q * Math.PI / 5, rr = q % 2 ? r * 0.45 : r; p.push([rr * Math.cos(t), rr * Math.sin(t)]); } return R(p); }
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
  const imgCache = {};
  function image(c, s, i) {
    const key = k(c, s);
    if (!imgCache[key] || imgCache[key].length <= i) {
      const rng = mulberry32(90001 + 101 * c + 7 * s), out = [];
      for (let q = 0; q <= Math.max(i, 2); q++) out.push(render(c, s, rng));
      imgCache[key] = out;
    }
    return imgCache[key][i];
  }

  // Ridge classifier (dual form), identical to posttrain_probe.js.
  function cholSolve(A, n, B, m) {
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
  function ridge(train) {
    const n = train.length, mean = new Float64Array(DIM);
    for (const t of train) for (let q = 0; q < DIM; q++) mean[q] += t.x[q] / n;
    const X = train.map(t => Float64Array.from(t.x, (v, q) => v - mean[q]));
    const K = new Float64Array(n * n);
    for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) { let s = 0; const a = X[i], b = X[j]; for (let q = 0; q < DIM; q++) s += a[q] * b[q]; K[i * n + j] = s; K[j * n + i] = s; }
    let tr = 0; for (let i = 0; i < n; i++) tr += K[i * n + i];
    const lam = 0.1 * tr / n, alphas = {};
    for (const [name, fn] of [['color', t => t.c], ['shape', t => t.s]]) {
      const A = Float64Array.from(K); for (let i = 0; i < n; i++) A[i * n + i] += lam;
      const Y = new Float64Array(n * N); train.forEach((t, i) => { Y[i * N + fn(t)] = 1; });
      alphas[name] = cholSolve(A, n, Y, N);
    }
    return { mean, X, alphas };
  }
  function predict(model, x, name) {
    const alpha = model.alphas[name], n = model.X.length;
    const xc = Float64Array.from(x, (v, q) => v - model.mean[q]);
    const kx = new Float64Array(n); for (let i = 0; i < n; i++) { let s = 0; const a = model.X[i]; for (let q = 0; q < DIM; q++) s += a[q] * xc[q]; kx[i] = s; }
    let best = -1, bv = -Infinity;
    for (let col = 0; col < N; col++) { let s = 0; for (let i = 0; i < n; i++) s += kx[i] * alpha[i * N + col]; if (s > bv) { bv = s; best = col; } }
    return best;
  }
  function heldAccuracy(model) {
    let n = 0, right = 0, color = 0, shape = 0, swapped = 0;
    for (const [c, s] of HELD) for (let i = 0; i < HELD_EVAL; i++) {
      const x = PP.emb[k(c, s)][i], pc = predict(model, x, 'color'), ps = predict(model, x, 'shape');
      n++; right += (pc === c && ps === s); color += pc === c; shape += ps === s; swapped += (pc === s && ps === c);
    }
    return { compound: right / n, color: color / n, shape: shape / n, swapped: swapped / n, n };
  }

  const IP = window.IP = window.IP || { queries: [] };
  const labeled = () => DIAG.flatMap(([c, s]) => [...Array(TRAIN_PER).keys()].map(i => ({ x: PP.emb[k(c, s)][i], c, s })));
  const words = shuffle(mulberry32(424242), WORDS).slice(0, 2 * N);
  IP.cw = words.slice(0, N); IP.sw = words.slice(N);
  const name = (c, s, order) => order === 'adj' ? `${IP.cw[c]} ${IP.sw[s]}` : `${IP.sw[s]} ${IP.cw[c]}`;

  IP.check = async () => {
    const KEY = sessionStorage.getItem('__PK');
    const body = { requests: [{ model: 'models/gemini-embedding-2', content: { parts: [{ inline_data: { mime_type: 'image/png', data: image(0, 1, 0) } }] }, outputDimensionality: DIM }] };
    const r = await fetch('https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:batchEmbedContents', { method: 'POST', headers: { 'Content-Type': 'application/json', 'x-goog-api-key': KEY }, body: JSON.stringify(body) });
    const j = await r.json();
    if (!r.ok) return { status: r.status, err: j.error && j.error.message };
    const a = j.embeddings[0].values, b = PP.emb[k(0, 1)][0];
    let dot = 0, na = 0, nb = 0; for (let q = 0; q < DIM; q++) { dot += a[q] * b[q]; na += a[q] * a[q]; nb += b[q] * b[q]; }
    return { cosine: dot / Math.sqrt(na * nb) };
  };

  IP.oracle = (perList = [1, 2]) => {
    const out = { labeled_only: heldAccuracy(ridge(labeled())) };
    for (const per of perList) out['true_labels_' + per] = heldAccuracy(ridge(labeled().concat(OFF.flatMap(([c, s]) => [...Array(per).keys()].map(i => ({ x: PP.emb[k(c, s)][i], c, s })))))));
    return out;
  };

  // One teacher query: 11 confounded labeled examples (all 6 compounds, 5 shown twice) plus one
  // unlabeled image. Demo images are labeled-set renders 2..17, so they never repeat a query image.
  function query(c, s, i, order) {
    const rng = mulberry32(777 + 1009 * c + 97 * s + 13 * i + (order === 'adj' ? 0 : 50000));
    const repeats = shuffle(rng, DIAG).slice(0, N - 1);
    const demos = shuffle(rng, DIAG.concat(repeats)).map(([dc, ds]) => ({ c: dc, s: ds, i: 2 + randint(rng, TRAIN_PER - 2) }));
    return { binding: false, images: demos.map(d => image(d.c, d.s, d.i)), labels: demos.map(d => name(d.c, d.s, order)), testImage: image(c, s, i), demos };
  }
  function parse(text) {
    const toks = (text.toLowerCase().match(/[a-z]+/g) || []);
    const cwd = toks.find(t => IP.cw.includes(t)), swd = toks.find(t => IP.sw.includes(t));
    return { color: cwd === undefined ? null : IP.cw.indexOf(cwd), shape: swd === undefined ? null : IP.sw.indexOf(swd) };
  }

  IP.teach = async (order, per = 1, conc = 2) => {
    const items = OFF.flatMap(([c, s]) => [...Array(per).keys()].map(i => ({ c, s, i })));
    const todo = items.filter(it => !IP.queries.find(q => q.order === order && q.c === it.c && q.s === it.s && q.i === it.i && q.ok));
    const sleep = ms => new Promise(r => setTimeout(r, ms));
    const workerFn = async () => {
      for (;;) {
        const it = todo.shift(); if (!it) return;
        const t = query(it.c, it.s, it.i, order);
        const res = await ICL.call(MODEL, t);
        const rec = { order, c: it.c, s: it.s, i: it.i, ok: res.ok, ts: new Date().toISOString() };
        if (res.ok) { const p = parse(res.text); Object.assign(rec, { raw: res.text.replace(/\s+/g, ' ').trim().slice(0, 60), pc: p.color, ps: p.shape, right: p.color === it.c && p.shape === it.s, swapped: p.color === it.s && p.shape === it.c, modelVersion: res.modelVersion }); }
        else rec.error = res.error;
        IP.queries.push(rec);
        await sleep(1500);
      }
    };
    await Promise.all([...Array(conc)].map(workerFn));
    return IP.report();
  };

  IP.report = () => {
    const out = { labeled_only: heldAccuracy(ridge(labeled())) };
    for (const order of ['adj', 'noun']) {
      const qs = IP.queries.filter(q => q.order === order && q.ok);
      if (!qs.length) continue;
      const usable = qs.filter(q => q.pc !== null && q.ps !== null);
      const teacher = { n: qs.length, usable: usable.length, right: qs.filter(q => q.right).length, swapped: qs.filter(q => q.swapped).length };
      const student = heldAccuracy(ridge(labeled().concat(usable.map(q => ({ x: PP.emb[k(q.c, q.s)][q.i], c: q.pc, s: q.ps })))));
      const oracle = heldAccuracy(ridge(labeled().concat(qs.map(q => ({ x: PP.emb[k(q.c, q.s)][q.i], c: q.c, s: q.s })))));
      out[order] = { teacher, student, oracle_same_images: oracle };
    }
    return out;
  };
  return 'in-context-to-post-training pilot ready: words ' + IP.cw.join(' ') + ' | ' + IP.sw.join(' ');
})();
