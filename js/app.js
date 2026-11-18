import * as THREE from '../vendor/three.module.min.js';
import { OrbitControls } from '../vendor/OrbitControls.js';

// ── DarkSHINE DS Dark Theme Constants ──
const COLORS = {
  bg: 0x121417,
  station: 0x666666,
  truth: 0xeaecef,
  v5: 0x178e89,
  v6: 0x665ae0,
  ds: 0xe68613,
  neither: 0xb85c65,
  bgHit: 0x7a828e,
  otherMc: 0x7a828e,
  fp: 0xd9534f,
  fn: 0xd9a441,
  cand: 0x666666,
};

const HIT_LAYER_HEX = [
  0x3d5a80, // N1
  0x178e89, // N2
  0x4ea89c, // N3
  0x665ae0, // N4
  0xc97b2b, // N5
  0xe68613, // N6
];

const STATION_Z = [7.83, 38.83, 89.83, 120.83, 151.83, 180.33];
const CLOUD_BOX = { xMin: -68, xMax: 46, yMin: -52, yMax: 52 };

// ── App State ──
const state = {
  mode: 'cloud',          // 'cloud' | 'st' (5,491) | 'full' (149,859)
  model: 'v6',            // 'v5' | 'v6'
  cloudLayout: 'split',   // 'split' | 'overlay'
  cloudPairing: 'recall', // 'recall' | 'quad'
  cloudOpacity: 0.16,
  threshold: 0.50,
  showBoth: true,
  showGnnOnly: true,
  showDsOnly: true,
  showNeither: false,
  cloudGeom: 'both',      // 'both' | 'lines' | 'hits'
  cloudCoords: 'truth',   // 'truth' | 'reco'
  momBand: 'all',
  occBand: 'all',
  species: 'all',
  showStations: true,
  showBgHits: true,
  stIdx: 1613,            // 0 .. 5490 (Case A default)
  stCohort: 'gnn_only_both',
  fullIdx: 0,             // 0 .. 149858
  stShowTarget: true,
  stShowV5: true,
  stShowV6: true,
  stShowDs: true,
  stShowOther: true,
  edgeTruth: false,
  edgePred: false,
  edgeFp: false,
  edgeFn: false,
  edgeCand: false,
  fullShowMc: true,
  cameraPreset: 'oblique',
  activeTab: 'summary',   // 'summary' | 'diagnostics'
  menuOpen: true,
  cardOpen: true,
  cleanMode: false,
};

// ── Lazy-Loaded Dataset Stores ──
let cloudData = null;
let stStore = null;       // { shard, routes, showcaseMap }
let stLoading = null;
let fullMeta = null;
let fullShards = new Map(); // shardIdx -> parsed shard views
let fullLoading = null;

let cloudGeomCache = new Map();
let cloudActorsKey = '';
let cloudActors = { leftFrames: null, rightFrames: null, entries: [] };
let diagSummaryText = '';

// ── Smooth Circular Sphere-Sprite Texture ──
function makeSphereSprite() {
  const sz = 64;
  const c = document.createElement('canvas');
  c.width = sz;
  c.height = sz;
  const ctx = c.getContext('2d');
  const g = ctx.createRadialGradient(sz * 0.38, sz * 0.36, sz * 0.04, sz * 0.5, sz * 0.5, sz * 0.48);
  g.addColorStop(0.0, 'rgba(255,255,255,1.0)');
  g.addColorStop(0.62, 'rgba(235,238,245,0.95)');
  g.addColorStop(0.88, 'rgba(195,202,214,0.55)');
  g.addColorStop(1.0, 'rgba(180,188,200,0.0)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, sz, sz);
  const tex = new THREE.CanvasTexture(c);
  tex.needsUpdate = true;
  return tex;
}

const HIT_SPRITE = makeSphereSprite();

// ── Three.js Dual-Scene Renderer ──
const canvas = document.getElementById('gl-canvas');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setClearColor(COLORS.bg, 1);

const sceneLeft = new THREE.Scene();
const sceneRight = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(34, 1.6, 1.0, 4000.0);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = false;
controls.screenSpacePanning = true;
controls.zoomSpeed = 1.15;
controls.rotateSpeed = 0.85;

// ── Helpers ──
function medianFloat(arr) {
  const n = arr.length;
  if (n === 0) return 0;
  const sorted = Float32Array.from(arr).sort();
  const mid = n >> 1;
  return (n & 1) ? sorted[mid] : 0.5 * (sorted[mid - 1] + sorted[mid]);
}

const fmtInt = (n) => Number(n).toLocaleString('en-US');

function sliceBinarySections(meta, buf) {
  const out = {};
  for (const [name, sec] of Object.entries(meta.sections)) {
    if (sec.dtype === 'float32') out[name] = new Float32Array(buf, sec.byteOffset, sec.length);
    else if (sec.dtype === 'uint32') out[name] = new Uint32Array(buf, sec.byteOffset, sec.length);
    else if (sec.dtype === 'int32') out[name] = new Int32Array(buf, sec.byteOffset, sec.length);
    else if (sec.dtype === 'uint16') out[name] = new Uint16Array(buf, sec.byteOffset, sec.length);
    else out[name] = new Uint8Array(buf, sec.byteOffset, sec.length);
  }
  return out;
}

function setLoadingBanner(visible, msg = '') {
  const mask = document.getElementById('loading-mask');
  if (msg) document.getElementById('loading-text').textContent = msg;
  mask.classList.toggle('hidden', !visible);
}

// ── Station Wireframe Builder ──
function buildStationFramesGeometry(bounds) {
  const { xMin, xMax, yMin, yMax } = bounds;
  const segs = [];
  for (const z of STATION_Z) {
    segs.push(
      xMin, yMin, z, xMax, yMin, z,
      xMax, yMin, z, xMax, yMax, z,
      xMax, yMax, z, xMin, yMax, z,
      xMin, yMax, z, xMin, yMin, z
    );
  }
  const z0 = STATION_Z[0];
  const z1 = STATION_Z[STATION_Z.length - 1];
  for (const [cx, cy] of [[xMin, yMin], [xMax, yMin], [xMax, yMax], [xMin, yMax]]) {
    segs.push(cx, cy, z0, cx, cy, z1);
  }
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(segs), 3));
  return geom;
}

const CLOUD_STATION_GEOM = buildStationFramesGeometry(CLOUD_BOX);

// ── Camera Presets ──
function applyCameraPreset(preset, boundsCenter = null, spanXY = 90, spanZ = 180) {
  state.cameraPreset = preset;
  const isDual = (state.mode === 'cloud' && state.cloudLayout === 'split');
  let cx = -8.0, cy = 0.0, cz = isDual ? 100.0 : 108.0;
  let dist = isDual ? 600.0 : 375.0;

  if (state.mode !== 'cloud' && boundsCenter) {
    cx = boundsCenter[0];
    cy = boundsCenter[1];
    cz = 0.5 * (STATION_Z[0] + STATION_Z[STATION_Z.length - 1]) + 10.0;
    dist = Math.max(spanZ * 1.95, spanXY * 3.1, 320.0);
  }

  let pos, up;
  if (preset === 'xz') {
    pos = [cx, cy + dist, cz];
    up = [1, 0, 0];
  } else if (preset === 'yz') {
    pos = [cx - dist, cy, cz];
    up = [0, 1, 0];
  } else {
    pos = [cx + 0.23 * dist, cy + 0.94 * dist, cz - 0.23 * dist];
    up = [0.97, -0.22, 0.06];
  }

  camera.up.set(up[0], up[1], up[2]).normalize();
  camera.position.set(pos[0], pos[1], pos[2]);
  controls.target.set(cx, cy, cz);
  controls.update();
  syncUIButtons();
}

// ── Data Loaders (Initial 1.1 MB Cloud + Lazy 5,491 Event + Lazy 149,859 Shards) ──
async function loadCloudBundle() {
  const [metaRes, binRes] = await Promise.all([
    fetch('./data/cloud_meta.json'),
    fetch('./data/cloud_5491.bin'),
  ]);
  if (!metaRes.ok || !binRes.ok) throw new Error('Failed to load cloud_5491 bundle');
  const [meta, buf] = await Promise.all([metaRes.json(), binRes.arrayBuffer()]);
  const s = sliceBinarySections(meta, buf);
  cloudData = {
    nEvents: meta.n_events,
    pTrue: s.p_true,
    dxBend: s.dx_bend,
    nLayers: s.n_layers,
    nEvHits: s.n_ev_hits,
    flags: s.flags,
    tracks: {
      truth: { off: s.truth_off, xyz: s.truth_xyz },
      v5: { off: s.v5_off, xyz: s.v5_xyz },
      v6: { off: s.v6_off, xyz: s.v6_xyz },
      ds: { off: s.ds_off, xyz: s.ds_xyz },
    },
  };
}

async function ensureStStoreLoaded() {
  if (stStore) return stStore;
  if (!stLoading) {
    setLoadingBanner(true, 'Loading all 5,491 second-track events & candidate fits…');
    stLoading = (async () => {
      const [mRes, bRes, rRes] = await Promise.all([
        fetch('./data/st_5491_meta.json'),
        fetch('./data/st_5491_geom.bin'),
        fetch('./data/st_5491_routes.json'),
      ]);
      if (!mRes.ok || !bRes.ok || !rRes.ok) throw new Error('Failed to load 5,491 single-event bundle');
      const [meta, buf, routes] = await Promise.all([mRes.json(), bRes.arrayBuffer(), rRes.json()]);
      const views = sliceBinarySections(meta, buf);
      const showcaseMap = new Map();
      for (const c of routes.showcase_cases) {
        showcaseMap.set(c.viewer_idx, c);
      }
      const sel = document.getElementById('sel-cohort');
      sel.innerHTML = '';
      for (const item of routes.cohort_labels) {
        const opt = document.createElement('option');
        opt.value = item.value;
        opt.textContent = item.title;
        sel.appendChild(opt);
      }
      sel.value = state.stCohort;
      stStore = { views, routes, showcaseMap, nEvents: routes.n_events };
      return stStore;
    })().finally(() => {
      stLoading = null;
      setLoadingBanner(false);
    });
  }
  return stLoading;
}

async function ensureFullShardLoaded(globalIdx) {
  if (!fullMeta) {
    setLoadingBanner(true, 'Loading 149,859-event index…');
    const res = await fetch('./data/full_gnn_meta.json');
    if (!res.ok) throw new Error('Failed to load full_gnn_meta.json');
    fullMeta = await res.json();
  }
  const sIdx = Math.min(fullMeta.shards.length - 1, Math.floor(globalIdx / fullMeta.shard_size));
  if (fullShards.has(sIdx)) {
    setLoadingBanner(false);
    return { shardIdx: sIdx, shardMeta: fullMeta.shards[sIdx], views: fullShards.get(sIdx) };
  }
  if (!fullLoading) {
    const sm = fullMeta.shards[sIdx];
    setLoadingBanner(true, `Loading full GNN shard ${sIdx + 1}/${fullMeta.shards.length} (events #${fmtInt(sm.ev_start)}–#${fmtInt(sm.ev_end - 1)})…`);
    fullLoading = (async () => {
      const bRes = await fetch(`./data/${sm.file}`);
      if (!bRes.ok) throw new Error(`Failed to fetch ${sm.file}`);
      const buf = await bRes.arrayBuffer();
      const views = sliceBinarySections(sm, buf);
      fullShards.set(sIdx, views);
    })().finally(() => {
      fullLoading = null;
      setLoadingBanner(false);
    });
  }
  await fullLoading;
  return { shardIdx: sIdx, shardMeta: fullMeta.shards[sIdx], views: fullShards.get(sIdx) };
}

// ── Extract Single Event from Packed Binary Views ──
function unpackShardEvent(views, localIdx) {
  const p0 = views.pos_off[localIdx];
  const p1 = views.pos_off[localIdx + 1];
  const nHits = p1 - p0;
  const pts = new Array(nHits);
  const layerId = new Uint8Array(nHits);
  for (let h = 0; h < nHits; h++) {
    const b = (p0 + h) * 3;
    pts[h] = [views.positions[b], views.positions[b + 1], views.positions[b + 2]];
    layerId[h] = views.layer_id[p0 + h];
  }

  const e0 = views.edge_off[localIdx];
  const e1 = views.edge_off[localIdx + 1];

  const m0 = views.mc_ev_off[localIdx];
  const m1 = views.mc_ev_off[localIdx + 1];
  const mcTracks = [];
  for (let t = m0; t < m1; t++) {
    const hs = views.mc_trk_off[t];
    const he = views.mc_trk_off[t + 1];
    const hits = Array.from(views.mc_hits.subarray(hs, he));
    mcTracks.push({ id: views.mc_ids[t], hits });
  }

  return {
    pts,
    layerId,
    e0,
    e1,
    edgeU: views.edge_u,
    edgeV: views.edge_v,
    edgeTruth: views.edge_truth,
    probV5: views.prob_v5,
    probV6: views.prob_v6,
    mcTracks,
  };
}

// ── 4-Cohort Cloud Slicing & Geometry Packing ──
function getCohortGeometries(cohortKey, modelKey, coordSrc, recoRoute, mom, occ, sp) {
  const cacheKey = `${cohortKey}|${modelKey}|${coordSrc}|${recoRoute}|${mom}|${occ}|${sp}`;
  if (cloudGeomCache.has(cacheKey)) return cloudGeomCache.get(cacheKey);

  const { nEvents, pTrue, nEvHits, flags, tracks } = cloudData;
  const gnnBit = modelKey === 'v5' ? 1 : 2;
  const srcKey = coordSrc === 'truth' ? 'truth' : (recoRoute === 'DS' ? 'ds' : modelKey);
  const { off, xyz } = tracks[srcKey];

  let nPts = 0;
  let nSegs = 0;
  const selected = [];

  for (let i = 0; i < nEvents; i++) {
    const p = pTrue[i];
    if (mom === 'low' && !(p < 300)) continue;
    if (mom === 'mid' && !(p >= 300 && p <= 900)) continue;
    if (mom === 'high' && !(p > 900)) continue;

    const nh = nEvHits[i];
    if (occ === 'low' && !(nh < 12)) continue;
    if (occ === 'mid' && !(nh >= 12 && nh <= 17)) continue;
    if (occ === 'high' && !(nh >= 18)) continue;

    const fl = flags[i];
    if (sp === 'ep' && !(fl & 16)) continue;
    if (sp === 'em' && !(fl & 32)) continue;

    const uGnn = (fl & gnnBit) !== 0;
    const uDs = (fl & 4) !== 0;
    const c = (uGnn && uDs) ? 'both' : (uGnn ? 'gnn_only' : (uDs ? 'ds_only' : 'neither'));
    if (c !== cohortKey) continue;

    const s = off[i];
    const e = off[i + 1];
    const k = e - s;
    if (k === 0) continue;
    selected.push(i);
    nPts += k;
    if (k >= 2) nSegs += (k - 1);
  }

  if (nPts === 0) {
    const empty = { lineGeom: null, hitGeom: null };
    cloudGeomCache.set(cacheKey, empty);
    return empty;
  }

  const hitBuf = new Float32Array(nPts * 3);
  const lineBuf = nSegs > 0 ? new Float32Array(nSegs * 6) : null;
  let hp = 0;
  let lp = 0;

  for (let idx = 0; idx < selected.length; idx++) {
    const i = selected[idx];
    const s = off[i];
    const e = off[i + 1];
    const k = e - s;
    for (let j = 0; j < k; j++) {
      const base = (s + j) * 3;
      hitBuf[hp++] = xyz[base];
      hitBuf[hp++] = xyz[base + 1];
      hitBuf[hp++] = xyz[base + 2];
      if (j < k - 1 && lineBuf) {
        const next = (s + j + 1) * 3;
        lineBuf[lp++] = xyz[base];
        lineBuf[lp++] = xyz[base + 1];
        lineBuf[lp++] = xyz[base + 2];
        lineBuf[lp++] = xyz[next];
        lineBuf[lp++] = xyz[next + 1];
        lineBuf[lp++] = xyz[next + 2];
      }
    }
  }

  const hitGeom = new THREE.BufferGeometry();
  hitGeom.setAttribute('position', new THREE.BufferAttribute(hitBuf, 3));

  let lineGeom = null;
  if (lineBuf) {
    lineGeom = new THREE.BufferGeometry();
    lineGeom.setAttribute('position', new THREE.BufferAttribute(lineBuf, 3));
  }

  const res = { lineGeom, hitGeom };
  cloudGeomCache.set(cacheKey, res);
  return res;
}

function computeCloudStats(modelKey, mom, occ, sp) {
  const { nEvents, pTrue, dxBend, flags, tracks } = cloudData;
  const gnnBit = modelKey === 'v5' ? 1 : 2;
  const tOff = tracks.truth.off;

  const buckets = {
    both: { p: [], dx: [], hits: 0 },
    gnn_only: { p: [], dx: [], hits: 0 },
    ds_only: { p: [], dx: [], hits: 0 },
    neither: { p: [], dx: [], hits: 0 },
  };

  for (let i = 0; i < nEvents; i++) {
    const p = pTrue[i];
    if (mom === 'low' && !(p < 300)) continue;
    if (mom === 'mid' && !(p >= 300 && p <= 900)) continue;
    if (mom === 'high' && !(p > 900)) continue;

    const nh = cloudData.nEvHits[i];
    if (occ === 'low' && !(nh < 12)) continue;
    if (occ === 'mid' && !(nh >= 12 && nh <= 17)) continue;
    if (occ === 'high' && !(nh >= 18)) continue;

    const fl = flags[i];
    if (sp === 'ep' && !(fl & 16)) continue;
    if (sp === 'em' && !(fl & 32)) continue;

    const uGnn = (fl & gnnBit) !== 0;
    const uDs = (fl & 4) !== 0;
    const key = (uGnn && uDs) ? 'both' : (uGnn ? 'gnn_only' : (uDs ? 'ds_only' : 'neither'));
    buckets[key].p.push(p);
    buckets[key].dx.push(Math.abs(dxBend[i]));
    buckets[key].hits += (tOff[i + 1] - tOff[i]);
  }

  const tot = Math.max(1,
    buckets.both.p.length + buckets.gnn_only.p.length +
    buckets.ds_only.p.length + buckets.neither.p.length
  );

  const out = {};
  for (const k of ['both', 'gnn_only', 'ds_only', 'neither']) {
    const b = buckets[k];
    const n = b.p.length;
    out[k] = {
      n,
      pct: (100.0 * n) / tot,
      hits: b.hits,
      pMed: medianFloat(b.p),
      dxMed: medianFloat(b.dx),
    };
  }
  out.totalSlice = (
    buckets.both.p.length + buckets.gnn_only.p.length +
    buckets.ds_only.p.length + buckets.neither.p.length
  );
  return out;
}

function clearSceneObjects(sc) {
  while (sc.children.length > 0) {
    const obj = sc.children[0];
    sc.remove(obj);
    if (obj.material) obj.material.dispose();
  }
}

// ── 1) Mode 'cloud': 4-Cohort 3D Clouds ──
function syncCloudScene(resetCam = false) {
  const split = state.cloudLayout === 'split';
  const gnnKey = state.model;
  const gnnLabel = gnnKey.toUpperCase();
  const gnnColor = gnnKey === 'v5' ? COLORS.v5 : COLORS.v6;

  const stats = computeCloudStats(gnnKey, state.momBand, state.occBand, state.species);
  const nTot = stats.totalSlice;
  const nGnnOk = stats.both.n + stats.gnn_only.n;
  const nDsOk = stats.both.n + stats.ds_only.n;

  const sliceTags = [];
  if (state.momBand !== 'all') sliceTags.push(state.momBand);
  if (state.occBand !== 'all') sliceTags.push(`occ:${state.occBand}`);
  if (state.species !== 'all') sliceTags.push(state.species === 'ep' ? 'e+' : 'e−');
  const sliceStr = sliceTags.length ? ` · ${sliceTags.join(', ')}` : '';

  document.getElementById('plaque-headline').textContent =
    `N = ${fmtInt(nTot)}${sliceStr} · ${gnnLabel} ${(100 * nGnnOk / Math.max(1, nTot)).toFixed(1)}% (${fmtInt(nGnnOk)}) vs DS ${(100 * nDsOk / Math.max(1, nTot)).toFixed(1)}% (${fmtInt(nDsOk)})`;
  document.getElementById('plaque-subline').textContent =
    `${gnnLabel}-only +${fmtInt(stats.gnn_only.n)} · DS-only +${fmtInt(stats.ds_only.n)} · Both ${fmtInt(stats.both.n)}`;

  const fmtStat = (s) => s.n === 0
    ? '0'
    : `${fmtInt(s.n)} (${s.pct.toFixed(1)}%) · p_med ${Math.round(s.pMed)} MeV/c · |Δx| ${s.dxMed.toFixed(1)} mm`;

  document.getElementById('txt-cloud-both').textContent = fmtStat(stats.both);
  document.getElementById('txt-cloud-gnn').textContent = fmtStat(stats.gnn_only);
  document.getElementById('txt-cloud-ds').textContent = fmtStat(stats.ds_only);
  document.getElementById('txt-cloud-neither').textContent = fmtStat(stats.neither);
  document.getElementById('pill-cloud-gnn').style.color = gnnKey === 'v5' ? '#178e89' : '#8A80F0';
  document.getElementById('lbl-sw-gnn').style.setProperty('--sw-col', gnnKey === 'v5' ? '#178e89' : '#665ae0');

  if (split) {
    if (state.cloudPairing === 'quad') {
      document.getElementById('pane-badge-left').textContent =
        `${gnnLabel}-only ${fmtInt(stats.gnn_only.n)} · DS-only ${fmtInt(stats.ds_only.n)}`;
      document.getElementById('pane-badge-right').textContent =
        `Both ${fmtInt(stats.both.n)} · Neither ${fmtInt(stats.neither.n)}`;
    } else {
      document.getElementById('pane-badge-left').textContent =
        `${gnnLabel} Recalled · ${fmtInt(nGnnOk)} (${(100 * nGnnOk / Math.max(1, nTot)).toFixed(1)}%)`;
      document.getElementById('pane-badge-right').textContent =
        `DS Recalled · ${fmtInt(nDsOk)} (${(100 * nDsOk / Math.max(1, nTot)).toFixed(1)}%)`;
    }
  }

  diagSummaryText =
    `Sample: DS digitized smear=1 multi-track cohort (5,491 events)\n` +
    `Active slice: N=${fmtInt(nTot)} · coords=${state.cloudCoords} · geom=${state.cloudGeom} · base_alpha=${state.cloudOpacity.toFixed(2)}\n` +
    `Truth hits in slice: Both=${fmtInt(stats.both.hits)}, ${gnnLabel}+=${fmtInt(stats.gnn_only.hits)}, DS+=${fmtInt(stats.ds_only.hits)}, Miss=${fmtInt(stats.neither.hits)}`;
  if (state.activeTab === 'diagnostics') {
    document.getElementById('diagnostics-pane').textContent = diagSummaryText;
  }

  const sig = `cloud|${split}|${state.cloudPairing}|${gnnKey}|${state.cloudCoords}|${state.momBand}|${state.occBand}|${state.species}`;
  if (cloudActorsKey !== sig) {
    clearSceneObjects(sceneLeft);
    clearSceneObjects(sceneRight);
    cloudActors = { leftFrames: null, rightFrames: null, entries: [] };

    const specBothGnn = { key: 'both', col: COLORS.truth, lMult: 0.50, hMult: 0.80, route: 'GNN' };
    const specBothDs = { key: 'both', col: COLORS.truth, lMult: 0.50, hMult: 0.80, route: 'DS' };
    const specGnnOnly = { key: 'gnn_only', col: gnnColor, lMult: 1.35, hMult: 1.85, route: 'GNN' };
    const specDsOnly = { key: 'ds_only', col: COLORS.ds, lMult: 1.85, hMult: 2.35, route: 'DS' };
    const specNeither = { key: 'neither', col: COLORS.neither, lMult: 0.48, hMult: 0.78, route: 'GNN' };

    const addFrame = (sc) => {
      const mat = new THREE.LineBasicMaterial({ color: COLORS.station, transparent: true, opacity: 0.28, depthWrite: false });
      const lines = new THREE.LineSegments(CLOUD_STATION_GEOM, mat);
      sc.add(lines);
      return lines;
    };

    const populatePane = (sc, specs) => {
      for (const spc of specs) {
        const { lineGeom, hitGeom } = getCohortGeometries(
          spc.key, gnnKey, state.cloudCoords, spc.route,
          state.momBand, state.occBand, state.species
        );
        let lObj = null;
        let hObj = null;
        if (lineGeom) {
          lObj = new THREE.LineSegments(lineGeom, new THREE.LineBasicMaterial({
            color: spc.col, transparent: true, opacity: 0.2, depthWrite: false,
          }));
          sc.add(lObj);
        }
        if (hitGeom) {
          hObj = new THREE.Points(hitGeom, new THREE.PointsMaterial({
            color: spc.col, size: 4.8, sizeAttenuation: false, map: HIT_SPRITE,
            transparent: true, opacity: 0.3, alphaTest: 0.04, depthWrite: false,
          }));
          sc.add(hObj);
        }
        cloudActors.entries.push({ lObj, hObj, key: spc.key, lMult: spc.lMult, hMult: spc.hMult });
      }
    };

    cloudActors.leftFrames = addFrame(sceneLeft);
    if (split) {
      cloudActors.rightFrames = addFrame(sceneRight);
      if (state.cloudPairing === 'quad') {
        populatePane(sceneLeft, [specGnnOnly, specDsOnly]);
        populatePane(sceneRight, [specBothGnn, specNeither]);
      } else {
        populatePane(sceneLeft, [specNeither, specBothGnn, specGnnOnly]);
        populatePane(sceneRight, [specNeither, specBothDs, specDsOnly]);
      }
    } else {
      populatePane(sceneLeft, [specNeither, specBothGnn, specGnnOnly, specDsOnly]);
    }
    cloudActorsKey = sig;
  }

  if (cloudActors.leftFrames) cloudActors.leftFrames.visible = state.showStations;
  if (cloudActors.rightFrames) cloudActors.rightFrames.visible = state.showStations;

  const visMap = {
    both: state.showBoth,
    gnn_only: state.showGnnOnly,
    ds_only: state.showDsOnly,
    neither: state.showNeither,
  };
  const showLines = state.cloudGeom === 'both' || state.cloudGeom === 'lines';
  const showHits = state.cloudGeom === 'both' || state.cloudGeom === 'hits';
  const alpha = state.cloudOpacity;

  for (const ent of cloudActors.entries) {
    const on = !!visMap[ent.key];
    if (ent.lObj) {
      ent.lObj.visible = on && showLines;
      ent.lObj.material.opacity = Math.min(0.92, Math.max(0.02, alpha * ent.lMult));
    }
    if (ent.hObj) {
      ent.hObj.visible = on && showHits;
      ent.hObj.material.opacity = Math.min(0.95, Math.max(0.03, alpha * ent.hMult));
    }
  }

  if (resetCam) applyCameraPreset(state.cameraPreset);
}

// ── Collinear Styled Segments & Thick 3D Route Tubes ──
function computeStyledSegmentPairs(pts, hitIndices, style = 'solid', slot = 0, nSlots = 1) {
  if (!hitIndices || hitIndices.length < 2) return [];
  const coords = hitIndices.map((h) => pts[h]).filter(Boolean);
  if (coords.length < 2) return [];

  const pairs = [];
  if (style === 'solid' && nSlots <= 1) {
    for (let i = 0; i < coords.length - 1; i++) {
      pairs.push([coords[i], coords[i + 1]]);
    }
  } else {
    let cycle, intervals;
    if (nSlots > 1) {
      cycle = 9.0;
      const span = 1.0 / nSlots;
      intervals = [[slot * span + 0.04 * span, (slot + 1) * span - 0.08 * span]];
    } else if (style === 'dash') {
      cycle = 5.2;
      intervals = [[0.0, 0.58]];
    } else {
      cycle = 6.8;
      intervals = [[0.0, 0.46], [0.64, 0.80]];
    }

    for (let i = 0; i < coords.length - 1; i++) {
      const a = coords[i];
      const b = coords[i + 1];
      const dx = b[0] - a[0], dy = b[1] - a[1], dz = b[2] - a[2];
      const len = Math.hypot(dx, dy, dz);
      if (len < 1e-6) continue;
      for (let s = 0; s < len; s += cycle) {
        for (const [f0, f1] of intervals) {
          const t0 = Math.min(1.0, (s + f0 * cycle) / len);
          const t1 = Math.min(1.0, (s + f1 * cycle) / len);
          if (t1 - t0 > 1e-4) {
            pairs.push([
              [a[0] + dx * t0, a[1] + dy * t0, a[2] + dz * t0],
              [a[0] + dx * t1, a[1] + dy * t1, a[2] + dz * t1],
            ]);
          }
        }
      }
    }
  }
  return pairs;
}

function buildStyledTrackSegments(pts, hitIndices, style = 'solid', slot = 0, nSlots = 1) {
  const pairs = computeStyledSegmentPairs(pts, hitIndices, style, slot, nSlots);
  if (!pairs.length) return null;
  const segs = new Float32Array(pairs.length * 6);
  let p = 0;
  for (const [a, b] of pairs) {
    segs[p++] = a[0]; segs[p++] = a[1]; segs[p++] = a[2];
    segs[p++] = b[0]; segs[p++] = b[1]; segs[p++] = b[2];
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(segs, 3));
  return g;
}

const UNIT_CYL = new THREE.CylinderGeometry(1, 1, 1, 8, 1, false);

function buildThickRouteGroup(pts, hitIndices, colorHex, radius = 0.46, style = 'solid', slot = 0, nSlots = 1, opacity = 1.0) {
  const pairs = computeStyledSegmentPairs(pts, hitIndices, style, slot, nSlots);
  if (!pairs.length) return null;
  const mat = opacity < 1.0
    ? new THREE.MeshBasicMaterial({ color: colorHex, transparent: true, opacity })
    : new THREE.MeshBasicMaterial({ color: colorHex });
  const mesh = new THREE.InstancedMesh(UNIT_CYL, mat, pairs.length);
  const upY = new THREE.Vector3(0, 1, 0);
  const vA = new THREE.Vector3();
  const vB = new THREE.Vector3();
  const dir = new THREE.Vector3();
  const mid = new THREE.Vector3();
  const quat = new THREE.Quaternion();
  const scl = new THREE.Vector3();
  const m4 = new THREE.Matrix4();

  pairs.forEach(([a, b], idx) => {
    vA.set(a[0], a[1], a[2]);
    vB.set(b[0], b[1], b[2]);
    dir.subVectors(vB, vA);
    const len = dir.length();
    mid.addVectors(vA, vB).multiplyScalar(0.5);
    quat.setFromUnitVectors(upY, dir.normalize());
    scl.set(radius, len, radius);
    m4.compose(mid, quat, scl);
    mesh.setMatrixAt(idx, m4);
  });
  mesh.instanceMatrix.needsUpdate = true;
  return mesh;
}

function drawEdgeGraphLayers(evObj) {
  const { pts, e0, e1, edgeU, edgeV, edgeTruth, probV5, probV6 } = evObj;
  const pArr = state.model === 'v5' ? probV5 : probV6;
  const thrU16 = Math.round(state.threshold * 65535.0);

  let nTp = 0, nFp = 0, nFn = 0;
  const segTruth = [], segPred = [], segFp = [], segFn = [], segCand = [];

  for (let e = e0; e < e1; e++) {
    const a = pts[edgeU[e]];
    const b = pts[edgeV[e]];
    if (!a || !b) continue;
    const isT = edgeTruth[e] === 1;
    const isP = pArr[e] >= thrU16;
    if (isT && isP) nTp++;
    else if (!isT && isP) nFp++;
    else if (isT && !isP) nFn++;

    if (state.edgeCand) segCand.push(...a, ...b);
    if (state.edgeTruth && isT) segTruth.push(...a, ...b);
    if (state.edgePred && isP) segPred.push(...a, ...b);
    if (state.edgeFp && (!isT && isP)) segFp.push(...a, ...b);
    if (state.edgeFn && (isT && !isP)) segFn.push(...a, ...b);
  }

  const addSegArray = (arr, col, op) => {
    if (!arr.length) return;
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(arr), 3));
    const mat = op < 1.0
      ? new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: op })
      : new THREE.LineBasicMaterial({ color: col });
    sceneLeft.add(new THREE.LineSegments(g, mat));
  };
  addSegArray(segCand, COLORS.cand, 0.16);
  addSegArray(segTruth, COLORS.truth, 0.78);
  addSegArray(segPred, state.model === 'v5' ? COLORS.v5 : COLORS.v6, 0.75);
  addSegArray(segFp, COLORS.fp, 1.0);
  addSegArray(segFn, COLORS.fn, 1.0);

  return { nEdges: e1 - e0, nTp, nFp, nFn };
}

// ── 2) Mode 'st': All 5,491 DS smear=1 Multi-Track Single Events ──
async function syncStScene(resetCam = false) {
  const store = await ensureStStoreLoaded();
  if (state.mode !== 'st') return;

  cloudActorsKey = '';
  clearSceneObjects(sceneLeft);
  clearSceneObjects(sceneRight);

  const idx = Math.max(0, Math.min(store.nEvents - 1, state.stIdx));
  state.stIdx = idx;
  const evObj = unpackShardEvent(store.views, idx);
  const pts = evObj.pts;
  const row = store.routes.rows[idx];
  const [targetId, pTrue, spName, nTargetLayers, sharedHits, targetHits, rV5, rV6, rDS] = row;

  let xMin = -35, xMax = 35, yMin = -35, yMax = 35;
  for (const [x, y] of pts) {
    if (x < xMin) xMin = x;
    if (x > xMax) xMax = x;
    if (y < yMin) yMin = y;
    if (y > yMax) yMax = y;
  }
  const pad = 8;
  if (state.showStations) {
    const fg = buildStationFramesGeometry({ xMin: xMin - pad, xMax: xMax + pad, yMin: yMin - pad, yMax: yMax + pad });
    sceneLeft.add(new THREE.LineSegments(fg, new THREE.LineBasicMaterial({
      color: COLORS.station, transparent: true, opacity: 0.24, depthWrite: false,
    })));
  }

  const targetSet = new Set(targetHits);
  if (state.showBgHits && pts.length > 0) {
    const bgCoords = [];
    for (let i = 0; i < pts.length; i++) {
      if (!targetSet.has(i) || !state.stShowTarget) bgCoords.push(...pts[i]);
    }
    if (bgCoords.length > 0) {
      const bgG = new THREE.BufferGeometry();
      bgG.setAttribute('position', new THREE.BufferAttribute(new Float32Array(bgCoords), 3));
      sceneLeft.add(new THREE.Points(bgG, new THREE.PointsMaterial({
        color: COLORS.bgHit, size: 8.5, sizeAttenuation: false, map: HIT_SPRITE,
        transparent: true, opacity: 0.36, alphaTest: 0.05, depthWrite: false,
      })));
    }
  }

  if (state.stShowTarget && targetHits.length > 0) {
    const tgCoords = [];
    for (const h of targetHits) if (pts[h]) tgCoords.push(...pts[h]);
    const tgG = new THREE.BufferGeometry();
    tgG.setAttribute('position', new THREE.BufferAttribute(new Float32Array(tgCoords), 3));
    sceneLeft.add(new THREE.Points(tgG, new THREE.PointsMaterial({
      color: COLORS.truth, size: 14.5, sizeAttenuation: false, map: HIT_SPRITE,
      transparent: true, opacity: 0.88, alphaTest: 0.05,
    })));
    const tgLine = buildStyledTrackSegments(pts, targetHits, 'solid');
    if (tgLine) {
      sceneLeft.add(new THREE.LineSegments(tgLine, new THREE.LineBasicMaterial({
        color: COLORS.truth, transparent: true, opacity: 0.32,
      })));
    }
  }

  if (state.stShowOther) {
    for (const mct of evObj.mcTracks) {
      if (mct.id === targetId) continue;
      const og = buildStyledTrackSegments(pts, mct.hits, 'dash');
      if (og) {
        sceneLeft.add(new THREE.LineSegments(og, new THREE.LineBasicMaterial({
          color: COLORS.otherMc, transparent: true, opacity: 0.25,
        })));
      }
    }
  }

  const routeMap = { V5: rV5, V6: rV6, DS: rDS };
  const routeSpecs = [
    { name: 'V5', on: state.stShowV5, col: COLORS.v5, style: 'dashdot' },
    { name: 'V6', on: state.stShowV6, col: COLORS.v6, style: 'solid' },
    { name: 'DS', on: state.stShowDs, col: COLORS.ds, style: 'dash' },
  ];

  if (state.stShowOther) {
    for (const rs of routeSpecs) {
      if (!rs.on) continue;
      const oCands = routeMap[rs.name][2];
      for (const oc of oCands) {
        const cg = buildStyledTrackSegments(pts, oc, 'solid');
        if (cg) {
          sceneLeft.add(new THREE.LineSegments(cg, new THREE.LineBasicMaterial({
            color: rs.col, transparent: true, opacity: 0.22,
          })));
        }
      }
    }
  }

  const activeMatched = routeSpecs.filter((rs) => rs.on && routeMap[rs.name][1].length >= 2);
  activeMatched.forEach((rs, slotIdx) => {
    const mHits = routeMap[rs.name][1];
    const tube = buildThickRouteGroup(pts, mHits, rs.col, 0.48, rs.style, slotIdx, activeMatched.length, 1.0);
    if (tube) sceneLeft.add(tube);
    if (rs.name === 'DS') {
      const offCoords = [];
      for (const h of mHits) if (!targetSet.has(h) && pts[h]) offCoords.push(...pts[h]);
      if (offCoords.length > 0) {
        const og = new THREE.BufferGeometry();
        og.setAttribute('position', new THREE.BufferAttribute(new Float32Array(offCoords), 3));
        sceneLeft.add(new THREE.Points(og, new THREE.PointsMaterial({
          color: COLORS.ds, size: 13.5, sizeAttenuation: false, map: HIT_SPRITE, alphaTest: 0.08,
        })));
      }
    }
  });

  const edgeStats = drawEdgeGraphLayers(evObj);

  // HUD updates
  document.getElementById('plaque-headline').textContent =
    `Event #${fmtInt(idx)} / ${fmtInt(store.nEvents)} · p_true ${pTrue.toFixed(1)} MeV/c`;
  document.getElementById('plaque-subline').textContent =
    `V5 ${rV5[0] ? 'ok' : 'fail'} · V6 ${rV6[0] ? 'ok' : 'fail'} · DS ${rDS[0] ? 'ok' : 'fail'}`;

  document.getElementById('txt-st-hdr').textContent =
    `${spName} · p_true ${pTrue.toFixed(1)} MeV/c · ${nTargetLayers} layers`;
  document.getElementById('txt-st-v5').textContent = rV5[3];
  document.getElementById('txt-st-v6').textContent = rV6[3];
  document.getElementById('txt-st-ds').textContent = rDS[3];

  const scCase = store.showcaseMap.get(idx);
  const noteEl = document.getElementById('txt-st-note');
  noteEl.textContent = scCase ? scCase.summary : '';
  noteEl.classList.toggle('hidden', !scCase);

  const cohList = store.routes.cohorts[state.stCohort] || [];
  const posInCoh = cohList.indexOf(idx);
  document.getElementById('txt-cohort-pos').textContent = posInCoh >= 0
    ? `${fmtInt(posInCoh + 1)} / ${fmtInt(cohList.length)} in filter`
    : `outside filter (${fmtInt(cohList.length)})`;

  document.getElementById('inp-event-idx').max = String(store.nEvents - 1);
  document.getElementById('inp-event-idx').value = String(idx);
  document.getElementById('dock-ev-pos').textContent = `#${fmtInt(idx)} / ${fmtInt(store.nEvents)}`;

  diagSummaryText =
    `Viewer index: #${idx} · target_id=${targetId} · digitized hits=${pts.length} · shared target hits=${sharedHits}\n` +
    `${rV5[4]}\n${rV6[4]}\n${rDS[4]}\n` +
    `Edge graph (${state.model.toUpperCase()} @ thr=${state.threshold.toFixed(2)}): E=${edgeStats.nEdges}, TP=${edgeStats.nTp}, FP=${edgeStats.nFp}, FN=${edgeStats.nFn}`;
  if (state.activeTab === 'diagnostics') {
    document.getElementById('diagnostics-pane').textContent = diagSummaryText;
  }

  syncUIButtons();
  if (resetCam) {
    const cx = 0.5 * (xMin + xMax);
    const cy = 0.5 * (yMin + yMax);
    const spanXY = Math.max(xMax - xMin, yMax - yMin, 30);
    applyCameraPreset(state.cameraPreset, [cx, cy, 95], spanXY, 180);
  }
}

// ── 3) Mode 'full': All 149,859 Full GNN-Digitized Events ──
async function syncFullScene(resetCam = false) {
  const idx = Math.max(0, Math.min(149858, state.fullIdx));
  state.fullIdx = idx;
  const { shardMeta, views } = await ensureFullShardLoaded(idx);
  if (state.mode !== 'full') return;

  cloudActorsKey = '';
  clearSceneObjects(sceneLeft);
  clearSceneObjects(sceneRight);

  const localIdx = idx - shardMeta.ev_start;
  const evObj = unpackShardEvent(views, localIdx);
  const pts = evObj.pts;

  let xMin = -35, xMax = 35, yMin = -35, yMax = 35;
  for (const [x, y] of pts) {
    if (x < xMin) xMin = x;
    if (x > xMax) xMax = x;
    if (y < yMin) yMin = y;
    if (y > yMax) yMax = y;
  }
  const pad = 8;
  if (state.showStations) {
    const fg = buildStationFramesGeometry({ xMin: xMin - pad, xMax: xMax + pad, yMin: yMin - pad, yMax: yMax + pad });
    sceneLeft.add(new THREE.LineSegments(fg, new THREE.LineBasicMaterial({
      color: COLORS.station, transparent: true, opacity: 0.24, depthWrite: false,
    })));
  }

  // Layer-colored digitized hits (N1..N6 DS ramp)
  if (state.showBgHits && pts.length > 0) {
    for (let lay = 0; lay < 6; lay++) {
      const coords = [];
      for (let h = 0; h < pts.length; h++) {
        if (evObj.layerId[h] === lay) coords.push(...pts[h]);
      }
      if (coords.length > 0) {
        const g = new THREE.BufferGeometry();
        g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(coords), 3));
        sceneLeft.add(new THREE.Points(g, new THREE.PointsMaterial({
          color: HIT_LAYER_HEX[lay], size: 11.0, sizeAttenuation: false, map: HIT_SPRITE, alphaTest: 0.06,
        })));
      }
    }
  }

  if (state.fullShowMc) {
    for (const mct of evObj.mcTracks) {
      const og = buildStyledTrackSegments(pts, mct.hits, 'dash');
      if (og) {
        sceneLeft.add(new THREE.LineSegments(og, new THREE.LineBasicMaterial({
          color: COLORS.truth, transparent: true, opacity: 0.34,
        })));
      }
    }
  }

  const es = drawEdgeGraphLayers(evObj);
  const prec = (es.nTp + es.nFp) > 0 ? (100 * es.nTp / (es.nTp + es.nFp)) : 100.0;
  const rec = (es.nTp + es.nFn) > 0 ? (100 * es.nTp / (es.nTp + es.nFn)) : 100.0;
  const gnnLabel = state.model.toUpperCase();

  document.getElementById('plaque-headline').textContent =
    `Full GNN · Event #${fmtInt(idx)} / 149,859 · ${pts.length} hits`;
  document.getElementById('plaque-subline').textContent =
    `${gnnLabel} @ ${state.threshold.toFixed(2)} · Recall ${rec.toFixed(1)}% · Precision ${prec.toFixed(1)}%`;

  document.getElementById('txt-full-hdr').textContent =
    `Event #${fmtInt(idx)} / 149,859 · ${pts.length} hits · ${es.nEdges} candidate edges`;
  document.getElementById('pill-full-model').textContent = gnnLabel;
  document.getElementById('pill-full-model').style.color = state.model === 'v5' ? '#178e89' : '#8A80F0';
  document.getElementById('txt-full-metrics').textContent =
    `Recall ${rec.toFixed(1)}% · Precision ${prec.toFixed(1)}% · TP ${es.nTp} · FP ${es.nFp} · FN ${es.nFn}`;
  document.getElementById('txt-full-mc').textContent =
    `${evObj.mcTracks.length} MC particle track(s) with ≥2 tracker hits`;

  document.getElementById('inp-event-idx').max = '149858';
  document.getElementById('inp-event-idx').value = String(idx);
  document.getElementById('dock-ev-pos').textContent = `#${fmtInt(idx)} / 149,859`;

  diagSummaryText =
    `Full GNN sample · global viewer_idx=#${idx} (shard ${shardMeta.file}, local #${localIdx})\n` +
    `Hits=${pts.length}, Candidate edges=${es.nEdges} (${gnnLabel} @ thr=${state.threshold.toFixed(2)}: TP=${es.nTp}, FP=${es.nFp}, FN=${es.nFn})\n` +
    `MC tracks: ${evObj.mcTracks.map((t) => `MC ${t.id} (${t.hits.length} hits)`).join(', ') || 'none'}`;
  if (state.activeTab === 'diagnostics') {
    document.getElementById('diagnostics-pane').textContent = diagSummaryText;
  }

  syncUIButtons();
  if (resetCam) {
    const cx = 0.5 * (xMin + xMax);
    const cy = 0.5 * (yMin + yMax);
    const spanXY = Math.max(xMax - xMin, yMax - yMin, 30);
    applyCameraPreset(state.cameraPreset, [cx, cy, 95], spanXY, 180);
  }
}

// ── Orientation Gizmo ──
const axesCanvas = document.getElementById('axes-canvas');
const axesCtx = axesCanvas.getContext('2d');
const invQuat = new THREE.Quaternion();
const tmpVec = new THREE.Vector3();

function drawOrientationGizmo() {
  const w = axesCanvas.width;
  const h = axesCanvas.height;
  axesCtx.clearRect(0, 0, w, h);
  if (state.cleanMode) return;

  invQuat.copy(camera.quaternion).invert();
  const cx = w * 0.5, cy = h * 0.52, r = w * 0.32;
  const axes = [
    { label: 'X', dir: [1, 0, 0], col: '#d9534f' },
    { label: 'Y', dir: [0, 1, 0], col: '#178e89' },
    { label: 'Z', dir: [0, 0, 1], col: '#e68613' },
  ].map((a) => {
    tmpVec.set(a.dir[0], a.dir[1], a.dir[2]).applyQuaternion(invQuat);
    return { ...a, x: tmpVec.x, y: -tmpVec.y, z: tmpVec.z };
  }).sort((a, b) => a.z - b.z);

  axesCtx.lineWidth = 3.2;
  axesCtx.font = '700 20px "Open Sans", sans-serif';
  axesCtx.textAlign = 'center';
  axesCtx.textBaseline = 'middle';

  for (const a of axes) {
    const ex = cx + a.x * r;
    const ey = cy + a.y * r;
    axesCtx.strokeStyle = a.col;
    axesCtx.beginPath();
    axesCtx.moveTo(cx, cy);
    axesCtx.lineTo(ex, ey);
    axesCtx.stroke();
    axesCtx.fillStyle = a.col;
    axesCtx.fillText(a.label, cx + a.x * (r + 18), cy + a.y * (r + 18));
  }
}

// ── UI State Synchronization ──
function syncUIButtons() {
  const isCloud = state.mode === 'cloud';
  const isSt = state.mode === 'st';
  const isFull = state.mode === 'full';
  const isSplit = isCloud && state.cloudLayout === 'split';

  document.getElementById('btn-mode-cloud').classList.toggle('active-teal', isCloud);
  document.getElementById('btn-mode-st').classList.toggle('active-teal', isSt);
  document.getElementById('btn-mode-full').classList.toggle('active-teal', isFull);

  document.getElementById('plaque-legend-cloud').classList.toggle('hidden', !isCloud);
  document.getElementById('plaque-legend-event').classList.toggle('hidden', isCloud);

  document.getElementById('btn-model-v5').classList.toggle('active-teal', state.model === 'v5');
  document.getElementById('btn-model-v6').classList.toggle('active-violet', state.model === 'v6');

  const slider = document.getElementById('main-slider');
  const sliderVal = document.getElementById('main-slider-val');
  if (isCloud) {
    slider.min = '0.04';
    slider.max = '0.55';
    slider.step = '0.01';
    slider.value = String(state.cloudOpacity);
    sliderVal.textContent = state.cloudOpacity.toFixed(2);
  } else {
    slider.min = '0.00';
    slider.max = '1.00';
    slider.step = '0.01';
    slider.value = String(state.threshold);
    sliderVal.textContent = state.threshold.toFixed(2);
  }

  document.getElementById('cloud-controls-panel').classList.toggle('hidden', !isCloud);
  document.getElementById('single-controls-panel').classList.toggle('hidden', isCloud);
  document.getElementById('st-cohort-box').classList.toggle('hidden', !isSt);
  document.getElementById('st-routes-box').classList.toggle('hidden', !isSt);
  document.getElementById('row-sw-full-mc').classList.toggle('hidden', !isFull);
  document.getElementById('row-sw-bg-hits').classList.toggle('hidden', isCloud);

  document.getElementById('split-line').classList.toggle('hidden', !isSplit);
  document.getElementById('pane-badge-left').classList.toggle('hidden', !isSplit || state.cleanMode);
  document.getElementById('pane-badge-right').classList.toggle('hidden', !isSplit || state.cleanMode);

  document.getElementById('btn-lay-split').classList.toggle('active-violet', state.cloudLayout === 'split');
  document.getElementById('btn-lay-overlay').classList.toggle('active-violet', state.cloudLayout === 'overlay');
  document.getElementById('row-split-pairing').classList.toggle('hidden', state.cloudLayout !== 'split');

  document.getElementById('btn-pair-recall').classList.toggle('active-teal', state.cloudPairing === 'recall');
  document.getElementById('btn-pair-quad').classList.toggle('active-teal', state.cloudPairing === 'quad');

  document.getElementById('btn-geom-both').classList.toggle('active-violet', state.cloudGeom === 'both');
  document.getElementById('btn-geom-lines').classList.toggle('active-violet', state.cloudGeom === 'lines');
  document.getElementById('btn-geom-hits').classList.toggle('active-violet', state.cloudGeom === 'hits');

  document.getElementById('btn-coord-truth').classList.toggle('active-teal', state.cloudCoords === 'truth');
  document.getElementById('btn-coord-reco').classList.toggle('active-teal', state.cloudCoords === 'reco');

  document.getElementById('dock-cam-oblique').classList.toggle('active-teal', state.cameraPreset === 'oblique');
  document.getElementById('dock-cam-xz').classList.toggle('active-teal', state.cameraPreset === 'xz');
  document.getElementById('dock-cam-yz').classList.toggle('active-teal', state.cameraPreset === 'yz');

  document.getElementById('dock-group-cloud').classList.toggle('hidden', !isCloud);
  document.getElementById('dock-group-single').classList.toggle('hidden', isCloud);
  document.getElementById('dock-cases-wrap').classList.toggle('hidden', !isSt);

  document.getElementById('dock-split').classList.toggle('active-violet', state.cloudLayout === 'split');
  document.getElementById('dock-overlay').classList.toggle('active-violet', state.cloudLayout === 'overlay');
  document.getElementById('dock-pair-recall').classList.toggle('active-teal', state.cloudPairing === 'recall');
  document.getElementById('dock-pair-quad').classList.toggle('active-teal', state.cloudPairing === 'quad');

  const caseIndices = [1613, 2589, 86];
  for (let k = 0; k < 3; k++) {
    document.getElementById(`dock-case-${k}`).classList.toggle('active-violet', isSt && state.stIdx === caseIndices[k]);
  }

  const isDiag = state.activeTab === 'diagnostics';
  document.getElementById('tab-summary').classList.toggle('active', !isDiag);
  document.getElementById('tab-diagnostics').classList.toggle('active', isDiag);
  document.getElementById('summary-cloud').classList.toggle('hidden', isDiag || !isCloud);
  document.getElementById('summary-st').classList.toggle('hidden', isDiag || !isSt);
  document.getElementById('summary-full').classList.toggle('hidden', isDiag || !isFull);
  document.getElementById('diagnostics-pane').classList.toggle('hidden', !isDiag);
  if (isDiag) {
    document.getElementById('diagnostics-pane').textContent = diagSummaryText;
  }

  document.getElementById('hud-plaque').classList.toggle('hidden', state.cleanMode);
  document.getElementById('hud-menu').classList.toggle('hidden', state.cleanMode);
  document.getElementById('hud-card').classList.toggle('hidden', state.cleanMode);
  document.getElementById('hud-dock').classList.toggle('hidden', state.cleanMode);
  document.getElementById('btn-exit-clean').classList.toggle('hidden', !state.cleanMode);
}

function refreshScene(resetCam = false) {
  syncUIButtons();
  if (!cloudData) return;
  if (state.mode === 'cloud') {
    syncCloudScene(resetCam);
  } else if (state.mode === 'st') {
    syncStScene(resetCam).catch((err) => setLoadingBanner(true, `Load error: ${err.message}`));
  } else {
    syncFullScene(resetCam).catch((err) => setLoadingBanner(true, `Load error: ${err.message}`));
  }
}

// ── Wire DOM Controls ──
function wireControls() {
  const bindClick = (id, fn) => document.getElementById(id).addEventListener('click', fn);

  bindClick('menu-hdr-toggle', () => {
    state.menuOpen = !state.menuOpen;
    document.getElementById('menu-body').classList.toggle('hidden', !state.menuOpen);
    document.getElementById('menu-chevron').textContent = state.menuOpen ? '▾' : '▸';
  });

  const bindSection = (hdrId, bodyId, chevId) => {
    bindClick(hdrId, () => {
      const body = document.getElementById(bodyId);
      const nowHidden = body.classList.toggle('hidden');
      document.getElementById(chevId).textContent = nowHidden ? '▸' : '▾';
    });
  };
  bindSection('hdr-slices', 'body-slices', 'chev-slices');
  bindSection('hdr-detector', 'body-detector', 'chev-detector');
  bindSection('hdr-edges', 'body-edges', 'chev-edges');

  bindClick('btn-card-collapse', () => {
    state.cardOpen = !state.cardOpen;
    document.getElementById('card-body').classList.toggle('hidden', !state.cardOpen);
    document.getElementById('btn-card-collapse').textContent = state.cardOpen ? '▾' : '▸';
  });

  bindClick('tab-summary', () => { state.activeTab = 'summary'; syncUIButtons(); });
  bindClick('tab-diagnostics', () => { state.activeTab = 'diagnostics'; syncUIButtons(); });

  const switchMode = (m) => {
    if (state.mode === m) return;
    state.mode = m;
    if (m === 'full') {
      state.edgeTruth = true;
      state.edgePred = true;
      state.edgeFp = true;
      state.edgeFn = true;
      document.getElementById('sw-edge-truth').checked = true;
      document.getElementById('sw-edge-pred').checked = true;
      document.getElementById('sw-edge-fp').checked = true;
      document.getElementById('sw-edge-fn').checked = true;
      document.getElementById('body-edges').classList.remove('hidden');
      document.getElementById('chev-edges').textContent = '▾';
    } else if (m === 'st') {
      state.edgeTruth = false;
      state.edgePred = false;
      state.edgeFp = false;
      state.edgeFn = false;
      document.getElementById('sw-edge-truth').checked = false;
      document.getElementById('sw-edge-pred').checked = false;
      document.getElementById('sw-edge-fp').checked = false;
      document.getElementById('sw-edge-fn').checked = false;
    }
    refreshScene(true);
  };
  bindClick('btn-mode-cloud', () => switchMode('cloud'));
  bindClick('btn-mode-st', () => switchMode('st'));
  bindClick('btn-mode-full', () => switchMode('full'));
  bindClick('dock-go-st', () => switchMode('st'));
  bindClick('dock-back-cloud', () => switchMode('cloud'));

  bindClick('btn-model-v5', () => { state.model = 'v5'; refreshScene(false); });
  bindClick('btn-model-v6', () => { state.model = 'v6'; refreshScene(false); });

  document.getElementById('main-slider').addEventListener('input', (e) => {
    const v = parseFloat(e.target.value);
    if (state.mode === 'cloud') state.cloudOpacity = v;
    else state.threshold = v;
    refreshScene(false);
  });

  const setLayout = (lay) => {
    if (state.cloudLayout === lay) return;
    state.cloudLayout = lay;
    refreshScene(true);
  };
  bindClick('btn-lay-split', () => setLayout('split'));
  bindClick('btn-lay-overlay', () => setLayout('overlay'));
  bindClick('dock-split', () => setLayout('split'));
  bindClick('dock-overlay', () => setLayout('overlay'));

  const setPairing = (p) => {
    state.cloudPairing = p;
    if (p === 'quad') {
      state.showBoth = true;
      state.showGnnOnly = true;
      state.showDsOnly = true;
      state.showNeither = true;
      document.getElementById('sw-both').checked = true;
      document.getElementById('sw-gnn').checked = true;
      document.getElementById('sw-ds').checked = true;
      document.getElementById('sw-neither').checked = true;
    }
    refreshScene(false);
  };
  bindClick('btn-pair-recall', () => setPairing('recall'));
  bindClick('btn-pair-quad', () => setPairing('quad'));
  bindClick('dock-pair-recall', () => setPairing('recall'));
  bindClick('dock-pair-quad', () => setPairing('quad'));

  const bindSwitch = (id, prop) => {
    document.getElementById(id).addEventListener('change', (e) => {
      state[prop] = e.target.checked;
      refreshScene(false);
    });
  };
  bindSwitch('sw-both', 'showBoth');
  bindSwitch('sw-gnn', 'showGnnOnly');
  bindSwitch('sw-ds', 'showDsOnly');
  bindSwitch('sw-neither', 'showNeither');
  bindSwitch('sw-stations', 'showStations');
  bindSwitch('sw-bg-hits', 'showBgHits');
  bindSwitch('sw-st-target', 'stShowTarget');
  bindSwitch('sw-st-v5', 'stShowV5');
  bindSwitch('sw-st-v6', 'stShowV6');
  bindSwitch('sw-st-ds', 'stShowDs');
  bindSwitch('sw-st-other', 'stShowOther');
  bindSwitch('sw-edge-truth', 'edgeTruth');
  bindSwitch('sw-edge-pred', 'edgePred');
  bindSwitch('sw-edge-fp', 'edgeFp');
  bindSwitch('sw-edge-fn', 'edgeFn');
  bindSwitch('sw-edge-cand', 'edgeCand');
  bindSwitch('sw-full-mc', 'fullShowMc');

  bindClick('btn-geom-both', () => { state.cloudGeom = 'both'; refreshScene(false); });
  bindClick('btn-geom-lines', () => { state.cloudGeom = 'lines'; refreshScene(false); });
  bindClick('btn-geom-hits', () => { state.cloudGeom = 'hits'; refreshScene(false); });
  bindClick('btn-coord-truth', () => { state.cloudCoords = 'truth'; refreshScene(false); });
  bindClick('btn-coord-reco', () => { state.cloudCoords = 'reco'; refreshScene(false); });

  document.getElementById('sel-mom').addEventListener('change', (e) => { state.momBand = e.target.value; refreshScene(false); });
  document.getElementById('sel-occ').addEventListener('change', (e) => { state.occBand = e.target.value; refreshScene(false); });
  document.getElementById('sel-species').addEventListener('change', (e) => { state.species = e.target.value; refreshScene(false); });

  // Cohort filter & cohort stepping in EVENT (5,491) mode
  document.getElementById('sel-cohort').addEventListener('change', (e) => {
    state.stCohort = e.target.value;
    if (stStore) {
      const list = stStore.routes.cohorts[state.stCohort] || [];
      if (list.length > 0 && !list.includes(state.stIdx)) {
        state.stIdx = list[0];
      }
      refreshScene(true);
    }
  });

  const stepCohort = (delta) => {
    if (!stStore) return;
    const list = stStore.routes.cohorts[state.stCohort] || [];
    if (!list.length) return;
    const curPos = list.indexOf(state.stIdx);
    const nextPos = curPos >= 0
      ? (curPos + delta + list.length) % list.length
      : (delta >= 0 ? 0 : list.length - 1);
    state.stIdx = list[nextPos];
    refreshScene(true);
  };
  bindClick('btn-coh-prev', () => stepCohort(-1));
  bindClick('btn-coh-next', () => stepCohort(1));

  // Direct event number jump
  const jumpToInput = () => {
    const raw = parseInt(document.getElementById('inp-event-idx').value, 10);
    if (Number.isNaN(raw)) return;
    if (state.mode === 'st') state.stIdx = Math.max(0, Math.min(5490, raw));
    else if (state.mode === 'full') state.fullIdx = Math.max(0, Math.min(149858, raw));
    refreshScene(true);
  };
  bindClick('btn-jump-go', jumpToInput);
  document.getElementById('inp-event-idx').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') jumpToInput();
  });

  const caseIndices = [1613, 2589, 86];
  for (let k = 0; k < 3; k++) {
    bindClick(`dock-case-${k}`, () => {
      state.mode = 'st';
      state.stIdx = caseIndices[k];
      refreshScene(true);
    });
  }

  const stepEvent = (delta) => {
    if (state.mode === 'st') {
      state.stIdx = (state.stIdx + delta + 5491) % 5491;
    } else if (state.mode === 'full') {
      state.fullIdx = (state.fullIdx + delta + 149859) % 149859;
    }
    refreshScene(true);
  };
  bindClick('dock-ev-prev', () => stepEvent(-1));
  bindClick('dock-ev-next', () => stepEvent(1));

  const triggerPreset = (p) => {
    state.cameraPreset = p;
    if (state.mode === 'cloud') applyCameraPreset(p);
    else refreshScene(true);
  };
  bindClick('dock-cam-oblique', () => triggerPreset('oblique'));
  bindClick('dock-cam-xz', () => triggerPreset('xz'));
  bindClick('dock-cam-yz', () => triggerPreset('yz'));

  const zoomStep = (factor) => {
    const offset = camera.position.clone().sub(controls.target).divideScalar(factor);
    camera.position.copy(controls.target).add(offset);
    controls.update();
  };
  bindClick('dock-zoom-in', () => zoomStep(1.22));
  bindClick('dock-zoom-out', () => zoomStep(1 / 1.22));

  bindClick('dock-clean', () => { state.cleanMode = true; syncUIButtons(); });
  bindClick('btn-exit-clean', () => { state.cleanMode = false; syncUIButtons(); });
}

// ── Main Render Loop ──
function renderFrame() {
  requestAnimationFrame(renderFrame);

  const stage = document.getElementById('dsv-stage');
  const w = stage.clientWidth;
  const h = stage.clientHeight;
  if (canvas.width !== Math.floor(w * renderer.getPixelRatio()) ||
      canvas.height !== Math.floor(h * renderer.getPixelRatio())) {
    renderer.setSize(w, h, false);
  }

  const isDual = (state.mode === 'cloud' && state.cloudLayout === 'split');
  if (isDual) {
    const halfW = Math.floor(w * 0.5);
    camera.aspect = Math.max(0.2, halfW / Math.max(1, h));
    camera.updateProjectionMatrix();

    renderer.setScissorTest(true);
    renderer.setViewport(0, 0, halfW, h);
    renderer.setScissor(0, 0, halfW, h);
    renderer.render(sceneLeft, camera);

    renderer.setViewport(halfW, 0, w - halfW, h);
    renderer.setScissor(halfW, 0, w - halfW, h);
    renderer.render(sceneRight, camera);
  } else {
    camera.aspect = Math.max(0.2, w / Math.max(1, h));
    camera.updateProjectionMatrix();

    renderer.setScissorTest(false);
    renderer.setViewport(0, 0, w, h);
    renderer.render(sceneLeft, camera);
  }

  drawOrientationGizmo();
}

// ── Boot ──
(async function init() {
  wireControls();
  renderFrame();
  try {
    await loadCloudBundle();
    refreshScene(true);
    setLoadingBanner(false);
  } catch (err) {
    setLoadingBanner(true, `Load error: ${err.message}`);
  }
})();
