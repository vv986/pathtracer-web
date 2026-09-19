// GPU orchestration: pipelines, per-scene buffers, camera, UI, frame loop.

import { CORE1, CORE2, MESH_PART, PRIM_STUB, BLIT_WGSL } from './shaders.js?v=16';
import { SCENE_BUILDERS, BUNNY_MATERIALS } from './scenes.js?v=16';

const errBox = document.getElementById('err');
function showErr(msg) {
  const t = errBox.textContent ? errBox.textContent + '\n' + msg : String(msg);
  errBox.textContent = t.slice(-4000);
}
window.onerror = (m, s, l, c) => { showErr(`JS error: ${m} (${l}:${c})`); };

const SPP_PER_FRAME = 2;
const MAX_TOTAL_SAMPLES = 4000;

const canvas = document.getElementById('c');
const hud = document.getElementById('hud');

let device, ctx, format;
let uniBuf, accumBuf, histBuf, histZero, outTex;
let primPipeline, meshPipeline, blitPipeline;
let fallbackTex, fallbackView;
let W = 0, H = 0;
let dummyBuf;

const sceneCache = {};
let scene = null;
let cam = null;
let meshMatName = 'chrome';

let frame = 0, totalSamples = 0;
let lastT = 0, tickCount = 0, fpsTimer = 0, fps = 0;
let lastTickAt = 0;
let captureFlag = false;
let building = false;
let INIT_DPR = 1; // frozen at init: a flapping devicePixelRatio would reset accumulation every frame

// adaptive resolution governor: keeps heavy scenes at ~30 fps by scaling the
// internal render resolution (the temporal denoiser hides the softness)
const TARGET_MS = 33;          // ~30 fps budget
let dynScale = 1.0;            // current dynamic scale (multiplies scene.resScale)
let frameMsEma = 16;
let govCooldown = 0;           // timestamp (s) of last adjustment

function desiredSize() {
  const dpr = INIT_DPR;
  let w = Math.max(320, Math.floor((canvas.clientWidth || window.innerWidth) * dpr));
  let h = Math.max(240, Math.floor((canvas.clientHeight || window.innerHeight) * dpr));
  const cap = 1700;
  if (w > cap) { h = Math.floor(h * cap / w); w = cap; }
  return [w, h];
}

function makeStorageF32(arr) {
  const b = device.createBuffer({ size: Math.max(16, arr.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  device.queue.writeBuffer(b, 0, arr);
  return b;
}

function makeStorageU32(arr) {
  const b = device.createBuffer({ size: Math.max(16, arr.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  device.queue.writeBuffer(b, 0, arr);
  return b;
}

async function makeModule(code, label) {
  const m = device.createShaderModule({ code });
  const info = await m.getCompilationInfo();
  const errs = info.messages.filter((x) => x.type === 'error');
  if (errs.length) {
    showErr(errs.map((x) => `[${label}] ${x.lineNum}:${x.linePos} ${x.message}`).join('\n'));
    throw new Error(label + ' shader compile failed');
  }
  return m;
}

function cameraPos() {
  const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
  const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw);
  return [
    cam.target[0] + cam.dist * cp * sy,
    cam.target[1] + cam.dist * sp,
    cam.target[2] + cam.dist * cp * cy,
  ];
}

function clampCam() {
  const base = cam.baseYaw !== undefined ? cam.baseYaw : Math.PI;
  cam.yaw = Math.max(base - cam.yawRange, Math.min(base + cam.yawRange, cam.yaw));
  cam.pitch = Math.max(cam.pitchRange[0], Math.min(cam.pitchRange[1], cam.pitch));
  cam.dist = Math.max(cam.distRange[0], Math.min(cam.distRange[1], cam.dist));
}

// ---- buffers & bind groups ----

function createFrameBuffers(w, h) {
  [W, H] = [w, h];
  canvas.width = W; canvas.height = H;
  accumBuf = device.createBuffer({ size: W * H * 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  histBuf = device.createBuffer({ size: W * H * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  histZero = new Float32Array(W * H * 4);
  outTex = device.createTexture({ size: [W, H], format: 'rgba16float',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING });
}

function buildBindGroup(rt) {
  const entries = [
    { binding: 0, resource: { buffer: uniBuf } },
    { binding: 1, resource: { buffer: accumBuf } },
    { binding: 2, resource: outTex.createView() },
    { binding: 3, resource: { buffer: rt.buffers.quadBuf } },
    { binding: 4, resource: { buffer: rt.buffers.sphereBuf } },
    { binding: 9, resource: { buffer: histBuf } },
  ];
  if (rt.def.type === 'mesh') {
    entries.push(
      { binding: 5, resource: { buffer: rt.buffers.nodeBuf } },
      { binding: 6, resource: { buffer: rt.buffers.vertBuf } },
      { binding: 7, resource: { buffer: rt.buffers.normBuf } },
      { binding: 8, resource: { buffer: rt.buffers.idxBuf } },
      { binding: 10, resource: { buffer: rt.buffers.palAlbBuf } },
      { binding: 11, resource: { buffer: rt.buffers.palPrmBuf } },
      { binding: 12, resource: { buffer: rt.buffers.texBuf } },
      { binding: 30, resource: rt.buffers.atlasViews[0] },
      { binding: 31, resource: rt.buffers.atlasViews[1] },
      { binding: 32, resource: rt.buffers.atlasViews[2] },
      { binding: 33, resource: rt.buffers.atlasViews[3] },
      { binding: 34, resource: rt.buffers.atlasViews[4] },
      { binding: 35, resource: rt.buffers.atlasViews[5] },
      { binding: 36, resource: rt.buffers.atlasViews[6] },
      { binding: 37, resource: rt.buffers.atlasViews[7] },
      { binding: 38, resource: rt.buffers.sampler },
      { binding: 15, resource: { buffer: rt.buffers.matUVBuf } },
      { binding: 16, resource: { buffer: rt.buffers.tanBuf } },
      { binding: 40, resource: rt.buffers.nrmViews[0] },
      { binding: 41, resource: rt.buffers.nrmViews[1] },
      { binding: 42, resource: rt.buffers.nrmViews[2] },
      { binding: 43, resource: rt.buffers.nrmViews[3] },
      { binding: 44, resource: rt.buffers.nrmViews[4] },
      { binding: 45, resource: rt.buffers.nrmViews[5] },
      { binding: 46, resource: rt.buffers.nrmViews[6] },
      { binding: 47, resource: rt.buffers.nrmViews[7] },
    );
  }
  const layout = rt.def.type === 'mesh' ? meshPipeline.getBindGroupLayout(0) : primPipeline.getBindGroupLayout(0);
  return device.createBindGroup({ layout, entries });
}

// ---- scene lifecycle ----

async function activateScene(name) {
  if (building) return;
  building = true;
  const buildStart = performance.now();
  hud.textContent = `构建场景 ${name} …(大场景 BVH 构建可能需要十几秒)`;
  await new Promise((r) => setTimeout(r, 30)); // let the HUD paint

  if (!sceneCache[name]) {
    const def = await SCENE_BUILDERS[name]();
    const rt = { def, buffers: {} };
    rt.buffers.sphereBuf = makeStorageF32(def.spheres);
    if (def.type === 'prim') {
      rt.buffers.quadBuf = makeStorageF32(def.quads);
    } else {
      const m = def.mesh;
      rt.buffers.quadBuf = dummyBuf;
      rt.buffers.nodeBuf = makeStorageF32(m.nodes);
      rt.buffers.vertBuf = makeStorageF32(m.verts4);
      rt.buffers.normBuf = makeStorageF32(m.vnorm);
      rt.buffers.idxBuf = makeStorageU32(m.idx);
      rt.buffers.palAlbBuf = makeStorageF32(m.palAlb);
      rt.buffers.palPrmBuf = makeStorageF32(m.palPrm);
      rt.buffers.texBuf = makeStorageF32(m.tex4);
      rt.buffers.tanBuf = makeStorageF32(m.tan4);
      // atlas pages (graceful fallback to flat gray when a model ships without textures)
      rt.buffers.matUVBuf = makeStorageF32(def.atlas ? def.atlas.matUV : new Float32Array(48));
      const views = [];
      if (def.atlas) {
        for (const b of def.atlas.bitmaps) {
          const tex = device.createTexture({ size: [b.width, b.height], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT });
          device.queue.copyExternalImageToTexture({ source: b }, { texture: tex }, [b.width, b.height]);
          views.push(tex.createView());
        }
      }
      while (views.length < 8) views.push(fallbackView);
      rt.buffers.atlasViews = views;
      const nrmViews = [];
      if (def.atlas && def.atlas.nrmBitmaps) {
        for (const b of def.atlas.nrmBitmaps) {
          const tex = device.createTexture({ size: [b.width, b.height], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT });
          device.queue.copyExternalImageToTexture({ source: b }, { texture: tex }, [b.width, b.height]);
          nrmViews.push(tex.createView());
        }
      }
      while (nrmViews.length < 8) nrmViews.push(fallbackView);
      rt.buffers.nrmViews = nrmViews;
      rt.buffers.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    }
    sceneCache[name] = rt;
  }
  scene = sceneCache[name];
  const c = scene.def.cam;
  cam = { target: [...c.target], baseYaw: c.baseYaw, yaw: c.yaw, pitch: c.pitch, dist: c.dist,
          yawRange: c.yawRange, pitchRange: c.pitchRange, distRange: c.distRange };

  scene.bindGroup = buildBindGroup(scene);
  device.queue.writeBuffer(histBuf, 0, histZero); // fresh history for a new scene

  frame = 0; totalSamples = 0;
  dynScale = 1.0; frameMsEma = 16; govCooldown = performance.now() / 1000 + 1.0;
  document.querySelectorAll('#ui button[data-scene]').forEach((b) => {
    b.classList.toggle('active', b.dataset.scene === name);
  });
  const buildMs = ((performance.now() - buildStart) / 1000).toFixed(1);
  if (scene.def.numTris > 0) {
    hud.textContent = `${scene.def.name} — ${scene.def.numTris.toLocaleString()} 三角形,BVH 就绪 (${buildMs}s)`;
  }
  building = false;
}

// ---- frame loop ----

const uniformScratch = new ArrayBuffer(128);

function tick(t) {
  if (document.hidden) { requestAnimationFrame(tick); return; }  // never burn GPU in background
  lastTickAt = performance.now();
  const dt = lastT ? (t - lastT) / 1000 : 0;
  lastT = t;
  tickCount++;
  if (t - fpsTimer > 1000) { fps = tickCount; tickCount = 0; fpsTimer = t; }

  // governor: adjust dynamic resolution toward the ~30 fps budget (time-based)
  const ms = dt * 1000;
  frameMsEma = frameMsEma * 0.9 + Math.min(ms, 250) * 0.1;
  const nowS = performance.now() / 1000;
  if (scene && totalSamples < MAX_TOTAL_SAMPLES && nowS - govCooldown > 0.5) {
    if (frameMsEma > TARGET_MS * 1.15 && dynScale > 0.3) {
      dynScale = Math.max(0.3, dynScale * 0.8);
      govCooldown = nowS;
    } else if (frameMsEma < TARGET_MS * 0.55) {
      const cap = scene.def.resScale ?? 1;
      if (dynScale < cap) { dynScale = Math.min(cap, dynScale * 1.15); govCooldown = nowS; }
    }
  }

  let [bw, bh] = desiredSize();
  // scale the internal resolution while PRESERVING the window's aspect ratio —
  // independent per-axis clamps here were causing horizontal stretching
  let s = (scene ? (scene.def.resScale ?? 1) : 1) * dynScale;
  s = Math.max(s, 320 / bw, 240 / bh); // floor: keep the render at least ~320 wide, proportionally
  let tw = Math.round(bw * s), th = Math.round(bh * s);
  const cap = 1700;
  if (tw > cap) { th = Math.round(th * cap / tw); tw = cap; } // proportional
  if (Math.abs(tw - W) > 2 || Math.abs(th - H) > 2) {
    createFrameBuffers(tw, th);
    frame = 0;
    blitBind = device.createBindGroup({
      layout: blitPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: outTex.createView() }],
    });
    if (scene) scene.bindGroup = buildBindGroup(scene);
  }
  if (!scene) { requestAnimationFrame(tick); return; }

  const pos = cameraPos();
  const u32 = new Uint32Array(uniformScratch, 0, 4);
  const f32 = new Float32Array(uniformScratch);
  u32.set([W, H, frame, SPP_PER_FRAME]);
  f32.set([pos[0], pos[1], pos[2], scene.def.fov], 4);
  f32.set([cam.target[0], cam.target[1], cam.target[2], scene.def.exposure], 8);
  f32.set([scene.def.quads ? scene.def.quads.length / 20 : 0,
           scene.def.spheres.length / 12,
           scene.def.lightQuadIdx, scene.def.lightSphereIdx], 12);
  f32.set([scene.def.lightQuadArea, scene.def.skyMode, scene.def.numTris, scene.def.maxDepth ?? 12], 16);
  f32.set([scene.def.sunDir[0], scene.def.sunDir[1], scene.def.sunDir[2], 0], 20);
  device.queue.writeBuffer(uniBuf, 0, uniformScratch);

  const enc = device.createCommandEncoder();
  if (totalSamples < MAX_TOTAL_SAMPLES) {
    const cp = enc.beginComputePass();
    cp.setPipeline(scene.def.type === 'mesh' ? meshPipeline : primPipeline);
    cp.setBindGroup(0, scene.bindGroup);
    cp.dispatchWorkgroups(Math.ceil(W / 8), Math.ceil(H / 8));
    cp.end();
    frame++;
    totalSamples += SPP_PER_FRAME;
  }
  const rp = enc.beginRenderPass({
    colorAttachments: [{
      view: ctx.getCurrentTexture().createView(),
      loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 },
    }],
  });
  rp.setPipeline(blitPipeline);
  rp.setBindGroup(0, blitBind);
  rp.draw(3);
  rp.end();
  device.queue.submit([enc.finish()]);

  // screenshot: capture the just-presented frame synchronously, save as PNG
  if (captureFlag && scene) {
    captureFlag = false;
    try {
      const c2 = document.createElement('canvas');
      c2.width = W; c2.height = H;
      c2.getContext('2d').drawImage(canvas, 0, 0);
      c2.toBlob((b) => {
        if (!b) return;
        const a = document.createElement('a');
        a.href = URL.createObjectURL(b);
        a.download = `pathtracer-${scene.def.name}-${totalSamples}spp.png`;
        a.click();
        setTimeout(() => URL.revokeObjectURL(a.href), 3000);
      }, 'image/png');
    } catch (e) { showErr('截图失败: ' + String(e)); }
  }

  const stat = totalSamples >= MAX_TOTAL_SAMPLES
    ? `已收敛 · ${totalSamples} spp`
    : `累积 ${totalSamples} spp · ${fps} fps`;
  hud.textContent = `${scene.def.name} — ${W}×${H} (dyn ${Math.round(dynScale * 100)}% × base ${Math.round((scene.def.resScale ?? 1) * 100)}%) · ${stat}` +
    (scene.def.numTris > 0 ? ` · ${scene.def.numTris.toLocaleString()} tris` : '');

  requestAnimationFrame(tick);
}

let blitBind = null;

// ---- init ----

async function init() {
  if (!navigator.gpu) throw new Error('此浏览器不支持 WebGPU,请使用最新版 Chrome / Edge / Firefox');
  // retry: after a GPU crash the adapter may be briefly unavailable
  let adapter = null;
  for (let i = 0; i < 6; i++) {
    adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) adapter = await navigator.gpu.requestAdapter();
    if (adapter) break;
    showErr(`GPU 适配器暂时不可用(${i + 1}/6),2 秒后重试…若持续失败请完全重启浏览器`);
    await new Promise((r) => setTimeout(r, 2000));
  }
  if (!adapter) throw new Error('未找到 GPU 适配器——若之前能正常显示,请完全退出浏览器(所有窗口)后重开;若从未成功过,请确认浏览器为最新版且已开启硬件加速');
  device = await adapter.requestDevice({ requiredLimits: {
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    maxBufferSize: adapter.limits.maxBufferSize,
    maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage,
  } });
  device.addEventListener('uncapturederror', (e) => showErr('GPU error: ' + e.error.message + '\n' + (e.error.reason || '')));
  device.lost.then((info) => showErr('GPU 设备丢失(' + (info.reason || 'unknown') + '),请刷新页面恢复。'));

  ctx = canvas.getContext('webgpu');
  format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format, alphaMode: 'opaque' });
  INIT_DPR = Math.min(window.devicePixelRatio || 1, 1.5);

  uniBuf = device.createBuffer({ size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  dummyBuf = device.createBuffer({ size: 128, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  fallbackTex = device.createTexture({ size: [1, 1], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT });
  device.queue.writeTexture({ texture: fallbackTex }, new Uint8Array([120, 120, 120, 255]), { bytesPerRow: 4 }, [1, 1]);
  fallbackView = fallbackTex.createView();

  const primModule = await makeModule(CORE1 + PRIM_STUB + CORE2, 'prim');
  const meshModule = await makeModule(CORE1 + MESH_PART + CORE2, 'mesh');
  const blitModule = await makeModule(BLIT_WGSL, 'blit');

  primPipeline = device.createComputePipeline({ layout: 'auto', compute: { module: primModule, entryPoint: 'main' } });
  meshPipeline = device.createComputePipeline({ layout: 'auto', compute: { module: meshModule, entryPoint: 'main' } });
  blitPipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: blitModule, entryPoint: 'vs' },
    fragment: { module: blitModule, entryPoint: 'fs', targets: [{ format }] },
    primitive: { topology: 'triangle-list' },
  });

  const [iw, ih] = desiredSize();
  createFrameBuffers(iw, ih);
  blitBind = device.createBindGroup({
    layout: blitPipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: outTex.createView() }],
  });

  await activateScene('field');
  requestAnimationFrame(tick);
}

// ---- interaction ----

let dragging = false, px = 0, py = 0;
canvas.addEventListener('pointerdown', (e) => { dragging = true; px = e.clientX; py = e.clientY; });
window.addEventListener('pointerup', () => { dragging = false; });
window.addEventListener('pointermove', (e) => {
  if (!dragging || !cam) return;
  cam.yaw -= (e.clientX - px) * 0.003;
  cam.pitch += (e.clientY - py) * 0.003;
  clampCam();
  px = e.clientX; py = e.clientY;
  frame = 0; totalSamples = 0;
});
canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  if (!cam) return;
  cam.dist *= Math.exp(e.deltaY * 0.001);
  clampCam();
  frame = 0; totalSamples = 0;
}, { passive: false });

document.querySelectorAll('#ui button[data-scene]').forEach((b) => {
  b.addEventListener('click', async () => {
    try { await activateScene(b.dataset.scene); } catch (e) { showErr(String(e)); building = false; }
  });
});
// keyboard shortcuts: 1-5 switch scenes
const KEY_SCENES = { 1: 'field', 2: 'bunny', 3: 'sponza', 4: 'bistro', 5: 'cornell' };
window.addEventListener('keydown', (e) => {
  const name = KEY_SCENES[e.key];
  if (name) {
    const btn = document.querySelector(`button[data-scene="${name}"]`);
    if (btn) btn.click();
  }
});
document.getElementById('shotBtn').addEventListener('click', () => { captureFlag = true; });

// sun elevation + exposure sliders (live: sun sphere rewritten in the spheres buffer)
const sunEl = document.getElementById('sunEl');
const expSl = document.getElementById('expSl');
function applySunElevation(deg) {
  if (!scene) return;
  const el = deg * Math.PI / 180;
  const az = scene.def.sunAzimuth !== undefined ? scene.def.sunAzimuth : Math.atan2(-0.45, -0.55);
  const dir = [Math.cos(el) * Math.sin(az), Math.sin(el), Math.cos(el) * Math.cos(az)];
  scene.def.sunDir = dir;
  const si = scene.def.lightSphereIdx;
  if (si >= 0 && scene.buffers.sphereBuf) {
    const c = dir.map((d) => d * 8000);
    const f = new Float32Array([c[0], c[1], c[2], 600]);
    device.queue.writeBuffer(scene.buffers.sphereBuf, si * 48, f);
  }
  frame = 0; totalSamples = 0;
}
sunEl.addEventListener('input', () => {
  document.getElementById('sunVal').textContent = sunEl.value + '°';
  if (cam) applySunElevation(+sunEl.value);
});
expSl.addEventListener('input', () => {
  document.getElementById('expVal').textContent = expSl.value + '%';
  if (scene) { scene.def.exposure = expSl.value / 100; }
});
window.addEventListener('keydown', (e) => { if (e.key === 'p' || e.key === 'P') captureFlag = true; });
const matSel = document.getElementById('matSel');
if (matSel) {
  matSel.addEventListener('change', () => {
    meshMatName = matSel.value;
    if (scene && scene.def.name === 'Bunny') {
      const m = BUNNY_MATERIALS[meshMatName];
      const palAlb = new Float32Array([m.albedo[0], m.albedo[1], m.albedo[2], m.mtype]);
      const palPrm = new Float32Array([m.rough, m.ior, 0, 0]);
      device.queue.writeBuffer(scene.buffers.palAlbBuf, 0, palAlb);
      device.queue.writeBuffer(scene.buffers.palPrmBuf, 0, palPrm);
      frame = 0; totalSamples = 0;
    }
  });
}
window.addEventListener('resize', () => { /* handled in tick via desiredSize check */ });

// Watchdog: keep rendering when rAF is throttled (occluded/automation contexts).
setInterval(() => {
  if (device && !document.hidden && performance.now() - lastTickAt > 400 && scene) tick(performance.now());
}, 120);

// debug hook (automation/testing): tweak the live camera from the console
window.__dbg = {
  get cam() { return cam; },
  reset() { frame = 0; totalSamples = 0; },
  sceneInfo() {
    if (!scene) return null;
    const m = scene.def.mesh;
    if (!m) return { name: scene.def.name, type: scene.def.type };
    let mnX = 1e30, mxX = -1e30, mnY = 1e30, mxY = -1e30, mnZ = 1e30, mxZ = -1e30;
    for (let i = 0; i < m.verts4.length; i += 4) {
      mnX = Math.min(mnX, m.verts4[i]); mxX = Math.max(mxX, m.verts4[i]);
      mnY = Math.min(mnY, m.verts4[i + 1]); mxY = Math.max(mxY, m.verts4[i + 1]);
      mnZ = Math.min(mnZ, m.verts4[i + 2]); mxZ = Math.max(mxZ, m.verts4[i + 2]);
    }
    return { name: scene.def.name, numTris: scene.def.numTris,
             x: [mnX.toFixed(2), mxX.toFixed(2)], y: [mnY.toFixed(2), mxY.toFixed(2)],
             z: [mnZ.toFixed(2), mxZ.toFixed(2)] };
  },
  // read back the GPU vertex buffer and measure its span (ground truth for upload)
  async gpuVertSpan() {
    if (!scene || !scene.buffers.vertBuf) return null;
    const src = scene.buffers.vertBuf;
    const staging = device.createBuffer({ size: src.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(src, 0, staging, 0, src.size);
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    await staging.mapAsync(GPUMapMode.READ);
    const f = new Float32Array(staging.getMappedRange());
    let mnX = 1e30, mxX = -1e30, mnY = 1e30, mxY = -1e30, mnZ = 1e30, mxZ = -1e30;
    for (let i = 0; i < f.length; i += 4) {
      mnX = Math.min(mnX, f[i]); mxX = Math.max(mxX, f[i]);
      mnY = Math.min(mnY, f[i + 1]); mxY = Math.max(mxY, f[i + 1]);
      mnZ = Math.min(mnZ, f[i + 2]); mxZ = Math.max(mxZ, f[i + 2]);
    }
    staging.unmap();
    staging.destroy();
    return { x: [mnX.toFixed(2), mxX.toFixed(2)], y: [mnY.toFixed(2), mxY.toFixed(2)], z: [mnZ.toFixed(2), mxZ.toFixed(2)] };
  },
  // read back the GPU index + node buffers and validate
  async gpuIdxCheck() {
    if (!scene || !scene.buffers.idxBuf) return null;
    const numTris = scene.def.numTris;
    const numVerts = scene.def.mesh.verts4.length / 4;
    const idxStaging = device.createBuffer({ size: numTris * 12, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(scene.buffers.idxBuf, 0, idxStaging, 0, numTris * 12);
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    await idxStaging.mapAsync(GPUMapMode.READ);
    const idx = new Uint32Array(idxStaging.getMappedRange());
    let bad = 0, maxIdx = 0, degenerate = 0;
    const verts = scene.def.mesh.verts4;
    for (let t = 0; t < numTris; t++) {
      const a = idx[t * 3], b = idx[t * 3 + 1], c = idx[t * 3 + 2];
      if (a >= numVerts || b >= numVerts || c >= numVerts) bad++;
      maxIdx = Math.max(maxIdx, a, b, c);
      const ax = verts[a * 4], ay = verts[a * 4 + 1], az = verts[a * 4 + 2];
      const bx = verts[b * 4], by = verts[b * 4 + 1], bz = verts[b * 4 + 2];
      const cx = verts[c * 4], cy = verts[c * 4 + 1], cz = verts[c * 4 + 2];
      const e1 = [bx - ax, by - ay, bz - az], e2 = [cx - ax, cy - ay, cz - az];
      const n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
      if (Math.hypot(...n) < 1e-10) degenerate++;
    }
    const first12 = Array.from(idx.slice(0, 12));
    idxStaging.unmap();
    idxStaging.destroy();
    // nodes: check meta ranges
    const nodeStaging = device.createBuffer({ size: scene.buffers.nodeBuf.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc2 = device.createCommandEncoder();
    enc2.copyBufferToBuffer(scene.buffers.nodeBuf, 0, nodeStaging, 0, scene.buffers.nodeBuf.size);
    device.queue.submit([enc2.finish()]);
    await device.queue.onSubmittedWorkDone();
    await nodeStaging.mapAsync(GPUMapMode.READ);
    const nf = new Float32Array(nodeStaging.getMappedRange());
    const numNodes = nf.length / 12;
    let badNode = 0;
    for (let i = 0; i < numNodes; i++) {
      const count = nf[i * 12 + 10];
      const a = nf[i * 12 + 8];
      if (count > 0 && (a < 0 || a + count > numTris)) badNode++;
      if (count === 0 && (a < 0 || a >= numNodes || nf[i * 12 + 9] < 0 || nf[i * 12 + 9] >= numNodes)) badNode++;
    }
    nodeStaging.unmap();
    nodeStaging.destroy();
    return { numVerts, numTris, maxIdx, badIdx: bad, degenerate, numNodes, badNode, first12 };
  },
  // read back the uniform buffer (ground truth for what the shader receives)
  async uniDump() {
    const st = device.createBuffer({ size: 128, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(uniBuf, 0, st, 0, 128);
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    await st.mapAsync(GPUMapMode.READ);
    const mapped = st.getMappedRange();
    const f = new Float32Array(mapped);
    const u = new Uint32Array(mapped);
    const out = {
      dims: [u[0], u[1], u[2], u[3]],
      cam: [f[4].toFixed(1), f[5].toFixed(1), f[6].toFixed(1), f[7].toFixed(2)],
      counts: [f[12], f[13], f[14], f[15]],
      light: [f[16], f[17], f[18], f[19]],
    };
    st.unmap();
    st.destroy();
    return out;
  },
  // statistics over the accumulation buffer: NaN pixels / near-black pixels
  async accumStats() {
    if (!accumBuf) return null;
    const st = device.createBuffer({ size: accumBuf.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(accumBuf, 0, st, 0, accumBuf.size);
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    await st.mapAsync(GPUMapMode.READ);
    const f = new Float32Array(st.getMappedRange());
    let nan = 0, dark = 0;
    const total = W * H;
    for (let i = 0; i < total; i++) {
      const r = f[i * 4], g = f[i * 4 + 1], b = f[i * 4 + 2];
      if (Number.isNaN(r) || Number.isNaN(g) || Number.isNaN(b)) nan++;
      else if (r * r + g * g + b * b < 1e-6) dark++;
    }
    st.unmap();
    st.destroy();
    return { total, nanFrac: (nan / total).toFixed(3), darkFrac: (dark / total).toFixed(3) };
  },
  // statistics over the mesh normal buffer: NaN / near-zero normals
  async vnormStats() {
    if (!scene || !scene.buffers.normBuf) return null;
    const st = device.createBuffer({ size: scene.buffers.normBuf.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(scene.buffers.normBuf, 0, st, 0, scene.buffers.normBuf.size);
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    await st.mapAsync(GPUMapMode.READ);
    const f = new Float32Array(st.getMappedRange());
    const n = f.length / 4;
    let nan = 0, tiny = 0;
    for (let i = 0; i < n; i++) {
      const x = f[i * 4], y = f[i * 4 + 1], z = f[i * 4 + 2];
      if (Number.isNaN(x) || Number.isNaN(y) || Number.isNaN(z)) nan++;
      else if (x * x + y * y + z * z < 1e-8) tiny++;
    }
    st.unmap();
    st.destroy();
    return { verts: n, nanNormals: nan, tinyNormals: tiny };
  },
};

init().catch((e) => showErr(errBox.textContent ? errBox.textContent + '\n' + String(e) : String(e)));
