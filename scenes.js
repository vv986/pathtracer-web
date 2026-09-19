// Scene definitions. Geometry packs into typed arrays that map 1:1 onto the
// WGSL storage-buffer structs in shaders.js.
//
// Mesh scenes (bunny/sponza/bistro): indexed vertices + material palette.
//   verts4 = xyz + material slot, palette = per-slot {albedo, mtype, rough, ior}.
// Material types: 0 diffuse, 1 metal (GGX), 2 glass, 3 emissive.

import { parseOBJ, parseGLTF, buildBVHIndexed } from './bvh.js?v=15';

const SUN_DIR = (() => {
  const l = Math.hypot(-0.45, 0.38, -0.55);
  return [-0.45 / l, 0.38 / l, -0.55 / l];
})();

function cross3(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
function dot3(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function norm3(a) {
  const l = Math.hypot(a[0], a[1], a[2]);
  return [a[0] / l, a[1] / l, a[2] / l];
}

// fetch with live download progress in the HUD; transparently prefers .gz
async function fetchBytes(url) {
  const hud = document.getElementById('hud');
  let res = await fetch(url + '.gz').catch(() => null);
  let body = null, total = 0;
  if (res && res.ok) {
    body = res.body.pipeThrough(new DecompressionStream('gzip'));
  } else {
    res = await fetch(url);
    if (!res.ok) throw new Error('无法加载 ' + url + ' (' + res.status + ')');
    body = res.body;
    total = +res.headers.get('Content-Length') || 0;
  }
  const reader = body.getReader();
  const chunks = [];
  let recv = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    recv += value.length;
    const mb = (recv / 1048576).toFixed(1);
    const tot = total ? ' / ' + (total / 1048576).toFixed(1) + ' MB' : ' MB';
    if (hud) hud.textContent = `下载模型 ${url.split('/').pop()} — ${mb}${tot} …`;
    await new Promise((r) => setTimeout(r, 0)); // let the HUD paint
  }
  const out = new Uint8Array(recv);
  let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.length; }
  return out;
}

function quad(Q, U, V, mtype, tex, alb, rough = 0) {
  const n = cross3(U, V);
  const nn = norm3(n);
  const D = dot3(nn, Q);
  return [
    Q[0], Q[1], Q[2], mtype,
    U[0], U[1], U[2], tex,
    V[0], V[1], V[2], 0,
    nn[0], nn[1], nn[2], D,
    alb[0], alb[1], alb[2], rough,
  ];
}

function sph(c, r, mtype, tex, rough, ior, alb) {
  return [c[0], c[1], c[2], r, mtype, tex, rough, ior, alb[0], alb[1], alb[2], 0];
}

function addBox(arr, a, b, deg, tr, mtype, tex, alb, rough) {
  const [x0, y0, z0] = a, [x1, y1, z1] = b;
  const th = deg * Math.PI / 180, c = Math.cos(th), s = Math.sin(th);
  const corner = (p) => [c * p[0] + s * p[2], p[1], -s * p[0] + c * p[2]]
    .map((v, i) => v + tr[i]);
  const faces = [
    [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]],
    [[x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1]],
    [[x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0]],
    [[x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]],
    [[x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0]],
    [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]],
  ];
  for (const [c0, c1, , c3] of faces) {
    const q0 = corner(c0), q1 = corner(c1), q3 = corner(c3);
    arr.push(quad(q0,
      [q1[0] - q0[0], q1[1] - q0[1], q1[2] - q0[2]],
      [q3[0] - q0[0], q3[1] - q0[1], q3[2] - q0[2]],
      mtype, tex, alb, rough));
  }
}

// pack mesh scene from BVH output + palette; matSlotPerVert: Uint32Array or null (=0)
function packMeshScene(mesh, palette) {
  const { positions, normals, tuvs, tans, tris } = mesh;
  const numVerts = normals.length / 3;
  const verts4 = new Float32Array(numVerts * 4);
  // normals MUST be padded to a 16-byte stride too: the shader reads array<vec4f>,
  // and an unpadded 3-float buffer desyncs every vertex past the first third
  // quantize normals/uv/tangents to packed i16 pairs (halves bandwidth; precision 3e-5)
  const pack16 = (a, b) => (((a & 0xFFFF) | ((b & 0xFFFF) << 16)) >>> 0);
  const clampI16 = (v) => Math.max(-32768, Math.min(32767, Math.round(v)));
  const vnP = new Uint32Array(numVerts * 2);
  const txP = new Uint32Array(numVerts);
  const tnP = new Uint32Array(numVerts * 2);
  for (let i = 0; i < numVerts; i++) {
    verts4[i * 4] = positions[i * 3];
    verts4[i * 4 + 1] = positions[i * 3 + 1];
    verts4[i * 4 + 2] = positions[i * 3 + 2];
    const nx = clampI16(Math.max(-1, Math.min(1, normals[i * 3])) * 32767);
    const ny = clampI16(Math.max(-1, Math.min(1, normals[i * 3 + 1])) * 32767);
    const nz = clampI16(Math.max(-1, Math.min(1, normals[i * 3 + 2])) * 32767);
    vnP[i * 2] = pack16(nx, ny);
    vnP[i * 2 + 1] = pack16(nz, 0);
    const uu = tuvs ? tuvs[i * 2] : 0;
    const vv = tuvs ? tuvs[i * 2 + 1] : 0;
    txP[i] = pack16(clampI16((uu - Math.floor(uu)) * 32767), clampI16((vv - Math.floor(vv)) * 32767));
    if (tans && (tans[i * 4] || tans[i * 4 + 1] || tans[i * 4 + 2])) {
      const tl = Math.hypot(tans[i * 4], tans[i * 4 + 1], tans[i * 4 + 2]) || 1;
      tnP[i * 2] = pack16(clampI16(tans[i * 4] / tl * 32767), clampI16(tans[i * 4 + 1] / tl * 32767));
      tnP[i * 2 + 1] = pack16(clampI16(tans[i * 4 + 2] / tl * 32767), clampI16(tans[i * 4 + 3] * 32767));
    }
  }
  if (mesh.matSlot) {
    for (let i = 0; i < numVerts; i++) verts4[i * 4 + 3] = mesh.matSlot[i];
  }
  const palAlb = new Float32Array(palette.length * 4);
  const palPrm = new Float32Array(palette.length * 4);
  palette.forEach((m, i) => {
    palAlb.set([m.albedo[0], m.albedo[1], m.albedo[2], m.mtype], i * 4);
    palPrm.set([m.rough, m.ior, 0, 0], i * 4);
  });
  return { verts4, vnormP: vnP, texP: txP, tanP: tnP, idx: mesh.idx, nodes: mesh.nodes, numTris: mesh.numTris, palAlb, palPrm };
}

// ---------------- Cornell Box ----------------

function buildCornell() {
  const quads = [];
  const WHITE = [0.73, 0.73, 0.73];
  quads.push(quad([200, 0, 0], [0, 600, 0], [0, 0, 556], 0, 0, [0.12, 0.45, 0.15]));
  quads.push(quad([900, 0, 0], [0, 600, 0], [0, 0, 556], 0, 0, [0.65, 0.05, 0.05]));
  quads.push(quad([200, 0, 0], [700, 0, 0], [0, 0, 556], 0, 0, WHITE));
  quads.push(quad([200, 600, 0], [700, 0, 0], [0, 0, 556], 0, 0, WHITE));
  quads.push(quad([200, 0, 556], [700, 0, 0], [0, 600, 0], 0, 0, WHITE));
  const lightIdx = quads.length;
  quads.push(quad([280, 599, 130], [340, 0, 0], [0, 0, 300], 3, 0, [4, 4, 4]));
  addBox(quads, [0, 0, 0], [260, 320, 260], 20, [540, 0, 140], 0, 0, WHITE);
  addBox(quads, [0, 0, 0], [220, 220, 220], -25, [240, 0, 280], 0, 0, WHITE);

  const spheres = [
    sph([280, 70, 80], 70, 2, 0, 0, 1.5, [1, 1, 1]),
  ];

  return {
    name: 'Cornell Box', type: 'prim',
    quads: new Float32Array(quads.flat()),
    spheres: new Float32Array(spheres.flat()),
    lightQuadIdx: lightIdx, lightSphereIdx: -1, lightQuadArea: 340 * 300,
    skyMode: 0, sunDir: [0, 1, 0], numTris: 0,
    resScale: 1.0, maxDepth: 12,
    exposure: 1.0, fov: Math.tan(19 * Math.PI / 180),
    cam: { target: [550, 300, 250], baseYaw: Math.PI, yaw: Math.PI, pitch: 0.0, dist: 780,
           yawRange: 0.30, pitchRange: [-0.28, 0.28], distRange: [700, 900] },
  };
}

// ---------------- Outdoor field ----------------

function buildField() {
  const spheres = [
    sph([0, -1000, 0], 1000, 0, 1, 0, 0, [0.20, 0.30, 0.10]),
    sph([-4.5, 1, 0.5], 1, 1, 0, 0.05, 0, [0.8, 0.6, 0.2]),
    sph([0, 1, 0], 1, 2, 0, 0, 1.5, [1, 1, 1]),
    sph([0, 1, 0], -0.9, 2, 0, 0, 1.5, [1, 1, 1]),
    sph([4.5, 1, -0.5], 1, 0, 0, 0, 0, [0.12, 0.25, 0.75]),
    sph([-2.2, 1, -2.8], 1, 0, 2, 0, 0, [1, 1, 1]),
    sph([-2.4, 0.4, 2.4], 0.4, 0, 0, 0, 0, [0.9, 0.15, 0.15]),
    sph([0.6, 0.2, 3.2], 0.2, 0, 0, 0, 0, [0.9, 0.5, 0.1]),
    sph([1.6, 0.3, -2.0], 0.3, 2, 0, 0, 1.5, [1, 1, 1]),
  ];

  for (let ix = -10; ix < 10; ix++) {
    for (let iz = -14; iz < -9; iz++) {
      const choose = Math.random();
      const c = [ix + 0.9 * Math.random(), 0.2, iz + 0.9 * Math.random()];
      if (choose < 0.7) {
        const alb = [Math.random() ** 2, Math.random() ** 2, Math.random() ** 2];
        spheres.push(sph(c, 0.2, 0, 0, 0, 0, alb));
      } else if (choose < 0.85) {
        const alb = [0.5 + 0.5 * Math.random(), 0.5 + 0.5 * Math.random(), 0.5 + 0.5 * Math.random()];
        spheres.push(sph(c, 0.2, 1, 0, 0.05 + 0.25 * Math.random(), 0, alb));
      } else {
        spheres.push(sph(c, 0.2, 2, 0, 0, 1.5, [1, 1, 1]));
      }
    }
  }

  const sunC = SUN_DIR.map((d) => d * 8000);
  const sunIdx = spheres.length;
  spheres.push(sph(sunC, 600, 3, 0, 0, 0, [150, 132, 108]));

  const quads = [];
  addBox(quads, [2.6, 0, -2.4], [3.8, 1.2, -1.2], 0, [0, 0, 0], 1, 0, [0.85, 0.85, 0.9], 0.05);

  return {
    name: 'Field', type: 'prim',
    quads: new Float32Array(quads.flat()),
    spheres: new Float32Array(spheres.flat()),
    lightQuadIdx: -1, lightSphereIdx: sunIdx, lightQuadArea: 0,
    skyMode: 1, sunDir: SUN_DIR, numTris: 0,
    resScale: 1.0, maxDepth: 12,
    exposure: 0.9, fov: Math.tan(15 * Math.PI / 180),
    cam: { target: [0, 1, 0], baseYaw: 0, yaw: 0, pitch: 0.11, dist: 14.2,
           yawRange: 0.55, pitchRange: [-0.05, 0.40], distRange: [10, 20] },
  };
}

// ---------------- mesh scenes ----------------

function groundAndSun() {
  const spheres = [sph([0, -1000, 0], 1000, 0, 1, 0, 0, [0.20, 0.30, 0.10])];
  const sunIdx = spheres.length;
  const sunC = SUN_DIR.map((d) => d * 8000);
  spheres.push(sph(sunC, 600, 3, 0, 0, 0, [150, 132, 108]));
  return { spheres, sunIdx };
}

function cameraFromBBox(bbox, fovDeg, opts) {
  const lo = bbox.lo, hi = bbox.hi;
  const c = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
  const diag = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
  const d = opts.distScale * diag;
  return {
    target: c, baseYaw: Math.PI, yaw: Math.PI, pitch: opts.pitch ?? 0.18, dist: d,
    yawRange: opts.yawRange ?? 1.2, pitchRange: [-0.25, 0.55],
    distRange: [diag * 0.15, diag * 2.5],
  };
}

const BUNNY_MATERIALS = {
  chrome: { mtype: 1, albedo: [0.9, 0.9, 0.92], rough: 0.06, ior: 1.5 },
  gold: { mtype: 1, albedo: [0.85, 0.65, 0.25], rough: 0.15, ior: 1.5 },
  glass: { mtype: 2, albedo: [1, 1, 1], rough: 0, ior: 1.5 },
  matte: { mtype: 0, albedo: [0.75, 0.73, 0.7], rough: 1, ior: 1.5 },
};

async function buildBunny(matName = 'chrome') {
  const mesh = parseOBJ(new TextDecoder().decode(await fetchBytes('models/bunny.obj')));

  const P = mesh.positions;
  let minX = 1e30, minY = 1e30, minZ = 1e30, maxX = -1e30, maxY = -1e30, maxZ = -1e30;
  for (let i = 0; i < P.length; i += 3) {
    minX = Math.min(minX, P[i]); maxX = Math.max(maxX, P[i]);
    minY = Math.min(minY, P[i + 1]); maxY = Math.max(maxY, P[i + 1]);
    minZ = Math.min(minZ, P[i + 2]); maxZ = Math.max(maxZ, P[i + 2]);
  }
  // uniform scale so the bunny is 2.3 units tall and stands on y=0, centered in xz
  const scale = 2.3 / (maxY - minY);
  const cx = (minX + maxX) / 2, cz = (minZ + maxZ) / 2;
  for (let i = 0; i < P.length; i += 3) {
    P[i] = (P[i] - cx) * scale;
    P[i + 1] = (P[i + 1] - minY) * scale;
    P[i + 2] = (P[i + 2] - cz) * scale;
  }

  const bvh = buildBVHIndexed(P, mesh.tris, null);
  const meshScene = packMeshScene(
    { positions: P, normals: bvh.vnorm, idx: bvh.idx, nodes: bvh.nodes, numTris: bvh.numTris },
    [BUNNY_MATERIALS[matName]]);

  const { spheres, sunIdx } = groundAndSun();
  return {
    name: 'Bunny', type: 'mesh',
    mesh: meshScene, spheres: new Float32Array(spheres.flat()),
    lightQuadIdx: -1, lightSphereIdx: sunIdx, lightQuadArea: 0,
    skyMode: 1, sunDir: SUN_DIR, numTris: bvh.numTris,
    resScale: 1.0, maxDepth: 10,
    materialNames: Object.keys(BUNNY_MATERIALS), currentMaterial: matName,
    exposure: 0.9, fov: Math.tan(19 * Math.PI / 180),
    cam: { target: [0, 0.95, 0], baseYaw: Math.PI, yaw: Math.PI, pitch: 0.10, dist: 4.6,
           yawRange: 0.55, pitchRange: [-0.05, 0.40], distRange: [3, 9] },
  };
}

async function buildSponza() {
  const mesh = parseOBJ(new TextDecoder().decode(await fetchBytes('models/sponza.obj')));

  // normalize: longest horizontal dimension to 34 units, min y at 0
  let minX = 1e30, minY = 1e30, minZ = 1e30, maxX = -1e30, maxY = -1e30, maxZ = -1e30;
  const P = mesh.positions;
  for (let i = 0; i < P.length; i += 3) {
    minX = Math.min(minX, P[i]); maxX = Math.max(maxX, P[i]);
    minY = Math.min(minY, P[i + 1]); maxY = Math.max(maxY, P[i + 1]);
    minZ = Math.min(minZ, P[i + 2]); maxZ = Math.max(maxZ, P[i + 2]);
  }
  const scale = 34 / Math.max(maxX - minX, maxZ - minZ, 1);
  const cx = (minX + maxX) / 2, cz = (minZ + maxZ) / 2;
  for (let i = 0; i < P.length; i += 3) {
    P[i] = (P[i] - cx) * scale;
    P[i + 1] = (P[i + 1] - minY) * scale;
    P[i + 2] = (P[i + 2] - cz) * scale;
  }

  const bvh = buildBVHIndexed(P, mesh.tris, null);
  const meshScene = packMeshScene(
    { positions: P, normals: bvh.vnorm, idx: bvh.idx, nodes: bvh.nodes, numTris: bvh.numTris },
    [{ mtype: 0, albedo: [0.72, 0.68, 0.60], rough: 1, ior: 1.5 }]);

  const { spheres, sunIdx } = groundAndSun();
  const bbox = { lo: [(minX - cx) * scale, 0, (minZ - cz) * scale], hi: [(maxX - cx) * scale, (maxY - minY) * scale, (maxZ - cz) * scale] };
  const cam = { ...cameraFromBBox(bbox, 22, { distScale: 0.14, pitch: 0.06, yawRange: 1.3 }),
                distRange: [1.5, 60] };
  cam.target[1] = bbox.lo[1] + (bbox.hi[1] - bbox.lo[1]) * 0.45; // courtyard eye height
  return {
    name: 'Sponza', type: 'mesh',
    mesh: meshScene, spheres: new Float32Array(spheres.flat()),
    lightQuadIdx: -1, lightSphereIdx: sunIdx, lightQuadArea: 0,
    skyMode: 1, sunDir: SUN_DIR, numTris: bvh.numTris,
    resScale: 0.66, maxDepth: 8,
    exposure: 0.85, fov: Math.tan(22 * Math.PI / 180),
    cam,
  };
}

async function buildBistro() {
  const json = JSON.parse(new TextDecoder().decode(await fetchBytes('models/bistro.gltf')));
  const bin = (await fetchBytes('models/bistro.bin')).buffer;

  const mesh = parseGLTF(json, bin, { tint: true });

  // diffuse texture atlas (built by tools/build_atlas.py); graceful fallback to flat colors
  let atlas = { bitmaps: [], matUV: new Float32Array(mesh.mats.length * 8) };
  try {
    const texJson = JSON.parse(new TextDecoder().decode(await fetchBytes('models/bistro_tex.json')));
    const bitmaps = [];
    const nrmBitmaps = [];
    for (let i = 0; i < texJson.pages; i++) {
      const bytes = await fetchBytes('models/atlas_' + i + '.webp');
      bitmaps.push(await createImageBitmap(new Blob([bytes], { type: 'image/jpeg' }), { premultiplyAlpha: 'none', colorSpaceConversion: 'none' }));
    }
    for (let i = 0; i < (texJson.nrmPages || 0); i++) {
      const bytes = await fetchBytes('models/nrm_' + i + '.webp');
      nrmBitmaps.push(await createImageBitmap(new Blob([bytes], { type: 'image/jpeg' }), { premultiplyAlpha: 'none', colorSpaceConversion: 'none' }));
    }
    const M = mesh.mats.length;
    const matUV = new Float32Array(M * 12);
    for (let mi = 0; mi < M; mi++) {
      const e = texJson.slots[String(mi)];
      if (e) matUV.set(e.slice(0, 12), mi * 12);
    }
    atlas = { bitmaps, nrmBitmaps, matUV };
  } catch (e) {
    console.warn('atlas unavailable, falling back to flat colors:', String(e).slice(0, 100));
  }

  const hud = document.getElementById('hud');
  hud.textContent = atlas.bitmaps.length
    ? '模型下载完成 — 构建 BVH(421 万三角形 + 贴图图集,页面会冻结十几秒)…'
    : '模型下载完成 — 构建 BVH(421 万三角形,页面会冻结十几秒)…';
  await new Promise((r) => setTimeout(r, 60));
  const bvh = buildBVHIndexed(mesh.positions, mesh.tris, mesh.normals);
  const meshScene = packMeshScene(
    { positions: mesh.positions, normals: bvh.vnorm, idx: bvh.idx, nodes: bvh.nodes,
      numTris: bvh.numTris, matSlot: mesh.matSlot },
    mesh.mats.map((m) => ({ mtype: m.mtype, albedo: m.albedo, rough: m.rough, ior: m.ior })));

  const { spheres, sunIdx } = groundAndSun();
  return {
    name: 'Bistro', type: 'mesh',
    mesh: meshScene, spheres: new Float32Array(spheres.flat()),
    atlas,
    lightQuadIdx: -1, lightSphereIdx: sunIdx, lightQuadArea: 0,
    skyMode: 1, sunDir: SUN_DIR, numTris: bvh.numTris,
    resScale: 0.5, maxDepth: 8,
    exposure: 0.9, fov: Math.tan(24 * Math.PI / 180),
    cam: { ...cameraFromBBox(mesh.bbox, 24, { distScale: 0.7, pitch: 0.25, yawRange: 1.4 }),
           distRange: [2, 200] },
  };
}

export const SCENE_BUILDERS = { cornell: buildCornell, field: buildField, bunny: buildBunny, sponza: buildSponza, bistro: buildBistro };
export { BUNNY_MATERIALS };
