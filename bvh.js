// Mesh processing: OBJ/glTF parsing, BVH construction (scales to millions of
// triangles), smooth normals. Output layout is consumed by shaders.js mesh part:
//   nodes : 12 floats/node  (bmin.xyz, pad, bmax.xyz, pad, mi: left|first, right, count, 0)
//   verts : 3 floats/vertex (+ caller may pack a material slot into verts[i*3+3] later
//           — the GPU reads verts as vec4f, so the caller uploads a padded copy)
//   idx   : 3 uint32 per triangle, reordered so each BVH leaf references a contiguous run.

// ---------- dynamic typed arrays (avoid multi-GB JS number arrays) ----------

class F32 {
  constructor(cap = 4096) { this.a = new Float32Array(cap); this.n = 0; }
  push3(x, y, z) {
    if (this.n + 3 > this.a.length) this.grow(this.n + 3);
    this.a[this.n] = x; this.a[this.n + 1] = y; this.a[this.n + 2] = z;
    this.n += 3;
  }
  grow(need) {
    let cap = this.a.length * 2 || 4096;
    while (cap < need) cap *= 2;
    const b = new Float32Array(cap);
    b.set(this.a);
    this.a = b;
  }
  view() { return this.a.subarray(0, this.n); }
}

class U32 {
  constructor(cap = 4096) { this.a = new Uint32Array(cap); this.n = 0; }
  push3(x, y, z) {
    if (this.n + 3 > this.a.length) this.grow(this.n + 3);
    this.a[this.n] = x; this.a[this.n + 1] = y; this.a[this.n + 2] = z;
    this.n += 3;
  }
  grow(need) {
    let cap = this.a.length * 2 || 4096;
    while (cap < need) cap *= 2;
    const b = new Uint32Array(cap);
    b.set(this.a);
    this.a = b;
  }
  view() { return this.a.subarray(0, this.n); }
}

// ---------- OBJ ----------

// Parses OBJ text into shared-vertex indexed triangles. 1-based and negative
// indices supported; polygons fan-triangulated. vn/vt tokens are ignored
// (smooth normals are recomputed from geometry for consistent shading).
export function parseOBJ(text) {
  const pos = new F32(1 << 16);
  const tris = new U32(1 << 16);
  let posLines = 0;

  let lineStart = 0;
  const len = text.length;
  while (lineStart < len) {
    let lineEnd = text.indexOf('\n', lineStart);
    if (lineEnd < 0) lineEnd = len;
    let i = lineStart;
    while (i < lineEnd && text.charCodeAt(i) === 32) i++; // skip spaces
    const c0 = text.charCodeAt(i);
    if (c0 === 118 && (lineEnd - i) >= 2 && text.charCodeAt(i + 1) === 32) { // "v "
      let a = i + 2, b = text.indexOf(' ', a), c;
      const x = +text.slice(a, b);
      a = b + 1; b = text.indexOf(' ', a);
      const y = +text.slice(a, b);
      a = b + 1; b = text.indexOf(' ', a);
      if (b < 0 || b > lineEnd) b = lineEnd;
      const z = +text.slice(a, b);
      pos.push3(x, y, z);
      posLines++;
    } else if (c0 === 102) { // "f"
      // collect vertex indices of the face
      let vi = [0, 0, 0, 0, 0, 0, 0, 0, 0]; let nv = 0;
      let p = i + 1;
      while (p < lineEnd) {
        while (p < lineEnd && (text.charCodeAt(p) === 32 || text.charCodeAt(p) === 13)) p++;
        if (p >= lineEnd) break;
        let q = p;
        while (q < lineEnd && text.charCodeAt(q) !== 32) q++;
        // token [p,q): v[/vt][/vn]
        let slash = -1;
        for (let k = p; k < q; k++) { if (text.charCodeAt(k) === 47) { slash = k; break; } }
        const end = slash < 0 ? q : slash;
        let v = parseInt(text.slice(p, end), 10);
        if (v < 0) v = posLines + v + 1; else v = v - 1;
        if (nv >= vi.length) { const b2 = new Array(vi.length * 2); vi.forEach((x, k2) => b2[k2] = x); vi = b2; }
        vi[nv++] = v;
        p = q;
      }
      for (let k = 1; k + 1 < nv; k++) {
        tris.push3(vi[0], vi[k], vi[k + 1]);
      }
    }
    lineStart = lineEnd + 1;
  }

  return { positions: pos.view(), tris: tris.view(), numVerts: pos.n / 3 };
}

// ---------- glTF (geometry only: positions, normals, indices, materials) ----------

// Parses glTF JSON + binary buffer into an indexed mesh with per-primitive
// materials expanded through the node hierarchy (instances are flattened).
// Returns { positions, normals, tris, matOfTri: Uint32Array (per triangle),
//           mats: [{albedo, mtype, rough, ior}], bbox }.
export function parseGLTF(json, bin, opts = {}) {
  const accs = json.accessors;
  const views = json.bufferViews;
  const bufs = json.buffers.map((b, i) => (i === 0 ? bin : null)); // secondary buffers unsupported
  const CT_FLOAT = 5126, CT_USHORT = 5123, CT_UINT = 5125;

  function readAccessor(ai) {
    const a = accs[ai];
    const view = views[a.bufferView];
    const base = (view.byteOffset || 0) + (a.byteOffset || 0);
    const ncomp = a.type === 'VEC3' ? 3 : a.type === 'VEC2' ? 2 : a.type === 'SCALAR' ? 1 : 4;
    const compSize = a.componentType === CT_FLOAT ? 4 : (a.componentType === CT_USHORT ? 2 : 4);
    const stride = view.byteStride || ncomp * compSize;
    const buf = bufs[view.buffer];
    const out = new Float32Array(a.count * ncomp);
    const dv = new DataView(buf);
    for (let i = 0; i < a.count; i++) {
      const s = base + i * stride;
      for (let k = 0; k < ncomp; k++) {
        out[i * ncomp + k] = a.componentType === CT_FLOAT ? dv.getFloat32(s + k * compSize, true)
          : a.componentType === CT_USHORT ? dv.getUint16(s + k * compSize, true)
          : dv.getUint32(s + k * compSize, true);
      }
    }
    return out;
  }

  function readIndices(ai) {
    const a = accs[ai];
    const view = views[a.bufferView];
    const base = (view.byteOffset || 0) + (a.byteOffset || 0);
    const buf = bufs[view.buffer];
    const out = new Uint32Array(a.count);
    const dv = new DataView(buf);
    for (let i = 0; i < a.count; i++) {
      out[i] = a.componentType === CT_USHORT ? dv.getUint16(base + i * 2, true)
        : dv.getUint32(base + i * 4, true);
    }
    return out;
  }

  function mat4mul(a, b) { // column-major 4x4
    const o = new Float32Array(16);
    for (let c = 0; c < 4; c++)
      for (let r = 0; r < 4; r++)
        o[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
    return o;
  }

  function trsMatrix(n) {
    if (n.matrix) return new Float32Array(n.matrix);
    const m = new Float32Array(16);
    const s = n.scale || [1, 1, 1];
    if (n.rotation) {
      const [x, y, z, w] = n.rotation;
      const x2 = x + x, y2 = y + y, z2 = z + z;
      const xx = x * x2, xy = x * y2, xz = x * z2, yy = y * y2, yz = y * z2, zz = z * z2;
      const wx = w * x2, wy = w * y2, wz = w * z2;
      // column-major T * R * S: scale folded into rotation columns
      m[0] = (1 - (yy + zz)) * s[0]; m[1] = (xy + wz) * s[0]; m[2] = (xz - wy) * s[0];
      m[4] = (xy - wz) * s[1]; m[5] = (1 - (xx + zz)) * s[1]; m[6] = (yz + wx) * s[1];
      m[8] = (xz + wy) * s[2]; m[9] = (yz - wx) * s[2]; m[10] = (1 - (xx + yy)) * s[2];
    } else {
      m[0] = s[0]; m[5] = s[1]; m[10] = s[2];
    }
    m[15] = 1;
    if (n.translation) { m[12] = n.translation[0]; m[13] = n.translation[1]; m[14] = n.translation[2]; }
    return m;
  }

  // world matrices per node (walk children down from roots)
  const worldOf = new Array(json.nodes.length);
  const children = {};
  for (let i = 0; i < json.nodes.length; i++) children[i] = [];
  const roots = [];
  const isChild = new Uint8Array(json.nodes.length);
  json.nodes.forEach((n, i) => {
    if (n.children) n.children.forEach((c) => { children[i].push(c); isChild[c] = 1; });
  });
  for (let i = 0; i < json.nodes.length; i++) if (!isChild[i]) roots.push(i);
  const stack = roots.map((r) => [r, trsMatrix(json.nodes[r])]);
  while (stack.length) {
    const [i, m] = stack.pop();
    if (worldOf[i]) continue;
    worldOf[i] = m;
    for (const c of children[i]) stack.push([c, mat4mul(m, trsMatrix(json.nodes[c]))]);
  }

  // materials
  const mats = [];
  for (const m of json.materials || []) {
    const ext = m.extensions && m.extensions.KHR_materials_pbrSpecularGlossiness;
    let diffuse = ext && ext.diffuseFactor ? ext.diffuseFactor.slice(0, 3) : [1, 1, 1];
    const spec = ext && ext.specularFactor ? ext.specularFactor.slice(0, 3) : [0.04, 0.04, 0.04];
    const gloss = ext && ext.glossinessFactor !== undefined ? ext.glossinessFactor : 1;
    const dLum = 0.2126 * diffuse[0] + 0.7152 * diffuse[1] + 0.0722 * diffuse[2];
    const sLum = 0.2126 * spec[0] + 0.7152 * spec[1] + 0.0722 * spec[2];
    let mtype = 0, rough = 1, albedo = diffuse;
    if (m.alphaMode === 'BLEND') {
      mtype = 2; rough = 0; albedo = [0.9, 0.95, 1.0];          // treat blend = glass
    } else if (dLum < 0.08 && sLum > 0.4) {
      mtype = 1; rough = Math.min(1, Math.max(0.04, 1 - gloss)); // metallic paint / windows
      albedo = spec;
    }
    if (opts.tint) {
      const h = Math.abs(Math.sin(mats.length * 12.9898) * 43758.5453) % 1;
      const k = 0.62 + 0.38 * h;
      albedo = albedo.map((v) => Math.min(1, v * k));
    }
    mats.push({ albedo, mtype, rough, ior: 1.5 });
  }

  // expand instances
  const verts = new F32(1 << 22);
  const norms = new F32(1 << 22);
  const tris = new U32(1 << 23);
  const matSlot = new U32(1 << 22); // per-vertex material index (exact: each glTF primitive owns its vertices)
  let numVerts = 0;
  const lo = [1e30, 1e30, 1e30], hi = [-1e30, -1e30, -1e30];

  for (let ni = 0; ni < json.nodes.length; ni++) {
    const n = json.nodes[ni];
    if (n.mesh === undefined) continue;
    const m = worldOf[ni];
    const mesh = json.meshes[n.mesh];
    for (const prim of mesh.primitives) {
      const P = readAccessor(prim.attributes.POSITION);
      const N = prim.attributes.NORMAL !== undefined ? readAccessor(prim.attributes.NORMAL) : null;
      const I = prim.indices !== undefined ? readIndices(prim.indices) : null;
      const base = numVerts;
      const vcount = P.length / 3;
      for (let i = 0; i < vcount; i++) {
        const x = P[i * 3], y = P[i * 3 + 1], z = P[i * 3 + 2];
        const wx = m[0] * x + m[4] * y + m[8] * z + m[12];
        const wy = m[1] * x + m[5] * y + m[9] * z + m[13];
        const wz = m[2] * x + m[6] * y + m[10] * z + m[14];
        verts.push3(wx, wy, wz);
        if (N) {
          const nx = N[i * 3], ny = N[i * 3 + 1], nz = N[i * 3 + 2];
          let tx = m[0] * nx + m[4] * ny + m[8] * nz;
          let ty = m[1] * nx + m[5] * ny + m[9] * nz;
          let tz = m[2] * nx + m[6] * ny + m[10] * nz;
          const l = Math.hypot(tx, ty, tz) || 1;
          norms.push3(tx / l, ty / l, tz / l);
        } else {
          norms.push3(0, 1, 0);
        }
        if (wx < lo[0]) lo[0] = wx; if (wx > hi[0]) hi[0] = wx;
        if (wy < lo[1]) lo[1] = wy; if (wy > hi[1]) hi[1] = wy;
        if (wz < lo[2]) lo[2] = wz; if (wz > hi[2]) hi[2] = wz;
      }
      numVerts += vcount;
      const count = I ? I.length : vcount;
      const matId = prim.material !== undefined ? prim.material : 0;
      for (let i = base; i < numVerts; i++) matSlot.a[i] = matId;
      for (let i = 0; i < count; i += 3) {
        tris.push3(base + I[i], base + I[i + 1], base + I[i + 2]);
      }
    }
  }

  // per-vertex material slot (only valid if matSlot.n == numVerts)
  const matSlotOut = matSlot.n === numVerts ? matSlot.a.slice(0, numVerts) : null;

  return {
    positions: verts.view(),
    normals: norms.view(),
    tris: tris.view(),
    matSlot: matSlotOut,
    mats,
    numVerts,
    bbox: { lo, hi },
  };
}

// ---------- BVH (in-place quickselect median split, indexed output) ----------

// positions: Float32Array xyz per vertex; indices: Uint32Array, 3 per triangle;
// normals: Float32Array xyz per vertex or null (smooth normals are computed).
// Returns { nodes: Float32Array (12 floats/node), idx: Uint32Array (ordered),
//           vnorm: Float32Array (per-vertex normals), numTris }.
export function buildBVHIndexed(positions, indices, normals) {
  const numTris = indices.length / 3;
  const numVerts = positions.length / 3;

  // smooth vertex normals when none provided
  let vnorm = normals;
  if (!vnorm) {
    vnorm = new Float32Array(numVerts * 3);
    for (let t = 0; t < numTris; t++) {
      const i0 = indices[t * 3] * 3, i1 = indices[t * 3 + 1] * 3, i2 = indices[t * 3 + 2] * 3;
      const ax = positions[i1] - positions[i0], ay = positions[i1 + 1] - positions[i0 + 1], az = positions[i1 + 2] - positions[i0 + 2];
      const bx = positions[i2] - positions[i0], by = positions[i2 + 1] - positions[i0 + 1], bz = positions[i2 + 2] - positions[i0 + 2];
      const nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx;
      vnorm[i0] += nx; vnorm[i0 + 1] += ny; vnorm[i0 + 2] += nz;
      vnorm[i1] += nx; vnorm[i1 + 1] += ny; vnorm[i1 + 2] += nz;
      vnorm[i2] += nx; vnorm[i2 + 1] += ny; vnorm[i2 + 2] += nz;
    }
    for (let i = 0; i < numVerts; i++) {
      const l = Math.hypot(vnorm[i * 3], vnorm[i * 3 + 1], vnorm[i * 3 + 2]);
      if (l > 0) { vnorm[i * 3] /= l; vnorm[i * 3 + 1] /= l; vnorm[i * 3 + 2] /= l; }
      else { vnorm[i * 3 + 1] = 1; }
    }
  }

  // per-triangle bounds and centroids
  const cMin = new Float32Array(numTris * 3);
  const cMax = new Float32Array(numTris * 3);
  const centroid = new Float32Array(numTris * 3);
  for (let t = 0; t < numTris; t++) {
    let mx = 1e30, my = 1e30, mz = 1e30, Mx = -1e30, My = -1e30, Mz = -1e30;
    for (let k = 0; k < 3; k++) {
      const i = indices[t * 3 + k] * 3;
      const x = positions[i], y = positions[i + 1], z = positions[i + 2];
      if (x < mx) mx = x; if (x > Mx) Mx = x;
      if (y < my) my = y; if (y > My) My = y;
      if (z < mz) mz = z; if (z > Mz) Mz = z;
    }
    cMin[t * 3] = mx; cMin[t * 3 + 1] = my; cMin[t * 3 + 2] = mz;
    cMax[t * 3] = Mx; cMax[t * 3 + 1] = My; cMax[t * 3 + 2] = Mz;
    centroid[t * 3] = (mx + Mx) * 0.5;
    centroid[t * 3 + 1] = (my + My) * 0.5;
    centroid[t * 3 + 2] = (mz + Mz) * 0.5;
  }

  const order = new Uint32Array(numTris);
  for (let i = 0; i < numTris; i++) order[i] = i;

  const nodes = []; // flat objects during build
  const LEAF = 8;

  // move the median (by centroid[axis]) of order[first..first+count) into place
  function quickselect(first, count, axis, mid) {
    let lo = first, hi = first + count - 1;
    const target = first + mid;
    while (lo < hi) {
      const pivot = centroid[order[hi] * 3 + axis];
      let i = lo;
      for (let j = lo; j < hi; j++) {
        if (centroid[order[j] * 3 + axis] < pivot) {
          const t = order[i]; order[i] = order[j]; order[j] = t;
          i++;
        }
      }
      const t = order[i]; order[i] = order[hi]; order[hi] = t;
      if (i === target) return;
      else if (i < target) lo = i + 1;
      else hi = i - 1;
    }
  }

  function build(first, count) {
    let mx = 1e30, my = 1e30, mz = 1e30, Mx = -1e30, My = -1e30, Mz = -1e30;
    for (let i = 0; i < count; i++) {
      const t = order[first + i];
      if (cMin[t * 3] < mx) mx = cMin[t * 3]; if (cMax[t * 3] > Mx) Mx = cMax[t * 3];
      if (cMin[t * 3 + 1] < my) my = cMin[t * 3 + 1]; if (cMax[t * 3 + 1] > My) My = cMax[t * 3 + 1];
      if (cMin[t * 3 + 2] < mz) mz = cMin[t * 3 + 2]; if (cMax[t * 3 + 2] > Mz) Mz = cMax[t * 3 + 2];
    }
    const id = nodes.push({ mx, my, mz, Mx, My, Mz, a: 0, b: 0, count: 0 }) - 1;
    if (count <= LEAF) {
      nodes[id].a = first;
      nodes[id].count = count;
      return id;
    }
    // longest centroid axis
    let cmx = 1e30, cmy = 1e30, cmz = 1e30, CMx = -1e30, CMy = -1e30, CMz = -1e30;
    for (let i = 0; i < count; i++) {
      const t = order[first + i] * 3;
      if (centroid[t] < cmx) cmx = centroid[t]; if (centroid[t] > CMx) CMx = centroid[t];
      if (centroid[t + 1] < cmy) cmy = centroid[t + 1]; if (centroid[t + 1] > CMy) CMy = centroid[t + 1];
      if (centroid[t + 2] < cmz) cmz = centroid[t + 2]; if (centroid[t + 2] > CMz) CMz = centroid[t + 2];
    }
    const ex = CMx - cmx, ey = CMy - cmy, ez = CMz - cmz;
    const axis = ex > ey ? (ex > ez ? 0 : 2) : (ey > ez ? 1 : 2);
    quickselect(first, count, axis, count >> 1);
    const mid = count >> 1;
    const L = build(first, mid);
    const R = build(first + mid, count - mid);
    nodes[id].a = L;
    nodes[id].b = R;
    return id;
  }

  build(0, numTris);

  // flatten (children contiguous) + materialize ordered index buffer
  const nodesFlat = new Float32Array(nodes.length * 12);
  const idx = new Uint32Array(numTris * 3);
  let nodeCount = 0, triCount = 0;
  function flatten(id) {
    const n = nodes[id];
    const my = nodeCount++;
    const o = my * 12;
    nodesFlat[o] = n.mx; nodesFlat[o + 1] = n.my; nodesFlat[o + 2] = n.mz;
    nodesFlat[o + 4] = n.Mx; nodesFlat[o + 5] = n.My; nodesFlat[o + 6] = n.Mz;
    if (n.count > 0) {
      nodesFlat[o + 8] = triCount;
      nodesFlat[o + 10] = n.count;
      for (let i = 0; i < n.count; i++) {
        const t = order[n.a + i];
        idx[triCount * 3] = indices[t * 3];
        idx[triCount * 3 + 1] = indices[t * 3 + 1];
        idx[triCount * 3 + 2] = indices[t * 3 + 2];
        triCount++;
      }
      return my;
    }
    const L = flatten(n.a);
    const R = flatten(n.b);
    nodesFlat[o + 8] = L;
    nodesFlat[o + 9] = R;
    nodesFlat[o + 10] = 0;
    return my;
  }
  flatten(0);

  return { nodes: nodesFlat, idx, vnorm, numTris };
}
