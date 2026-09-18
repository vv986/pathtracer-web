// WGSL shaders for the multi-scene GPU path tracer.
// Structure: CORE1 (shared helpers, bindings 0-4) -> scene-specific block
// (mesh: BVH bindings 5-7 + mesh_hit; prim: dummy mesh_hit) -> CORE2 (scene_hit + trace + entry).

export const CORE1 = /* wgsl */ `
struct Uniforms {
  dims      : vec4u,   // W, H, frame, spp this frame
  cam_pos_f : vec4f,   // camera position xyz, fov_scale
  cam_tgt   : vec4f,   // target xyz, exposure
  counts    : vec4f,   // numQuads, numSpheres, lightQuadIdx, lightSphereIdx
  light_inf : vec4f,   // lightQuadArea, skyMode, numTris, unused
  sun_dir   : vec4f,   // xyz, mesh material type
  mesh_mat  : vec4f,   // mesh albedo rgb, roughness
  mesh_x    : vec4f,   // ior, unused
};

struct Quad {
  q  : vec4f,   // origin xyz, material type
  uu : vec4f,   // edge u xyz, texture id
  vv : vec4f,   // edge v xyz, unused
  nd : vec4f,   // outward normal xyz, plane distance D
  alb: vec4f,   // albedo rgb, roughness
};

struct Sphere {
  pr : vec4f,   // center xyz, radius (may be negative for hollow shells)
  m  : vec4f,   // material type, texture id, roughness, ior
  alb: vec4f,   // albedo rgb, unused
};

struct Mat {
  mtype: i32,   // 0 diffuse, 1 metal (GGX), 2 glass, 3 emissive
  tex  : i32,   // 0 solid, 1 checker, 2 marble
  rough: f32,
  ior  : f32,
  alb  : vec3f,
};

struct Hit {
  t    : f32,
  p    : vec3f,
  n    : vec3f,
  front: bool,
  mat  : Mat,
};

@group(0) @binding(0) var<uniform> u : Uniforms;
@group(0) @binding(1) var<storage, read_write> accum : array<vec4f>;
@group(0) @binding(2) var out_tex : texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<storage, read> quads : array<Quad>;
@group(0) @binding(4) var<storage, read> spheres : array<Sphere>;
@group(0) @binding(9) var<storage, read_write> history : array<vec4f>;

const PI = 3.14159265358979;
const MAX_DEPTH = 12u;

// ---- RNG (PCG hash) ----

fn pcg(state: ptr<function, u32>) -> u32 {
  var s = *state;
  s = s * 747796405u + 2891336453u;
  let word = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  *state = s;
  return (word >> 22u) ^ word;
}

fn rand(state: ptr<function, u32>) -> f32 {
  return f32(pcg(state)) * (1.0 / 4294967296.0);
}

fn hash_seed(x: u32, y: u32, frame: u32) -> u32 {
  var s = x * 1973u + y * 9277u + frame * 26699u + 1u;
  pcg(&s);
  pcg(&s);
  return s;
}

// ---- sampling ----

fn cosine_dir(r: ptr<function, u32>) -> vec3f {
  let z = sqrt(1.0 - rand(r));
  let phi = 2.0 * PI * rand(r);
  let s = sqrt(max(0.0, 1.0 - z * z));
  return vec3f(s * cos(phi), s * sin(phi), z);
}

fn random_to_sphere(radius: f32, distance_squared: f32, r: ptr<function, u32>) -> vec3f {
  let r1 = rand(r);
  let r2 = rand(r);
  let z = 1.0 + r2 * (sqrt(max(0.0, 1.0 - radius * radius / distance_squared)) - 1.0);
  let phi = 2.0 * PI * r1;
  let x = cos(phi) * sqrt(max(0.0, 1.0 - z * z));
  let y = sin(phi) * sqrt(max(0.0, 1.0 - z * z));
  return vec3f(x, y, z);
}

fn onb(w_in: vec3f) -> mat3x3f {
  let w = normalize(w_in);
  let a = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), abs(w.x) > 0.9);
  let v = normalize(cross(w, a));
  let uu = cross(w, v);
  return mat3x3f(uu, v, w);
}

// ---- environment ----

fn sky(rd: vec3f) -> vec3f {
  let t = clamp(0.5 * (rd.y + 1.0), 0.0, 1.0);
  var c = mix(vec3f(0.72, 0.80, 0.90), vec3f(0.10, 0.24, 0.60), pow(t, 0.8));
  let s = max(0.0, dot(rd, normalize(u.sun_dir.xyz)));
  c = c + vec3f(1.0, 0.75, 0.5) * 0.12 * pow(s, 6.0);
  return c;
}

// ---- procedural textures ----

fn nhash(p: vec3f) -> f32 {
  let q = fract(p * 0.3183099 + vec3f(0.11, 0.27, 0.43)) * 50.0;
  return fract(q.x * q.y * q.z * (q.x + q.y + q.z));
}

fn vnoise(p: vec3f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  let a = mix(mix(nhash(i), nhash(i + vec3f(1.0, 0.0, 0.0)), u.x),
              mix(nhash(i + vec3f(0.0, 1.0, 0.0)), nhash(i + vec3f(1.0, 1.0, 0.0)), u.x), u.y);
  let b = mix(mix(nhash(i + vec3f(0.0, 0.0, 1.0)), nhash(i + vec3f(1.0, 0.0, 1.0)), u.x),
              mix(nhash(i + vec3f(0.0, 1.0, 1.0)), nhash(i + vec3f(1.0, 1.0, 1.0)), u.x), u.y);
  return mix(a, b, u.z);
}

fn turb(p: vec3f) -> f32 {
  var a = 0.0;
  var w = 1.0;
  var q = p;
  for (var i = 0; i < 5; i = i + 1) {
    a = a + w * vnoise(q);
    w = w * 0.5;
    q = q * 2.0;
  }
  return abs(a);
}

fn tex_color(tex: i32, p: vec3f, alb: vec3f) -> vec3f {
  if (tex == 1i) {
    // checkerboard
    let k = floor(p.x * 3.125) + floor(p.y * 3.125) + floor(p.z * 3.125);
    let odd = select(0.0, 1.0, abs(k % 2.0) > 0.5);
    return mix(alb, vec3f(0.90, 0.90, 0.90), odd);
  }
  if (tex == 2i) {
    // marble
    let m = 0.5 * (1.0 + sin(4.0 * p.z + 10.0 * turb(p)));
    return vec3f(1.0, 1.0, 1.0) * m;
  }
  return alb;
}

// ---- materials ----

fn schlick(cosine: f32, ior: f32) -> f32 {
  var r0 = (1.0 - ior) / (1.0 + ior);
  r0 = r0 * r0;
  return r0 + (1.0 - r0) * pow(1.0 - cosine, 5.0);
}

fn g1(mu: f32, a2: f32) -> f32 {
  return 2.0 * mu / (mu + sqrt(a2 + (1.0 - a2) * mu * mu));
}

fn ggx_sample(rd: vec3f, n: vec3f, alph: f32, r: ptr<function, u32>) -> vec3f {
  // VNDF sampling (Heitz 2018): sample the half-vector distribution *visible*
  // from the view direction, which avoids below-surface reflections at grazing angles.
  let o = -normalize(rd);
  let b = onb(n);
  let ve = vec3f(dot(o, b[0]), dot(o, b[1]), dot(o, b[2]));
  let vh = normalize(vec3f(alph * ve.x, alph * ve.y, ve.z));
  let t1 = select(vec3f(1.0, 0.0, 0.0), normalize(cross(vh, vec3f(0.0, 0.0, 1.0))), vh.z < 0.999);
  let t2 = cross(t1, vh);
  let rr = sqrt(rand(r));
  let phi = 2.0 * PI * rand(r);
  let s1 = rr * cos(phi);
  let s2 = rr * sin(phi);
  let s = 0.5 * (1.0 + vh.z);
  let s2b = (1.0 - s) * sqrt(max(0.0, 1.0 - s1 * s1)) + s * s2;
  let nh = normalize(s1 * t1 + s2b * t2 + sqrt(max(0.0, 1.0 - s1 * s1 - s2b * s2b)) * vh);
  let hl = normalize(vec3f(alph * nh.x, alph * nh.y, max(1e-6, nh.z)));
  return b * hl;   // world-space half vector
}

// ---- intersection primitives ----

fn hit_quad(qd: Quad, ro: vec3f, rd: vec3f, tmin: f32, tmax: f32, h: ptr<function, Hit>) -> bool {
  let n_un = cross(qd.uu.xyz, qd.vv.xyz);
  let nn = normalize(n_un);
  let denom = dot(nn, rd);
  if (abs(denom) < 1e-8) { return false; }
  let t = (qd.nd.w - dot(nn, ro)) / denom;
  if (t < tmin || t > tmax) { return false; }
  let p = ro + rd * t;
  let pv = p - qd.q.xyz;
  let w = n_un / dot(n_un, n_un);
  let alpha = dot(w, cross(pv, qd.vv.xyz));
  let beta = dot(w, cross(qd.uu.xyz, pv));
  if (alpha < 0.0 || alpha > 1.0 || beta < 0.0 || beta > 1.0) { return false; }
  let front = dot(rd, nn) < 0.0;
  let mat = Mat(i32(qd.q.w), i32(qd.uu.w), qd.alb.w, 0.0, qd.alb.xyz);
  (*h) = Hit(t, p, select(-nn, nn, front), front, mat);
  return true;
}

fn hit_sphere_s(s: Sphere, ro: vec3f, rd: vec3f, tmin: f32, tmax: f32, h: ptr<function, Hit>) -> bool {
  let rad = s.pr.w;
  let oc = ro - s.pr.xyz;
  let a = dot(rd, rd);
  let hb = dot(oc, rd);
  let c = dot(oc, oc) - rad * rad;
  let disc = hb * hb - a * c;
  if (disc < 0.0) { return false; }
  let sq = sqrt(disc);
  var t = (-hb - sq) / a;
  if (t < tmin || t > tmax) {
    t = (-hb + sq) / a;
    if (t < tmin || t > tmax) { return false; }
  }
  let p = ro + rd * t;
  let n_out = (p - s.pr.xyz) / rad;
  let front = dot(rd, n_out) < 0.0;
  let mat = Mat(i32(s.m.x), i32(s.m.y), s.m.z, s.m.w, s.alb.xyz);
  (*h) = Hit(t, p, select(-n_out, n_out, front), front, mat);
  return true;
}

fn scene_hit_prims(ro: vec3f, rd: vec3f, tmin: f32, tmax: f32, h: ptr<function, Hit>) -> bool {
  var best = tmax;
  var found = false;
  var tmp: Hit;

  let nq = u32(u.counts.x);
  for (var i = 0u; i < nq; i = i + 1u) {
    if (hit_quad(quads[i], ro, rd, tmin, best, &tmp)) {
      best = tmp.t;
      found = true;
      *h = tmp;
    }
  }

  let ns = u32(u.counts.y);
  for (var i = 0u; i < ns; i = i + 1u) {
    if (hit_sphere_s(spheres[i], ro, rd, tmin, best, &tmp)) {
      best = tmp.t;
      found = true;
      *h = tmp;
    }
  }
  return found;
}

// ---- light sampling (quad area light or emissive sphere) ----

fn quad_pdf_value(L: Quad, ro: vec3f, rd: vec3f, area: f32) -> f32 {
  let n_un = cross(L.uu.xyz, L.vv.xyz);
  let nn = normalize(n_un);
  let denom = dot(nn, rd);
  if (abs(denom) < 1e-8) { return 0.0; }
  let t = (L.nd.w - dot(nn, ro)) / denom;
  if (t < 0.001) { return 0.0; }
  let p = ro + rd * t;
  let pv = p - L.q.xyz;
  let w = n_un / dot(n_un, n_un);
  let alpha = dot(w, cross(pv, L.vv.xyz));
  let beta = dot(w, cross(L.uu.xyz, pv));
  if (alpha < 0.0 || alpha > 1.0 || beta < 0.0 || beta > 1.0) { return 0.0; }
  let d2 = t * t * dot(rd, rd);
  let cosv = abs(dot(rd, nn)) / length(rd);
  return d2 / (cosv * area);
}

fn sph_light_solid_angle(d2: f32, r2: f32) -> f32 {
  if (d2 <= r2) { return 0.0; }
  let cos_max = sqrt(1.0 - r2 / d2);
  return 2.0 * PI * (1.0 - cos_max);
}

fn sph_light_pdf_dir(dir_c: vec3f, dir: vec3f, rad: f32) -> f32 {
  let d2 = dot(dir_c, dir_c);
  let sa = sph_light_solid_angle(d2, rad * rad);
  if (sa <= 0.0) { return 0.0; }
  let c = dot(normalize(dir), normalize(dir_c));
  let cos_max = sqrt(max(0.0, 1.0 - rad * rad / d2));
  return select(0.0, 1.0 / sa, c >= cos_max);
}

fn sph_light_gen(dir_c: vec3f, rad: f32, r: ptr<function, u32>) -> vec3f {
  let d2 = dot(dir_c, dir_c);
  let b = onb(dir_c);
  return b * random_to_sphere(rad, d2, r);
}
`;

// Mesh scene: BVH data bindings + traversal. Prim scenes stub mesh_hit out.
export const MESH_PART = /* wgsl */ `
struct BVHNode {
  bmin : vec4f,
  bmax : vec4f,
  mi   : vec4f,   // packed as floats; leaf: x = first tri, z = count; internal: x = left, y = right
};

@group(0) @binding(5) var<storage, read> nodes : array<BVHNode>;
@group(0) @binding(6) var<storage, read> verts : array<vec4f>;   // xyz + material slot
@group(0) @binding(7) var<storage, read> vnorms : array<vec4f>;
@group(0) @binding(8) var<storage, read> mIdx : array<u32>;      // 3 per triangle
@group(0) @binding(10) var<storage, read> mat_alb : array<vec4f>; // rgb + mtype
@group(0) @binding(11) var<storage, read> mat_prm : array<vec4f>; // rough + ior

fn aabb_hit(bmin: vec3f, bmax: vec3f, ro: vec3f, inv: vec3f, tmax: f32) -> bool {
  let t0 = (bmin - ro) * inv;
  let t1 = (bmax - ro) * inv;
  let tmin_i = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
  let tmax_i = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
  return tmax_i >= max(tmin_i, 0.0) && tmin_i < tmax;
}

fn mesh_hit(ro: vec3f, rd: vec3f, tmin: f32, tmax: f32, h: ptr<function, Hit>) -> bool {
  var best = tmax;
  var found = false;
  var stack: array<u32, 48>;
  var sp = 0u;
  stack[0] = 0u;
  sp = 1u;
  let inv = 1.0 / rd;

  loop {
    if (sp == 0u) { break; }
    sp = sp - 1u;
    let nd = nodes[stack[sp]];
    if (!aabb_hit(nd.bmin.xyz, nd.bmax.xyz, ro, inv, best)) { continue; }

    let cnt = u32(nd.mi.z);
    if (cnt > 0u) {
      let first = u32(nd.mi.x);
      for (var i = 0u; i < cnt; i = i + 1u) {
        let ti = first + i;
        let i0 = mIdx[ti * 3u + 0u];
        let i1 = mIdx[ti * 3u + 1u];
        let i2 = mIdx[ti * 3u + 2u];
        let v0 = verts[i0].xyz;
        let v1 = verts[i1].xyz;
        let v2 = verts[i2].xyz;
        let e1 = v1 - v0;
        let e2 = v2 - v0;
        let pv = cross(rd, e2);
        let det = dot(e1, pv);
        if (abs(det) < 1e-9) { continue; }
        let invd = 1.0 / det;
        let tv = ro - v0;
        let uu = dot(tv, pv) * invd;
        if (uu < 0.0 || uu > 1.0) { continue; }
        let qv = cross(tv, e1);
        let wv = dot(rd, qv) * invd;
        if (wv < 0.0 || uu + wv > 1.0) { continue; }
        let t = dot(e2, qv) * invd;
        if (t < tmin || t > best) { continue; }

        let n_g = normalize(cross(e1, e2));
        let front = dot(rd, n_g) < 0.0;
        let n0 = vnorms[i0].xyz;
        let n1 = vnorms[i1].xyz;
        let n2 = vnorms[i2].xyz;
        var n_s = normalize((1.0 - uu - wv) * n0 + uu * n1 + wv * n2);
        if (dot(n_s, n_g) < 0.0) { n_s = -n_s; }

        let ms = u32(verts[i0].w);
        let mat = Mat(i32(mat_alb[ms].w), 0i, mat_prm[ms].x, mat_prm[ms].y, mat_alb[ms].xyz);
        (*h) = Hit(t, ro + rd * t, select(n_g, n_s, true), front, mat);
        found = true;
        best = t;
      }
    } else {
      // internal node: visit the nearer child first (tightens best-t faster)
      let l = u32(nd.mi.x);
      let r = u32(nd.mi.y);
      let hitL = aabb_hit(nodes[l].bmin.xyz, nodes[l].bmax.xyz, ro, inv, best);
      let hitR = aabb_hit(nodes[r].bmin.xyz, nodes[r].bmax.xyz, ro, inv, best);
      if (hitL && hitR) {
        let cl = (nodes[l].bmin.xyz + nodes[l].bmax.xyz) * 0.5;
        let cr = (nodes[r].bmin.xyz + nodes[r].bmax.xyz) * 0.5;
        // true = L is nearer; push far first so near is popped first
        let lNearer = dot(cl - ro, rd) < dot(cr - ro, rd);
        let nearC = select(r, l, lNearer);
        let farC = select(l, r, lNearer);
        stack[sp] = farC; sp = sp + 1u;
        stack[sp] = nearC; sp = sp + 1u;
      } else if (hitL) {
        stack[sp] = l; sp = sp + 1u;
      } else if (hitR) {
        stack[sp] = r; sp = sp + 1u;
      }
    }
  }
  return found;
}
`;

export const PRIM_STUB = /* wgsl */ `
fn mesh_hit(ro: vec3f, rd: vec3f, tmin: f32, tmax: f32, h: ptr<function, Hit>) -> bool {
  return false;
}
`;

export const CORE2 = /* wgsl */ `
fn scene_hit(ro: vec3f, rd: vec3f, tmin: f32, tmax: f32, h: ptr<function, Hit>) -> bool {
  var found = scene_hit_prims(ro, rd, tmin, tmax, h);
  if (u.light_inf.z > 0.5) {
    var tmp: Hit;
    let best = select(tmax, (*h).t, found);
    if (mesh_hit(ro, rd, tmin, best, &tmp)) {
      *h = tmp;
      found = true;
    }
  }
  return found;
}

fn light_gen(p: vec3f, r: ptr<function, u32>) -> vec3f {
  let lqi = i32(u.counts.z);
  let lsi = i32(u.counts.w);
  if (lqi >= 0) {
    let L = quads[lqi];
    return L.q.xyz + rand(r) * L.uu.xyz + rand(r) * L.vv.xyz - p;
  }
  let s = spheres[lsi];
  return sph_light_gen(s.pr.xyz - p, s.pr.w, r);
}

fn light_pdf(p: vec3f, dir: vec3f) -> f32 {
  let lqi = i32(u.counts.z);
  let lsi = i32(u.counts.w);
  if (lqi >= 0) {
    return quad_pdf_value(quads[lqi], p, dir, u.light_inf.x);
  }
  let s = spheres[lsi];
  return sph_light_pdf_dir(s.pr.xyz - p, dir, s.pr.w);
}

const DEBUG_COLORS = 0; // 0 off, 1 fail/NaN, 2 first-hit normals, 3 uniform visualization

fn trace(ro0: vec3f, rd0: vec3f, rng: ptr<function, u32>) -> vec3f {
  if (DEBUG_COLORS == 3i) {
    // uniform visualization: R = skyMode, G = maxDepth/16, B = numSpheres/400
    return vec3f(f32(u.light_inf.y), u.light_inf.w / 16.0, f32(u.counts.y) / 400.0);
  }
  var radiance = vec3f(0.0);
  var tp = vec3f(1.0);
  var ro = ro0;
  var rd = rd0;
  var failed = false;
  var nan = false;

  var bounce = 0u;
  loop {
    if (bounce >= max(u32(u.light_inf.w), 1u)) { break; }
    bounce = bounce + 1u;

    // Russian roulette: terminate low-throughput paths early (unbiased)
    if (bounce > 3u) {
      let p_survive = clamp(max(tp.r, max(tp.g, tp.b)), 0.05, 0.95);
      if (rand(rng) > p_survive) { break; }
      tp = tp / p_survive;
    }

    var h: Hit;
    if (!scene_hit(ro, rd, 0.001, 1e30, &h)) {
      if (u.light_inf.y > 0.5) {
        radiance = radiance + tp * sky(rd);
      }
      break;
    }

    if (DEBUG_COLORS == 2i && bounce == 1u) {
      return abs(h.n) * 2.0;
    }

    // emissive (front face only)
    if (h.mat.mtype == 3i) {
      if (h.front) {
        radiance = radiance + tp * h.mat.alb;
      }
      break;
    }

    // glass
    if (h.mat.mtype == 2i) {
      let ior = select(h.mat.ior, 1.0 / h.mat.ior, h.front);
      let d = normalize(rd);
      let ct = min(dot(-d, h.n), 1.0);
      let st = sqrt(1.0 - ct * ct);
      var dir: vec3f;
      var offs: vec3f;
      if (ior * st > 1.0 || schlick(ct, ior) > rand(rng)) {
        dir = reflect(d, h.n);
        offs = h.n * 0.002;        // reflection: stay on the incident side
      } else {
        dir = refract(d, h.n, ior);
        offs = -h.n * 0.002;       // refraction: cross to the far side
      }
      ro = h.p + offs;
      rd = dir;
      continue;
    }

    // metal (GGX with VNDF sampling — no light mixture, specular-like path)
    if (h.mat.mtype == 1i) {
      let a2 = max(h.mat.rough * h.mat.rough, 1e-6);
      let alph = sqrt(a2);
      let o = -normalize(rd);
      let hh = ggx_sample(rd, h.n, alph, rng);
      if (dot(hh, h.n) <= 0.0) { failed = true; break; }
      var wi = reflect(rd, hh);
      wi = normalize(wi);
      let cos_i = dot(wi, h.n);
      if (cos_i <= 0.0) { failed = true; break; }
      // VNDF estimator: weight = albedo * G1(cos_i) / (n·h)
      tp = tp * h.mat.alb * (g1(cos_i, a2) / max(dot(h.n, hh), 1e-6));
      ro = h.p + h.n * 0.002;
      rd = wi;
      continue;
    }

    // diffuse: mixture importance sampling
    let ldir = light_gen(h.p, rng);
    let cdir = onb(h.n) * cosine_dir(rng);
    var chosen = ldir;
    if (rand(rng) >= 0.5) { chosen = cdir; }
    chosen = normalize(chosen);

    let pl = light_pdf(h.p, chosen);
    let pc = max(0.0, dot(h.n, chosen)) / PI;
    let pdf = 0.5 * pl + 0.5 * pc;
    if (pdf < 1e-8) { failed = true; break; }
    let scat = max(0.0, dot(h.n, chosen)) / PI;
    if (scat <= 0.0) { failed = true; break; }

    tp = tp * tex_color(h.mat.tex, h.p, h.mat.alb) * (scat / pdf);
    ro = h.p + h.n * 0.002;
    rd = chosen;
  }

  if (DEBUG_COLORS == 1i && failed) { return vec3f(8.0, 0.0, 8.0); }
  if (DEBUG_COLORS == 1i && !(radiance.x == radiance.x && radiance.y == radiance.y && radiance.z == radiance.z)) {
    return vec3f(0.0, 8.0, 0.0);
  }
  return radiance;
}

fn aces_curve(v: f32) -> f32 {
  return (v * (2.51 * v + 0.03)) / (v * (2.43 * v + 0.59) + 0.14);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let W = u32(u.dims.x);
  let H = u32(u.dims.y);
  if (gid.x >= W || gid.y >= H) { return; }

  let idx = gid.y * W + gid.x;

  let frame = u.dims.z;
  let spp = u.dims.w;

  var rng = hash_seed(gid.x, gid.y, frame);

  let fwd = normalize(u.cam_tgt.xyz - u.cam_pos_f.xyz);
  let right = normalize(cross(fwd, vec3f(0.0, 1.0, 0.0)));
  let up = cross(right, fwd);
  let aspect = f32(W) / f32(H);
  let tanf = u.cam_pos_f.w;

  var radiance = vec3f(0.0);
  for (var s = 0u; s < spp; s = s + 1u) {
    let jitter = vec2f(rand(&rng), rand(&rng)) - vec2f(0.5, 0.5);
    let uv = (vec2f(f32(gid.x), f32(gid.y)) + vec2f(0.5, 0.5) + jitter) / vec2f(f32(W), f32(H));
    let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let dir = normalize(ndc.x * right * tanf * aspect + ndc.y * up * tanf + fwd);
    // firefly clamp: cap each sample so rare bright specular spikes don't sparkle
    radiance = radiance + min(trace(u.cam_pos_f.xyz, dir, &rng), vec3f(20.0));
  }

  let prev = select(accum[idx].rgb, vec3f(0.0), frame == 0u);
  let total = prev + radiance;
  accum[idx] = vec4f(total, 1.0);

  let count = f32(frame + 1u) * f32(spp);
  var c = vec3f(aces_curve(total.r / count), aces_curve(total.g / count), aces_curve(total.b / count));
  c = pow(c, vec3f(1.0 / 2.2));

  // temporal EMA denoiser: while the camera is moving (small frame count) the
  // displayed image blends heavily against history, cutting drag noise; the
  // blend converges to 1.0 when stationary so accumulation stays unbiased.
  let hist = history[idx].rgb;
  let blend = clamp(f32(frame + 1u) / 24.0, 0.08, 1.0);
  let shown = mix(hist, c, blend);
  history[idx] = vec4f(shown, 1.0);
  textureStore(out_tex, vec2i(gid.xy), vec4f(shown, 1.0));
}
`;

export const BLIT_WGSL = /* wgsl */ `
struct VOut { @builtin(position) pos: vec4f };

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VOut {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var o: VOut;
  o.pos = vec4f(p[vi], 0.0, 1.0);
  return o;
}

@group(0) @binding(0) var tex : texture_2d<f32>;

@fragment
fn fs(@builtin(position) fp: vec4f) -> @location(0) vec4f {
  return textureLoad(tex, vec2i(fp.xy), 0);
}
`;
