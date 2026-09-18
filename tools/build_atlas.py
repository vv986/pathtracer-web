"""Bistro 漫反射贴图图集构建器 v2。

升级:hero 贴图按「材质可见面积」选取(实例展开后的三角形面积,含节点变换),
而不是按贴图原始分辨率。头部材质(占大部分画面)获得 1024² 贴图,其余 256²/128²。

用法: python tools/build_atlas.py [源仓库目录]
输出: models/atlas_0..N.jpg + models/bistro_tex.json
"""
import json
import os
import struct
import sys
from collections import defaultdict

from PIL import Image

SRC = sys.argv[1] if len(sys.argv) > 1 else 'D:/bistro-src'
WEB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(WEB, 'models')

PAGE = 2048
TILE_HERO = 1024   # hero 材质贴图上限
TILE_MID = 256     # 次级
TILE_STD = 128     # 其余
HERO_COUNT = 20    # hero 材质数量(按可见面积)
MID_COUNT = 60     # 次级材质数量
MAX_PAGES = 8
MAX_DEPTH_HERO = 24

os.chdir(SRC)
g = json.load(open('bistro.gltf', encoding='utf-8'))
imgs = g['images']
texs = g['textures']
mats = g['materials']
meshes = g['meshes']
nodes = g['nodes']


def resolve_diff_uri(mat):
    ext = mat.get('extensions', {}).get('KHR_materials_pbrSpecularGlossiness', {})
    dt = ext.get('diffuseTexture')
    if dt is None:
        return None
    ti = dt.get('index')
    if ti is None:
        return None
    src = texs[ti].get('source')
    if src is None:
        return None
    uri = imgs[src]['uri']
    for cand in (uri, os.path.splitext(uri)[0] + '.dds'):
        if os.path.exists(cand):
            return cand
    return None


def read_accessor(ai):
    a = g['accessors'][ai]
    v = g['bufferViews'][a['bufferView']]
    base = (v.get('byteOffset') or 0) + (a.get('byteOffset') or 0)
    ncomp = {'VEC3': 3, 'VEC2': 2, 'SCALAR': 1}[a['type']]
    csz = {5126: 4, 5123: 2, 5125: 4}[a['componentType']]
    stride = v.get('byteStride') or ncomp * csz
    buf = bin_buffer[v.get('buffer', 0)]
    out = []
    for i in range(a['count']):
        s = base + i * stride
        vals = []
        for k in range(ncomp):
            off = s + k * csz
            if a['componentType'] == 5126:
                vals.append(struct.unpack_from('<f', buf, off)[0])
            elif a['componentType'] == 5123:
                vals.append(struct.unpack_from('<H', buf, off)[0])
            else:
                vals.append(struct.unpack_from('<I', buf, off)[0])
        out.append(vals)
    return out


# 二进制 buffer(bistro.bin)
bin_buffer = []
for b in g['buffers']:
    if 'uri' in b:
        bin_buffer.append(open(b['uri'], 'rb').read())
    else:
        bin_buffer.append(b'')

# ---- 节点世界矩阵 + 实例展开的材质三角形数(面积代理) ----
children = defaultdict(list)
is_child = set()
for i, n in enumerate(nodes):
    for c in n.get('children', []):
        children[i].append(c)
        is_child.add(c)


def mat_mul(a, b):
    o = [0.0] * 16
    for c in range(4):
        for r in range(4):
            o[c * 4 + r] = sum(a[k * 4 + r] * b[c * 4 + k] for k in range(4))
    return o


def trs(n):
    if 'matrix' in n:
        return list(n['matrix'])
    m = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    s = n.get('scale', [1, 1, 1])
    if 'rotation' in n:
        x, y, z, w = n['rotation']
        x2, y2, z2 = x + x, y + y, z + z
        xx, xy, xz = x * x2, x * y2, x * z2
        yy, yz, zz = y * y2, y * z2, z * z2
        wx, wy, wz = w * x2, w * y2, w * z2
        r = [1 - (yy + zz), xy + wz, xz - wy,
             xy - wz, 1 - (xx + zz), yz + wx,
             xz + wy, yz - wx, 1 - (xx + yy)]
        for c in range(3):
            for r2 in range(3):
                m[c * 4 + r2] = r[r2 * 3 + c] * s[c] if False else r[r2 * 3 + c] * s[r2]
    else:
        for c in range(3):
            m[c * 4 + c] = s[c]
    if 'translation' in n:
        m[12], m[13], m[14] = n['translation']
    return m


roots = [i for i in range(len(nodes)) if i not in is_child]
world = [None] * len(nodes)
stack = [(r, trs(nodes[r])) for r in roots]
while stack:
    i, m = stack.pop()
    if world[i] is not None:
        continue
    world[i] = m
    for c in children[i]:
        stack.append((c, mat_mul(m, trs(nodes[c]))))


def det3(m):
    return abs(m[0] * m[5] * m[10] + m[4] * m[9] * m[2] + m[8] * m[1] * m[6]
               - m[8] * m[5] * m[2] - m[4] * m[1] * m[10] - m[0] * m[9] * m[6])


tris_per_mat = defaultdict(int)
for i, n in enumerate(nodes):
    if 'mesh' not in n:
        continue
    d = abs(det3(world[i]))
    for prim in meshes[n['mesh']].get('primitives', []):
        mi = prim.get('material', 0)
        ai = prim.get('indices')
        cnt = (g['accessors'][ai]['count'] if ai is not None else 0) // 3
        tris_per_mat[mi] += cnt * max(d, 1e-6)

print('top materials by tris:', sorted(tris_per_mat.items(), key=lambda kv: -kv[1])[:5])

# ---- 材质 → 贴图,按可见面积排序 ----
mat_area = {}
for mi in range(len(mats)):
    u = resolve_diff_uri(mats[mi])
    if u:
        mat_area[mi] = (u, tris_per_mat.get(mi, 0))

# 汇总每张贴图的总可见面积
tex_area = defaultdict(float)
for mi, (u, a) in mat_area.items():
    tex_area[u] += a

ranked = sorted(tex_area.keys(), key=lambda u: -tex_area[u])
heroes = set(ranked[:HERO_COUNT])
mids = set(ranked[HERO_COUNT:HERO_COUNT + MID_COUNT])
print(f'heroes: {len(heroes)}, mids: {len(mids)}, std: {len(ranked) - len(heroes) - len(mids)}')


def tile_size(u):
    w, h = info[u] if False else (0, 0)
    return (0, 0)


# 载入尺寸(懒加载一次)
info = {}
for u in tex_area:
    with Image.open(u) as im:
        info[u] = im.size


def tile_size(u):
    w, h = info[u]
    if u in heroes:
        box = TILE_HERO
    elif u in mids:
        box = TILE_MID
    else:
        box = TILE_STD
    s = min(box / w, box / h, 1.0)
    return max(8, int(w * s)), max(8, int(h * s))


# ---- shelf 装箱 ----
pages = []


def new_page():
    p = {'img': Image.new('RGB', (PAGE, PAGE), (110, 110, 110)), 'x': 0, 'y': 0, 'rowh': 0}
    pages.append(p)
    return p


placed = {}
cur = new_page()
items = sorted(((u, *tile_size(u)) for u in tex_area), key=lambda x: -x[2])
for k, (u, tw, th) in enumerate(items):
    if cur['x'] + tw > PAGE:
        cur['y'] += cur['rowh']
        cur['x'] = 0
        cur['rowh'] = 0
    if cur['y'] + th > PAGE:
        if len(pages) >= MAX_PAGES:
            print('WARN: 页数超限,剩余贴图降级为 128²')
            while k < len(items):
                u2 = items[k][0]
                tw2, th2 = TILE_STD, TILE_STD
                tw2, th2 = min(tw2, items[k][1]), min(th2, items[k][2])
                if cur['x'] + tw2 > PAGE:
                    cur['y'] += cur['rowh']; cur['x'] = 0; cur['rowh'] = 0
                if cur['y'] + th2 > PAGE:
                    cur = new_page()
                im = Image.open(u2).convert('RGB').resize((tw2, th2), Image.LANCZOS)
                cur['img'].paste(im, (cur['x'], cur['y']))
                placed[u2] = (len(pages) - 1, cur['x'], cur['y'], tw2, th2)
                cur['x'] += tw2
                k += 1
            break
        cur = new_page()
    im = Image.open(u).convert('RGB')
    if im.size != (tw, th):
        im = im.resize((tw, th), Image.LANCZOS)
    cur['img'].paste(im, (cur['x'], cur['y']))
    placed[u] = (len(pages) - 1, cur['x'], cur['y'], tw, th)
    cur['x'] += tw
    if th > cur['rowh']:
        cur['rowh'] = th
    if k % 20 == 0:
        print(f'  packed {k + 1}/{len(items)}')

# ---- 输出 ----
slots = {}
for mi, (u, _) in mat_area.items():
    if u in placed:
        pg, x, y, w, h = placed[u]
        slots[str(mi)] = [x / PAGE, y / PAGE, w / PAGE, h / PAGE, pg, 1]

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, 'bistro_tex.json'), 'w') as f:
    json.dump({'pages': len(pages), 'slots': slots}, f)
for i, p in enumerate(pages):
    p['img'].save(os.path.join(OUT, f'atlas_{i}.jpg'), quality=90)
    print(f'atlas_{i}.jpg saved ({os.path.getsize(os.path.join(OUT, f"atlas_{i}.jpg")) // 1024} KB)')
print(f'DONE: {len(pages)} pages, {len(placed)}/{len(tex_area)} textures placed')
