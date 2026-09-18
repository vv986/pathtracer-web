"""Bistro 漫反射贴图图集构建器。

从 niagara_bistro 仓库读取 glTF 材质引用的 *_diff.dds(Pillow 解码 DX10),
按分辨率分级(头部 512² / 其余 128²),shelf 装箱进 2048² 的 JPEG 图集页,
输出: models/atlas_0.jpg ... + models/bistro_tex.json

用法: python tools/build_atlas.py [源仓库目录]
"""
import json
import os
import sys

from PIL import Image

SRC = sys.argv[1] if len(sys.argv) > 1 else 'D:/bistro-src'
WEB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(WEB, 'models')

PAGE = 2048
TILE_HERO = 512
TILE_STD = 128
HERO_COUNT = 24
MAX_PAGES = 8
MAX_DEPTH_HERO = 24

os.chdir(SRC)
g = json.load(open('bistro.gltf', encoding='utf-8'))
imgs = g['images']
texs = g['textures']
mats = g['materials']


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
    if os.path.exists(uri):
        return uri
    base, _ = os.path.splitext(uri)
    dds = base + '.dds'
    if os.path.exists(dds):
        return dds
    return None


entries = []
for i, m in enumerate(mats):
    u = resolve_diff_uri(m)
    if u:
        entries.append((i, u))

unique = {}
for mi, u in entries:
    unique.setdefault(u, []).append(mi)
print(f'materials with diffuse tex: {len(entries)}, unique textures: {len(unique)}')

info = {}
for u in unique:
    with Image.open(u) as im:
        info[u] = im.size

ranked = sorted(unique.keys(), key=lambda u: -(info[u][0] * info[u][1]))
heroes = set(ranked[:HERO_COUNT])


def tile_size(u):
    w, h = info[u]
    box = TILE_HERO if u in heroes else TILE_STD
    s = min(box / w, box / h, 1.0)
    return max(8, int(w * s)), max(8, int(h * s))


pages = []


def new_page():
    p = {'img': Image.new('RGB', (PAGE, PAGE), (110, 110, 110)),
         'x': 0, 'y': 0, 'rowh': 0}
    pages.append(p)
    return p


placed = {}  # uri -> (pageIdx, x, y, w, h)
cur = new_page()
items = sorted(((u, *tile_size(u)) for u in unique), key=lambda x: -x[2])
for k, (u, tw, th) in enumerate(items):
    if cur['x'] + tw > PAGE:
        cur['y'] += cur['rowh']
        cur['x'] = 0
        cur['rowh'] = 0
    if cur['y'] + th > PAGE:
        if len(pages) >= MAX_PAGES:
            raise RuntimeError('图集页数超过上限,需要降低贴图分级')
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

slots = {}
for i in range(len(mats)):
    u = resolve_diff_uri(mats[i])
    if u and u in placed:
        pg, x, y, w, h = placed[u]
        slots[str(i)] = [x / PAGE, y / PAGE, w / PAGE, h / PAGE, pg, 1]
    else:
        slots[str(i)] = [0.0, 0.0, 0.0, 0.0, 0, 0]

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, 'bistro_tex.json'), 'w') as f:
    json.dump({'pages': len(pages), 'slots': slots}, f)
for i, p in enumerate(pages):
    p['img'].save(os.path.join(OUT, f'atlas_{i}.jpg'), quality=90)
    print(f'atlas_{i}.jpg saved')
print(f'DONE: {len(pages)} pages, {len(placed)}/{len(unique)} textures placed')
