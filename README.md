# Path Tracer Web — GPU 实时路径追踪(WebGPU)

CPU 版路径追踪器(`../pathtracer`)的实时化里程碑:同样的核心算法(混合重要性采样、玻璃
delta 路径、自发光、ACES 色调映射)跑在 WebGPU compute shader 里,浏览器内实时渲染、可交互。

**五个场景**(顶部按钮切换):

| 场景 | 几何 | 亮点 |
|------|------|------|
| 户外场景 | ~110 球体 + 金属盒 | 太阳立体角采样、棋盘格、程序化大理石、GGX 金属 |
| 兔子模型 | 斯坦福兔子 69k tris(OBJ + BVH) | 材质下拉框:镜面/金色/玻璃/哑光 |
| Sponza | 262k tris(OBJ + BVH) | 克罗地亚中庭建筑,拱廊与软阴影 |
| Bistro | **4.2M tris**(Amazon Lumberyard,glTF 实例展开) | 百万面 BVH 压力测试,254 种材质调色板 |
| Cornell Box | 18 四边形 + 玻璃球 | 面光源重要性采样基准场景 |

## 运行

WebGPU 要求安全上下文,用本地服务器打开(需要 Chrome / Edge 113+):

```bat
cd pathtracer-web
python -m http.server 8765
```

然后访问 http://localhost:8765/

老版本单文件 Cornell 场景保留在 `cornell.html`。模型来自
[McGuire Graphics Archive](https://casual-effects.com/data/) 与
[zeux/niagara_bistro](https://github.com/zeux/niagara_bistro),放在 `models/` 下。

## 操作

| 操作 | 效果 |
|------|------|
| 按住拖拽 | 轨道旋转(限制在场景范围内) |
| 滚轮 | 推近 / 拉远 |
| 移动视角 | 重置累积;**时域 EMA 降噪**让拖动中的画面保持可读(代价是轻微拖影) |
| 静止不动 | 持续累积到 4000 spp 后自动停止(省 GPU) |
| 什么都不做 | **自适应分辨率调节器**自动把帧率锁在 ~30 fps(帧时间反馈,动态升降内部分辨率) |

重场景的性能分级:Sponza 基准 66% 分辨率 / 8 层弹射,Bistro 50% / 8 层;
调节器在此之上继续动态微调(下限 30%),配合降噪器保证流畅与画面的平衡。

## 实现

无依赖、无构建步骤,ES module 分文件:

- `shaders.js` — 两套 WGSL 管线:
  - **prim 管线**:解析图元(四边形 + 球体存储缓冲)
  - **mesh 管线**:索引化顶点 + BVH 节点栈式遍历 + Möller–Trumbore + 平滑法线 + 材质调色板
- 共享核心:PCG 随机数、ONB、余弦/球面立体角采样、50/50 混合重要性采样(灯光 ↔ 余弦半球)、
  GGX **VNDF** 金属采样、Schlick 菲涅尔、**俄罗斯轮盘赌终止**(3 次弹射后按吞吐量概率)、
  **萤火虫钳制**(单采样亮度上限)、**薄壁玻璃**( architectural windows 直通染色)、
  棋盘/大理石程序化纹理、ACES 色调映射、**时域 EMA 降噪**(history 缓冲,blend 随帧数从 0.08 → 1)
- `bvh.js` — 类型化数组的 OBJ 解析器(支持百万行)、glTF 解析器(二进制 buffer、TRS 层级
  变换、实例展开、specGlossiness 材质)、**分箱 SAH BVH**(表面积启发式,遍历成本较中位数
  切分低 20~40%,175 万三角形秒级构建)
- `scenes.js` — 五个场景的几何打包
- `tools/build_atlas.py` — Bistro 贴图图集构建器:156 张 DDS 漫反射贴图(Pillow 解码 DX10)→
  分级缩放(头部 512²/其余 128²)→ shelf 装箱进 2 页 2048² JPEG + 材质→图块 UV 映射表
- `app.js` — 管线/绑定组管理、轨道相机、渐进式累积、场景切换、设备上限协商、rAF 看门狗、错误面板

## 踩过的坑(记录供参考)

- `meta` 是 WGSL 保留字
- storage buffer 里打包的整数索引必须以 f32 写、着色器 `u32()` 转换读(声明 vec4u 会读到位模式错误)
- 绑定组校验:dummy 缓冲也必须 ≥ 结构体 minBindingSize
- 三角形网格的次级光线要沿法线偏移起点,否则自相交痂斑
- 金属 GGX 必须用 **VNDF** 采样:NDF 采样在掠射角产生大量低于表面的无效样本,剪影系统性发黑
- 自动化/遮挡环境里 devicePixelRatio 会抖动 → 每帧重建缓冲、累积永远清零(DPR 必须冻结)
- 百万面模型必须用**索引化顶点**(共享顶点 + u32 索引),逐三角形复制会爆显存
- mesh 着色器用了 10 个 storage buffer,需要协商 `maxStorageBuffersPerShaderStage`
- python http.server 不带缓存头,Chrome 会按启发式缓存模块文件,**改代码不生效**——
  `serve.py` 发 no-cache 头,开发时用 `python serve.py` 代替
- 重场景按 `resScale`(内部分辨率)和 `maxDepth`(弹射深度)分级:Bistro 0.5/8,Sponza 0.66/8
- BVH 索引缓冲分配成 `Uint32Array(numTris)` 而写入 3×numTris:类型数组越界写**静默丢弃**,
  网格被"虫蛀"成残片——64 位机器上乘法别省
- 单线程 `http.server` 会被一个挂起的浏览器连接卡死(所有后续请求超时),必须用线程池版服务器
- GPU 调试读回三件套:`mapAsync` 前必须 `await queue.onSubmittedWorkDone()`(否则读到未初始化的零);
  源缓冲必须带 `COPY_SRC`(缺了不报错,copy 被验证拒绝,读回还是零);
  `getMappedRange()` 只能调一次(第二次返回重叠范围错误)。仪器先于结论——统计读数异常时先怀疑仪表

## 性能参考(iGPU,自动化遮挡环境;前台可见时 rAF 通常跑满 60 fps)

| 场景 | 分辨率 | 帧率 |
|------|--------|------|
| Cornell / 户外 | 1280×720 @2spp | 60+ fps(实测峰值 417 ticks/s) |
| Sponza(262k tris) | 1280×720 @2spp | ~15 fps |
| Bistro(**4.2M tris**) | 1548×871 @2spp | ~2 fps |

## 与 CPU 版对照

| | CPU 版 (`../pathtracer`) | GPU 版(本项目) |
|---|---|---|
| 算法 | 路径追踪 + 混合重要性采样 | 相同 |
| 场景 | Cornell Box / 户外(运动模糊、体积雾) | 五场景,最大 4.2M tris |
| 画质上限 | 400+ spp,数分钟 | 交互 2 spp,静止收敛到 4000 spp |
| 交互 | 无(命令行) | 实时轨道相机、场景切换、时域降噪 |
