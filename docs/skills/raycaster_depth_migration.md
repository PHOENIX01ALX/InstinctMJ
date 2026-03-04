# Skill: Raycaster / 深度图迁移指南 — 旧框架 → mjlab

## 概述

本文档描述如何将 旧框架 的 RayCaster / RayCasterCamera 系统及其 InstinctLab 扩展（GroupedRayCaster、NoisyCamera）迁移到 mjlab，全部使用 **mjlab 原生 API 风格**。

The biggest change is configuration style: 旧框架 uses nested @configclass definitions; mjlab uses dictionaries of config objects.

---

## 1. 类层次对照

### 旧框架 原版

```
SensorBase
├── RayCaster              # 基础网格光线投射
│   ├── RayCasterCamera    # 针孔相机光线投射（深度图）
│   └── (InstinctLab)
│       └── GroupedRayCaster           # 多动态网格 + 碰撞分组
│           └── GroupedRayCasterCamera # 相机版 grouped raycaster
│               └── (NoisyCameraMixin)
│                   └── NoisyGroupedRayCasterCamera  # 含噪声管线 + 历史缓冲
```

### mjlab 原版

```
Sensor[T]  (ABC, Generic)
├── RayCastSensor[RayCastData]    # 基于 mujoco_warp.rays() 的 BVH 光线投射
├── CameraSensor[CameraSensorData] # 原生 MuJoCo 渲染相机（RGB + Depth）
├── ContactSensor[ContactData]
└── BuiltinSensor
```

### Instinct_mjlab 迁移目标

```
RayCastSensor (mjlab)                     # 高度扫描直接使用
└── GroupedRayCaster                      # 扩展：drift / ray_starts 缓冲区支持随机化
    └── GroupedRayCasterCamera            # 扩展：相机内参 + distance_to_image_plane 计算
        └── (NoisyCameraMixin)
            └── NoisyGroupedRayCasterCamera  # 扩展：噪声管线 + 历史缓冲
```

**核心原则**: 底层继承 `mjlab.sensor.RayCastSensor`，使用 `mujoco_warp.rays()` BVH 加速光线投射。配置全部采用 mjlab 原生字段（`frame=ObjRef`、`ray_alignment`、`include_geom_groups` 等），不保留 旧框架 风格的 `prim_path`、`mesh_prim_paths`、`attach_yaw_only` 等字段。

---

## 2. 传感器配置映射

### 2.1 高度扫描（RayCastSensorCfg）

**旧框架:**
```python
from legacy_framework.sensors import RayCasterCfg, patterns

height_scanner = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/torso_link",
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    attach_yaw_only=True,
    pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
    mesh_prim_paths=["/World/ground"],
    debug_vis=False,
    max_distance=100.0,
)
```

**mjlab:**
```python
from mjlab.sensor import RayCastSensorCfg, GridPatternCfg, ObjRef

height_scanner = RayCastSensorCfg(
    name="height_scanner",
    frame=ObjRef(type="body", name="torso_link", entity="robot"),
    pattern=GridPatternCfg(
        resolution=0.1,
        size=(1.6, 1.0),
    ),
    ray_alignment="yaw",
    max_distance=100.0,
    include_geom_groups=(0,),
    exclude_parent_body=False,
    debug_vis=False,
)
```

### 2.2 深度相机（GroupedRayCasterCamera + 噪声管线）

mjlab 原生 `RayCastSensorCfg` 不支持「深度图像投影」和「噪声管线」，需使用 Instinct_mjlab 扩展类。但配置字段全部用 mjlab 原生风格。

**旧框架:**
```python
from instinctlab.sensors import NoisyGroupedRayCasterCameraCfg
from legacy_framework.sensors import patterns

camera = NoisyGroupedRayCasterCameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/torso_link",
    mesh_prim_paths=["/World/ground/", "/World/envs/env_.*/Robot/.*"],
    aux_mesh_and_link_names={"torso_link_rev_1_0": None, "head_link": "head_link"},
    offset=NoisyGroupedRayCasterCameraCfg.OffsetCfg(
        pos=(0.048, 0.015, 0.463), rot=(0.998, 0.009, 0.407, 0.0), convention="world",
    ),
    attach_yaw_only=False,
    pattern_cfg=patterns.PinholeCameraPatternCfg(
        focal_length=1.0,
        horizontal_aperture=2 * math.tan(math.radians(87) / 2),
        vertical_aperture=2 * math.tan(math.radians(58) / 2),
        height=27, width=48,
    ),
    data_types=["distance_to_image_plane"],
    noise_pipeline={
        "normalize": DepthNormalizationCfg(depth_range=(0.0, 2.0), normalize=True),
        "crop_and_resize": CropAndResizeCfg(crop_region=(2, 2, 2, 2), resize_shape=(18, 32)),
    },
    depth_clipping_behavior="max",
    min_distance=0.05,
)
```

**mjlab:**
```python
from mjlab.sensor import ObjRef, PinholeCameraPatternCfg
from instinct_mjlab.sensors.noisy_camera import NoisyGroupedRayCasterCameraCfg
from instinct_mjlab.utils.noise import DepthNormalizationCfg, CropAndResizeCfg

camera = NoisyGroupedRayCasterCameraCfg(
    name="camera",
    frame=ObjRef(type="body", name="torso_link", entity="robot"),
    pattern=PinholeCameraPatternCfg(
        fovy=58.0,            # 垂直视场角（度），替代 focal_length + aperture
        height=27,
        width=48,
    ),
    offset=NoisyGroupedRayCasterCameraCfg.OffsetCfg(
        pos=(0.048, 0.015, 0.463),
        rot=(0.998, 0.009, 0.407, 0.0),
        convention="world",
    ),
    ray_alignment="base",     # 相机用完整旋转，不用 "yaw"
    max_distance=100.0,
    include_geom_groups=(0, 1, 2, 3, 4, 5),  # 包含所有 geom group（含机器人自身）
    exclude_parent_body=False,
    data_types=["distance_to_image_plane"],
    noise_pipeline={
        "normalize": DepthNormalizationCfg(depth_range=(0.0, 2.0), normalize=True),
        "crop_and_resize": CropAndResizeCfg(crop_region=(2, 2, 2, 2), resize_shape=(18, 32)),
    },
    depth_clipping_behavior="max",
    min_distance=0.05,
)
```

---

## 3. 配置字段映射详表

### 3.1 RayCasterCfg → RayCastSensorCfg

| 旧框架 | mjlab | 说明 |
|---|---|---|
| `prim_path="{ENV_REGEX_NS}/Robot/torso_link"` | `frame=ObjRef(type="body", name="torso_link", entity="robot")` | body/site/geom 三种引用类型 |
| `offset=OffsetCfg(pos=...)` | 直接编码到 pattern 的射线起点 | 高度偏移在观测函数 `offset` 参数中处理 |
| `attach_yaw_only=True` | `ray_alignment="yaw"` | 可选 `"base"` / `"yaw"` / `"world"` |
| `pattern_cfg=GridPatternCfg(...)` | `pattern=GridPatternCfg(...)` | 字段名变更 |
| `mesh_prim_paths=["/World/ground"]` | `include_geom_groups=(0,)` | 按 MuJoCo geom group 过滤 |
| `max_distance=100.0` | `max_distance=100.0` | 直接对应 |
| `debug_vis=True` | `debug_vis=True` + `VizCfg(...)` | mjlab 支持更精细的可视化配置 |
| — | `exclude_parent_body=True` | mjlab 新增：排除自身 body 的碰撞 |

### 3.2 GroupedRayCasterCameraCfg 扩展字段

| 字段 | 说明 |
|---|---|
| `OffsetCfg(pos, rot, convention)` | 相机安装偏移，convention 支持 `"opengl"` / `"ros"` / `"world"` |
| `data_types=["distance_to_image_plane"]` | 输出数据类型，可选 `"distance_to_camera"` / `"normals"` |
| `depth_clipping_behavior` | `"max"` / `"zero"` / `"none"`，控制超距裁剪行为 |
| `min_distance=0.05` | 忽略过近的命中点 |

### 3.3 PinholeCameraPatternCfg

| 旧框架 | mjlab | 说明 |
|---|---|---|
| `focal_length` + `horizontal_aperture` + `vertical_aperture` | `fovy`（垂直视场角，度） | mjlab 用单一 fovy 参数 |
| `height`, `width` | `height`, `width` | 直接对应 |
| `.func()` 生成射线 | `.generate_rays(mj_model, device)` | 生成方法名不同 |

> mjlab 原生 `PinholeCameraPatternCfg` 也提供 `from_mujoco_camera(model, cam_id)` 和 `from_intrinsic_matrix(K, width, height)` 工厂方法。

---

## 4. 数据结构差异

### 4.1 RayCastData

| 旧框架 `RayCasterData` | mjlab `RayCastData` | 说明 |
|---|---|---|
| `pos_w` [N, 3] | `pos_w` [B, 3] | 传感器原点世界坐标 |
| `quat_w` [N, 4] | `quat_w` [B, 4] | 传感器朝向 |
| `ray_hits_w` [N, R, 3] | `hit_pos_w` [B, R, 3] | 命中点世界坐标，**字段名变更** |
| — | `distances` [B, R] | 射线距离（mjlab 新增） |
| — | `normals_w` [B, R, 3] | 命中面法线（mjlab 新增） |

### 4.2 CameraData（GroupedRayCasterCamera 输出）

| 字段 | 形状 | 说明 |
|---|---|---|
| `pos_w` | [N, 3] | 相机世界位置 |
| `quat_w_world` | [N, 4] | 相机世界朝向（world convention） |
| `intrinsic_matrices` | [N, 3, 3] | 内参矩阵 |
| `image_shape` | (H, W) | 图像尺寸 |
| `output["distance_to_image_plane"]` | [N, H, W, 1] | 深度图（像平面距离） |
| `output["distance_to_camera"]` | [N, H, W, 1] | 深度图（相机距离） |
| `output["normals"]` | [N, H, W, 3] | 法线图 |
| `output["{type}_noised"]` | 同上 | 噪声处理后（NoisyCameraMixin） |
| `output["{type}_noised_history"]` | [N, T, H, W, C] | 历史帧缓冲（NoisyCameraMixin） |

### 4.3 mjlab 原生 CameraSensor vs GroupedRayCasterCamera

| 特性 | `CameraSensor` (mjlab 原生) | `GroupedRayCasterCamera` (Instinct 扩展) |
|---|---|---|
| 底层实现 | MuJoCo 渲染管线 | `mujoco_warp.rays()` BVH 光线投射 |
| RGB 支持 | ✅ `uint8 [B,H,W,3]` | ❌ |
| Depth 支持 | ✅ `float32 [B,H,W,1]` | ✅ `distance_to_image_plane` / `distance_to_camera` |
| 噪声管线 | ❌ | ✅ NoisyCameraMixin |
| 历史缓冲 | ❌ | ✅ AsyncCircularBuffer |
| 适用场景 | RGB 渲染、简单深度 | 噪声深度图、历史帧采样、域随机化 |

---

## 5. 执行流程对照

### 旧框架

```
RayCaster._initialize_warp_meshes()  → 加载 USD mesh 到 warp
SensorBase._update_impl()
    → wp.mesh_query_ray()            → Warp mesh raycast
    → 更新 ray_hits_w
```

### mjlab 三阶段流程

```
Sensor.initialize()
    → RayCastSensor.initialize()
        → PatternCfg.generate_rays()      # 生成本地射线方向
    ↓
SensorContext.prepare()                    # PRE-GRAPH (PyTorch)
    → RayCastSensor.prepare_rays()
        → 旋转射线到世界坐标系
        → 写入 _ray_pnt / _ray_vec (warp arrays)
    ↓
mujoco_warp.rays()                         # IN-GRAPH (CUDA graph)
    → BVH 加速光线投射
    → 写入 _ray_dist / _ray_geomid
    ↓
SensorContext.finalize()                   # POST-GRAPH (PyTorch)
    → RayCastSensor.postprocess_rays()
        → 计算 hit_pos_w, distances, normals_w
```

### GroupedRayCasterCamera 扩展流程

```
GroupedRayCaster.prepare_rays()            # 覆写 pre-graph
    → 提取 frame 位姿 (body/site/geom)
    → 按 ray_alignment 变换射线
    → 应用 drift 偏移
    → 写入 _ray_pnt / _ray_vec
    ↓
mujoco_warp.rays()                         # 不变
    ↓
GroupedRayCasterCamera.postprocess_rays()   # 覆写 post-graph
    → 计算 distance_to_image_plane (像平面投影)
    → 应用 depth_clipping_behavior
    → 更新 CameraData.output 字典
    ↓
NoisyGroupedRayCasterCamera.postprocess_rays()  # 覆写
    → apply_noise_pipeline_to_all_data_types()
    → update_history_buffers()
```

---

## 6. 场景挂载方式

### 旧框架

```python
@configclass
class MySceneCfg(InteractiveSceneCfg):
    height_scanner = RayCasterCfg(prim_path="...", ...)
    camera = NoisyGroupedRayCasterCameraCfg(prim_path="...", ...)
```

### mjlab（纯函数风格）

```python
from mjlab.scene import SceneCfg
from mjlab.sensor import RayCastSensorCfg, GridPatternCfg, ObjRef, ContactSensorCfg, ContactMatch

scene = SceneCfg(
    terrain=TerrainImporterCfg(terrain_type="plane"),
    num_envs=4096,
)
scene.entities = {"robot": get_g1_robot_cfg()}
scene.sensors = (
    RayCastSensorCfg(
        name="height_scanner",
        frame=ObjRef(type="body", name="torso_link", entity="robot"),
        pattern=GridPatternCfg(resolution=0.1, size=(1.6, 1.0)),
        ray_alignment="yaw",
        include_geom_groups=(0,),
    ),
    camera_cfg,          # NoisyGroupedRayCasterCameraCfg 实例
    contact_sensor_cfg,  # ContactSensorCfg 实例
)
```

### Instinct_mjlab 推荐写法（字典工厂）

```python
def make_perceptive_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    scene = SceneCfg(
        terrain=TerrainImporterCfg(terrain_type="plane"),
        num_envs=1 if play else 4096,
        entities={"robot": get_g1_robot_cfg()},
        sensors=(height_scanner_cfg(), camera_cfg(), contact_sensor_cfg()),
    )
    observations = {
        "actor": ObservationGroupCfg(terms=actor_terms(), concatenate_terms=False),
        "critic": ObservationGroupCfg(terms=critic_terms(), concatenate_terms=False),
    }
    return ManagerBasedRlEnvCfg(
        scene=scene,
        observations=observations,
        rewards=reward_terms(),
        events=event_terms(play=play),
        terminations=termination_terms(),
        actions=action_terms(),
        commands=command_terms(),
    )
```

---

## 7. 观测函数迁移

### 7.1 height_scan

**旧框架:**
```python
height_scan = ObsTerm(
    func=mdp.height_scan,
    params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 20.0},
)
```

**mjlab:**
```python
height_scan = ObservationTermCfg(
    func=mdp.height_scan,
    params={"sensor_name": "height_scanner", "offset": 20.0},
    clip=[-20.0, 20.0],
)
```

**差异**: `sensor_cfg: SceneEntityCfg` → `sensor_name: str`

mjlab 的 `height_scan` 实现：
```python
def height_scan(env, sensor_name: str, offset: float = 0.0) -> torch.Tensor:
    sensor: RayCastSensor = env.scene[sensor_name]
    return sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.hit_pos_w[..., 2] - offset
```

### 7.2 visualizable_image（深度图观测）

```python
depth_image = ObservationTermCfg(
    func=instinct_mdp.visualizable_image,
    params={
        "sensor_cfg": SceneEntityCfg("camera"),
        "data_type": "distance_to_image_plane_noised",
    },
)
```

`visualizable_image` 从 `sensor.data.output[data_type]` 读取数据：
- 输入 `[N, H, W, C]` → 输出 `[N, C, H, W]`（permute）
- 历史模式 `[N, T, H, W, C]` → `[N, T, H, W]`（squeeze last dim）

### 7.3 delayed_visualizable_image（延迟帧采样）

```python
depth_image = ObservationTermCfg(
    func=instinct_mdp.delayed_visualizable_image,
    params={
        "data_type": "distance_to_image_plane_noised_history",
        "sensor_cfg": SceneEntityCfg("camera"),
        "history_skip_frames": 5,
        "num_output_frames": 8,
        "delayed_frame_ranges": (0, 1),
    },
)
```

需要传感器配置 `data_histories` 来启用历史缓冲：
```python
camera = NoisyGroupedRayCasterCameraCfg(
    ...,
    data_histories={"distance_to_image_plane_noised": 37},  # 历史长度
)
```

---

## 8. 事件随机化

### 8.1 randomize_ray_offsets（光线偏移随机化）

模拟传感器安装误差，直接修改 `sensor.ray_starts` 和 `sensor.ray_directions`。

```python
randomize_ray_offsets = EventTermCfg(
    func=instinct_mdp.randomize_ray_offsets,
    mode="startup",
    params={
        "asset_cfg": SceneEntityCfg("camera"),
        "offset_pose_ranges": {
            "x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01),
            "roll": (-math.radians(2), math.radians(2)),
            "pitch": (-math.radians(10), math.radians(10)),
            "yaw": (-math.radians(2), math.radians(2)),
        },
        "distribution": "uniform",
    },
)
```

**前置条件**: 传感器须暴露 `ray_starts` / `ray_directions` 属性（GroupedRayCaster 满足，mjlab 原生 RayCastSensor 不暴露）。

### 8.2 randomize_camera_offsets（相机偏移随机化）

模拟相机标定误差，修改相机世界位姿偏移。

```python
randomize_camera_offsets = EventTermCfg(
    func=instinct_mdp.randomize_camera_offsets,
    mode="startup",
    params={
        "asset_cfg": SceneEntityCfg("camera"),
        "offset_pose_ranges": {
            "x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01),
            "roll": (-math.radians(2), math.radians(2)),
            "pitch": (-math.radians(5), math.radians(5)),
            "yaw": (-math.radians(2), math.radians(2)),
        },
    },
)
```

**前置条件**: 传感器须暴露 `set_world_poses()` 和 `cfg.offset`（GroupedRayCasterCamera 满足）。

---

## 9. 噪声管线（NoisyCameraMixin）

### 架构

```
NoisyCameraCfgMixin
├── noise_pipeline: dict[str, NoiseCfg]    # 有序噪声处理链
└── data_histories: dict[str, int]         # 历史帧记录

NoisyCameraMixin
├── build_noise_pipeline()                 # 初始化噪声函数实例
├── apply_noise_pipeline(data, env_ids)    # 顺序应用所有噪声
├── build_history_buffers()                # 初始化 AsyncCircularBuffer
└── update_history_buffers(env_ids)        # 追加当前帧到历史
```

### 可用噪声类型（instinct_mjlab.utils.noise）

| 噪声类 | 说明 |
|---|---|
| `DepthNormalizationCfg` | 深度归一化到指定范围 |
| `CropAndResizeCfg` | 裁剪并调整分辨率 |
| `DepthContourNoiseCfg` | 深度图轮廓噪声 |
| `DepthArtifactNoiseCfg` | 深度图伪影 |
| `DepthSkyArtifactNoiseCfg` | 天空区域深度伪影 |
| `GaussianBlurNoiseCfg` | 高斯模糊 |
| `RangeBasedGaussianNoiseCfg` | 基于距离的高斯噪声 |
| `StereoTooCloseNoiseCfg` | 双目近距离无效区噪声 |

### 数据流

```
distance_to_image_plane (raw)
    → noise_pipeline 顺序处理
    → distance_to_image_plane_noised
        → history_buffers 追加
        → distance_to_image_plane_noised_history [N, T, H, W, 1]
```

---

## 10. 完整环境配置示例

### 纯函数风格（推荐）

```python
import math
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import (
    RayCastSensorCfg, GridPatternCfg, PinholeCameraPatternCfg,
    ObjRef, ContactSensorCfg, ContactMatch,
)
from mjlab.managers.observation_manager import ObservationTermCfg, ObservationGroupCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
import mjlab.envs.mdp as mdp
import instinct_mjlab.envs.mdp as instinct_mdp
from instinct_mjlab.sensors.noisy_camera import NoisyGroupedRayCasterCameraCfg
from instinct_mjlab.utils.noise import DepthNormalizationCfg, CropAndResizeCfg


def make_perceptive_env_cfg() -> ManagerBasedRlEnvCfg:
    # --- 传感器 ---
    height_scanner = RayCastSensorCfg(
        name="height_scanner",
        frame=ObjRef(type="body", name="torso_link", entity="robot"),
        pattern=GridPatternCfg(resolution=0.1, size=(1.6, 1.0)),
        ray_alignment="yaw",
        max_distance=100.0,
        include_geom_groups=(0,),
    )

    camera = NoisyGroupedRayCasterCameraCfg(
        name="camera",
        frame=ObjRef(type="body", name="torso_link", entity="robot"),
        pattern=PinholeCameraPatternCfg(fovy=58.0, height=27, width=48),
        offset=NoisyGroupedRayCasterCameraCfg.OffsetCfg(
            pos=(0.048, 0.015, 0.463),
            rot=(0.998, 0.009, 0.407, 0.0),
            convention="world",
        ),
        ray_alignment="base",
        include_geom_groups=(0, 1, 2, 3, 4, 5),
        data_types=["distance_to_image_plane"],
        noise_pipeline={
            "normalize": DepthNormalizationCfg(depth_range=(0.0, 2.0), normalize=True),
            "crop_and_resize": CropAndResizeCfg(crop_region=(2, 2, 2, 2), resize_shape=(18, 32)),
        },
        depth_clipping_behavior="max",
        min_distance=0.05,
    )

    contact_forces = ContactSensorCfg(
        name="contact_forces",
        primary=ContactMatch(mode="body", pattern=".*", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="maxforce",
        history_length=3,
        track_air_time=True,
    )

    # --- 场景 ---
    scene = SceneCfg(
        terrain=TerrainImporterCfg(terrain_type="plane"),
        num_envs=4096,
    )
    scene.sensors = (height_scanner, camera, contact_forces)

    # --- 观测 ---
    observations = {
        "actor": ObservationGroupCfg(
            terms={
                "depth_image": ObservationTermCfg(
                    func=instinct_mdp.visualizable_image,
                    params={"sensor_cfg": SceneEntityCfg("camera"),
                            "data_type": "distance_to_image_plane_noised"},
                ),
                "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
                "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
                "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
                "last_action": ObservationTermCfg(func=mdp.last_action),
            },
            concatenate_terms=False,
            enable_corruption=True,
        ),
        "critic": ObservationGroupCfg(
            terms={
                "height_scan": ObservationTermCfg(
                    func=mdp.height_scan,
                    params={"sensor_name": "height_scanner", "offset": 0.0},
                    clip=[-20.0, 20.0],
                ),
                "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
                "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
                "last_action": ObservationTermCfg(func=mdp.last_action),
            },
            concatenate_terms=False,
            enable_corruption=False,
        ),
    }

    # --- 事件 ---
    events = {
        "randomize_ray_offsets": EventTermCfg(
            func=instinct_mdp.randomize_ray_offsets,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("camera"),
                "offset_pose_ranges": {
                    "x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01),
                    "pitch": (-math.radians(10), math.radians(10)),
                },
            },
        ),
    }

    return ManagerBasedRlEnvCfg(
        scene=scene,
        observations=observations,
        events=events,
        rewards={...},
        terminations={...},
        decimation=4,
        episode_length_s=10.0,
    )
```

---

## 11. 迁移检查清单

- [ ] **传感器类选择**:
  - 高度扫描 → `RayCastSensorCfg`（mjlab 原生）
  - 深度相机 + 噪声 + 历史 → `NoisyGroupedRayCasterCameraCfg`（Instinct 扩展）
  - RGB 相机 → `CameraSensorCfg`（mjlab 原生）

- [ ] **配置字段**:
  - `prim_path` → `frame=ObjRef(type=..., name=..., entity=...)`
  - `attach_yaw_only` → `ray_alignment="yaw"`
  - `mesh_prim_paths` → `include_geom_groups=(0,)`
  - `pattern_cfg` → `pattern`
  - `focal_length + aperture` → `fovy`

- [ ] **数据访问**:
  - `ray_hits_w` → `hit_pos_w`
  - 深度图: `sensor.data.output["distance_to_image_plane"]`
  - 噪声深度图: `sensor.data.output["distance_to_image_plane_noised"]`
  - 历史帧: `sensor.data.output["distance_to_image_plane_noised_history"]`

- [ ] **观测函数参数**:
  - `sensor_cfg=SceneEntityCfg(...)` → `sensor_name="..."`（仅 mjlab 原生 `height_scan`）

- [ ] **场景挂载**:
  - 所有传感器通过 `scene.sensors = (cfg1, cfg2, ...)` 元组挂载

- [ ] **导入路径**:
  ```python
  from mjlab.sensor import RayCastSensorCfg, GridPatternCfg, PinholeCameraPatternCfg, ObjRef
  from instinct_mjlab.sensors.noisy_camera import NoisyGroupedRayCasterCameraCfg
  ```

---

## 12. 已知限制

1. **include_geom_groups**: mjlab 按 MuJoCo geom group 过滤碰撞体。若原 旧框架 依赖精确 mesh 路径匹配（只对地形做光线投射），需确保对应 geom 的 group 编号正确。

2. **GroupedRayCaster 扩展必要性**: mjlab 原生 `RayCastSensor` 不暴露 `ray_starts` / `ray_directions` 缓冲区，无法直接用于 `randomize_ray_offsets`。需要噪声深度图或射线随机化时仍需使用 GroupedRayCaster 系列扩展。

3. **CUDA Graph**: mjlab 光线投射在 CUDA graph 内执行（`mujoco_warp.rays()`），GroupedRayCaster 的 `prepare_rays()` 在 pre-graph 阶段运行。避免在其中做 graph-breaking 操作（如 tensor 值依赖的 Python 分支）。

4. **Perceptive shadowing 注册**: `Instinct_mjlab` 中 perceptive 任务注册当前被注释掉（地形依赖缺失），启用前需确保地形生成器完整。
