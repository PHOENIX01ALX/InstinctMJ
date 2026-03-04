# Parkour 任务可视化说明

## 概述

Parkour 任务的命令可视化已从 旧框架 的 `VisualizationMarkers`（空实现）迁移到 mjlab 的原生 MuJoCo 几何体可视化系统。

## 可视化内容

当 `debug_vis=True` 时，会显示以下内容：

### 1. 目标位置（红色圆柱）
- 颜色：红色半透明 (1.0, 0.0, 0.0, 0.6)
- 形状：圆柱体
- 半径：`target_dis_threshold` 配置值（默认 0.4m）
- 高度：0.1m
- 位置：当前环境的目标位置 `pos_command_w`

### 2. 命令速度箭头（绿色）
- 颜色：绿色 (0.0, 1.0, 0.0, 0.7)
- 起点：机器人基座位置 + 0.5m 高度偏移
- 方向：命令速度方向（从机器人体坐标系转换到世界坐标系）
- 长度：速度大小 × 0.5 缩放因子

### 3. 实际速度箭头（蓝色）
- 颜色：蓝色 (0.0, 0.5, 1.0, 0.7)
- 起点：机器人基座位置 + 0.5m 高度偏移
- 方向：实际速度方向（从机器人体坐标系转换到世界坐标系）
- 长度：速度大小 × 0.5 缩放因子

### 4. 所有可行目标点（蓝色圆柱，可选）
- 仅当 `patch_vis=True` 时显示
- 颜色：蓝色半透明 (0.0, 0.0, 1.0, 0.3)
- 形状：圆柱体
- 半径：`target_dis_threshold × 0.8`
- 高度：0.05m
- 位置：当前地形类型的所有 flat patches

## 配置

在 `parkour_env_cfg.py` 中：

```python
cfg.commands = {
    "base_velocity": PoseVelocityCommandCfg(
        entity_name="robot",
        debug_vis=True,  # 启用可视化
        patch_vis=False,  # 是否显示所有 flat patches（可能影响性能）
        target_dis_threshold=0.4,  # 目标圆柱半径
        ...
    ),
}
```

## 实现细节

### 与 旧框架 的差异

| 特性 | 旧框架 | mjlab (新实现) |
|------|-----------|----------------|
| 可视化系统 | `VisualizationMarkers` (USD) | MuJoCo 原生几何体 |
| 目标位置 | 红色圆柱 | 红色圆柱（相同） |
| 命令速度 | 绿色箭头 | 绿色箭头（相同） |
| 实际速度 | 蓝色箭头 | 蓝色箭头（相同） |
| Flat patches | 蓝色圆柱 | 蓝色圆柱（相同） |
| 性能 | 依赖 USD 场景图 | 直接使用 MuJoCo API，更高效 |

### 技术实现

使用 `DebugVisualizer` 接口：
- `visualizer.add_cylinder()` - 绘制目标位置和 flat patches
- `visualizer.add_arrow()` - 绘制速度箭头

坐标系转换：
- 速度命令从机器人体坐标系转换到世界坐标系
- 使用简化的 yaw 旋转（仅考虑水平面旋转）

## 性能考虑

- 默认只可视化当前查看的环境（`env_idx`）
- 可通过 viewer 设置 `show_all_envs=True` 显示所有环境
- `patch_vis=True` 会显示大量圆柱体（每个地形类型 50 个 patches），可能影响性能

## 调试建议

1. 如果看不到可视化：
   - 确认 `debug_vis=True`
   - 检查机器人是否已初始化（位置不在原点）
   - 确认 viewer 正在显示正确的环境索引

2. 如果箭头方向不对：
   - 检查速度命令是否为零
   - 验证坐标系转换（体坐标系 → 世界坐标系）

3. 如果目标位置不显示：
   - 确认 terrain 配置了 `flat_patch_sampling` 中的 `"target"` 键
   - 检查 `valid_targets` 张量是否有效

## 示例

运行 parkour 任务并启用可视化：

```bash
python scripts/instinct_rl/play.py --task Instinct-Parkour-G1-v0
```

在 viewer 中应该能看到：
- 红色圆柱标记目标位置
- 绿色箭头显示命令速度
- 蓝色箭头显示实际速度
- （可选）蓝色圆柱显示所有可行目标点
