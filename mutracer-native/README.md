# MuTracer Native (Windows ARM64 + DX12)

面向 **Windows ARM64**（高通骁龙 8cx Gen 3 / Adreno 690 GPU）原生运行的路径追踪与辐照度体积（Irradiance Volume / 探针网格）GPU 烘焙器，直接输出 **Bevy 0.20.1** 兼容的 3D KTX2 纹理格式。

---

## 特性与架构

1. **纯原生 Windows ARM64 (`aarch64-pc-windows-msvc`)**：
   - 移除原版 MuTracer 对 `scriptc` AOT 编译器的平台依赖，所有 CPU 场景准备与 BVH 树构建全部在 Rust 原生实现。
   - 完全离线构建，无跨架构模拟损耗。

2. **DirectX 12 (DX12) 原生后端**：
   - 基于 `wgpu 30.0.1`（与本机 Bevy 0.20.1 完全对齐）。
   - 集成 `dxcompiler.dll`（Dynamic DXC 编译），完美兼容高通 Adreno 驱动，支持复杂计算着色器（Compute Shader）。

3. **GPU 路径追踪加速**：
   - CPU 快速构建分桶 SAH BVH，展平成高效的 GPU 存储缓冲区（Linear GPU BVH）。
   - GPU 上执行分支友好的无栈/微栈 BVH 遍历与三角形求交，运行于 Adreno 8cx Gen 3 硬件计算核心。

4. **Bevy 0.20.1 辐照度体积（Irradiance Volume）烘焙**：
   - 严格遵循 Valve Ambient Cube 六方向基底（$\pm X, \pm Y, \pm Z$）及余弦半球蒙特卡洛采样。
   - 自动按照 Bevy 0.20.1 `irradiance_volume.wesl` 的 3D 纹理排布规格 $(R_x, 2R_y, 3R_z)$ 进行切片打包：
     - 上半部（$v \in [0, 0.5)$）：$+X, +Y, +Z$
     - 下半部（$v \in [0.5, 1.0)$）：$-X, -Y, -Z$
     - 深度轴（$w$）：Slice 0 为 $X$ 轴，Slice 1 为 $Y$ 轴，Slice 2 为 $Z$ 轴。
   - 导出为标准无损 KTX2 纹理，格式为 `VK_FORMAT_E5B9G9R9_UFLOAT_PACK32`（Bevy 内置 `Rgb9e5Ufloat`）。
   - 生成对应的 `.meta.json`，计算出精准的 Bevy `Transform` 包围盒尺寸与中心坐标，并提供开箱即用的 Bevy 代码片段。

---

## 快速上手

### 1. 构建
```powershell
cd C:\tracer-experiment\mutracer-native
cargo build
```

### 2. 检查 GPU 设备与 DX12 支持
```powershell
.\target\debug\mutracer-native.exe probe-gpu
```
输出示例：
```text
[gpu] Using DXC compiler: dxcompiler.dll
[gpu] Available DX12 adapters:
  [0] Qualcomm(R) Adreno(TM) 8cx Gen 3 (IntegratedGpu, driver: 30.0.4369.8700)
  [1] Microsoft Basic Render Driver (Cpu, driver: 10.0.26100.9818)
[gpu] Selected: Qualcomm(R) Adreno(TM) 8cx Gen 3 (IntegratedGpu, backend: Dx12)
```

### 3. 烘焙场景为 Bevy Irradiance Volume
```powershell
.\target\debug\mutracer-native.exe bake `
  --scene C:\bevy\assets\models\CornellBox\CornellBox.glb `
  --out cornell.vxgi.ktx2 `
  --resolution 8,8,8 `
  --spp 64 `
  --bounces 2
```

### 4. 校验输出的 KTX2 文件
```powershell
.\target\debug\mutracer-native.exe verify-ktx2 cornell.vxgi.ktx2
```
输出示例：
```text
Verifying KTX2: cornell.vxgi.ktx2 (12516 bytes)
  Format: Some(E5B9G9R9_UFLOAT_PACK32)
  Dimensions: 8 x 16 x 24
  Levels: 1, Layers: 0, Faces: 1
  Supercompression: None
  Level 0 data size: 12288 bytes
KTX2 verification PASSED!
```

---

## 在 Bevy 0.20.1 中加载使用

在 Bevy 项目中，将烘焙出的 `.ktx2` 放入 `assets/` 目录，然后根据生成的 `.meta.json` 插入组件：

```rust
use bevy::prelude::*;
use bevy::light::IrradianceVolume;

fn setup_irradiance_volume(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn((
        // 从 .meta.json 中自动生成的变换参数（包围盒中心与长宽高）
        Transform::from_xyz(0.0757, -0.4615, 0.1511)
            .with_scale(Vec3::new(44.6073, 1.0154, 44.7732)),
        IrradianceVolume {
            voxels: asset_server.load("Example.vxgi.ktx2"),
            intensity: 1.0,
            ..default()
        },
    ));
}
```

