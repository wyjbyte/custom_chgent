# 高熵结构优化 - 阶段一PSO核心组件

本项目实现了高熵合金结构优化的阶段一粒子群优化（PSO）核心组件。目标函数为 **F = E_CHGNet - T * S_config**，其中 E_CHGNet 是 CHGNet 预测的能量，S_config 是配置熵，T 是温度参数。

## 功能特性

### 1. PSO粒子表示设计
- **粒子数据结构**: `PSOParticle` 类完整描述晶体中所有原子的排列方式
- **位点群编码**: 粒子编码可交换位点群内的原子排列，支持多个独立位点群
- **约束满足**: 严格维持每个位点群内元素种类和数量不变

### 2. 种群初始化
- **多样性策略**: 使用三种不同的随机化策略生成多样化的初始种群
- **约束保证**: 确保每个粒子严格遵守位点群内元素种类和数量约束
- **多位点群支持**: 处理多个独立位点群的初始化

### 3. 结构生成
- **结构转换**: `generate_structure_from_particle_stage1` 函数将 PSO 粒子转换为 pymatgen.Structure
- **一致性验证**: 保持晶胞参数和位点坐标，只改变原子种类
- **格式兼容**: 支持 pymatgen 和 ASE 格式之间的转换

### 4. 配置熵计算
- **混合熵公式**: 使用统计力学的混合熵公式计算 S_config
- **位点群分离**: 对每个位点群分别计算配置熵
- **温度效应**: 考虑温度对配置熵的影响

### 5. 适应度函数
- **能量计算**: 调用 CHGNet 获取结构能量 E_CHGNet
- **熵计算**: 调用配置熵计算模块获取 S_config
- **单位统一**: 处理能量和熵的量纲统一问题（eV 和 eV/K）

## 模块结构

```
chgnet/optimization/
├── __init__.py           # 模块初始化
├── particle.py           # PSO粒子表示
├── population.py         # 种群初始化
├── structure_ops.py      # 结构生成操作
├── entropy.py           # 配置熵计算
└── fitness.py           # 适应度函数
```

## 快速开始

### 基本使用示例

```python
from chgnet.optimization import (
    PSOParticle,
    initialize_population,
    generate_structure_from_particle_stage1,
    calculate_fitness_stage1
)
from chgnet.optimization.fitness import create_chgnet_calculator

# 1. 定义位点群
site_groups_definition = {
    "group_A": {
        "site_indices": [0, 1, 2, 3],
        "elements": [6, 6, 14, 14],  # 2个C，2个Si
        "element_counts": {"C": 2, "Si": 2}
    }
}

# 2. 初始化种群
population = initialize_population(10, site_groups_definition)

# 3. 创建 CHGNet 计算器
calculator = create_chgnet_calculator()

# 4. 评估适应度
for particle in population:
    structure = generate_structure_from_particle_stage1(
        base_structure, particle, site_groups_definition
    )
    fitness = calculate_fitness_stage1(
        structure, T=500.0, site_groups_definition, calculator, particle
    )
    print(f"适应度: {fitness:.4f} eV")
```

### 高熵合金示例

运行完整的高熵合金优化演示：

```bash
python examples/pso_stage1_demo.py
```

## 技术细节

### 粒子编码方式
- 使用字典结构存储多个位点群的排列
- 每个位点群用 numpy 数组表示原子排列
- 数组索引对应位点索引，数组值对应原子种类（原子序数）

### 约束满足机制
- 严格验证位点群内元素种类和数量
- 原子交换只在同类位点群内发生
- 提供约束验证函数确保解的可行性

### 能量和熵的单位处理
- CHGNet 输出能量单位：eV
- 配置熵单位：k_B (eV/K)
- 通过温度参数 T 实现单位统一：F = E - T*S (eV)

### 数值稳定性
- 处理对数计算中的零值情况
- 提供适应度计算失败的异常处理
- 验证计算结果的合理性

## 测试验证

运行完整测试套件：

```bash
python tests/test_pso_stage1.py
```

测试包括：
- ✅ PSO粒子创建和操作
- ✅ 种群初始化和约束验证
- ✅ 结构生成和一致性检查
- ✅ 配置熵计算
- ✅ CHGNet集成和适应度计算
- ✅ 完整工作流程验证

## 优化建议

### 温度参数选择
- **高温度** (T > 800K): 更倾向于高熵配置，探索性强
- **中温度** (T = 300-800K): 平衡能量和熵，适合大多数情况
- **低温度** (T < 300K): 更倾向于低能量配置，开发性强

### 性能优化
- 使用 GPU 加速 CHGNet 计算
- 实现并行适应度评估
- 缓存重复结构的计算结果

### 扩展方向
- 实现完整的 PSO 主循环
- 添加自适应温度参数
- 支持更复杂的约束条件

## 依赖项

- numpy >= 1.20.0
- torch >= 1.9.0
- pymatgen >= 2022.0.0
- ase >= 3.22.0
- chgnet (本项目)

## 贡献指南

1. 确保所有测试通过
2. 添加适当的中文注释
3. 遵循现有代码风格
4. 更新相关文档

## 许可证

本项目遵循 CHGNet 的许可证条款。

---

*实现日期: 2025年*  
*版本: 阶段一核心组件*