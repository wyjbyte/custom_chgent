# 高熵结构PSO优化系统 - 合并版本

本项目成功合并了三个相关的分支，实现了完整的高熵结构PSO优化系统：

1. **PR #2**: 高熵结构优化框架 - 原子半径表处理、位点群定义、构型熵计算
2. **PR #3**: PSO核心组件 - PSO粒子表示、种群初始化、结构生成、适应度函数
3. **PR #4**: 完整PSO优化系统 - 包含CHGNet集成的完整优化流程

## 系统架构

### 1. 基础框架 (chgnet.model.dynamics)
- `HighEntropyOptimizer`: 高熵结构优化器基础类
- 原子半径表加载与解析
- 位点群识别和元素统计
- 基础构型熵计算

### 2. PSO核心组件 (chgnet.optimization)
- `PSOParticle`: PSO粒子表示和操作
- `initialize_population`: 种群初始化
- `generate_structure_from_particle_stage1`: 结构生成
- `calculate_configurational_entropy`: 配置熵计算
- `calculate_fitness_stage1`: 适应度函数

### 3. 完整优化系统 (optimization/)
- `PSOOptimizer`: 完整的PSO优化器
- `CHGNetEnergyCalculator`: CHGNet计算器封装
- `Particle`, `ParticleSwarm`: 高级粒子编码系统
- 完整的结果保存和分析功能

## 主要功能

### 目标函数
- **F = E_CHGNet - T × S_config**
- E_CHGNet: 通过CHGNet神经网络势计算的能量
- S_config: 基于位点群的构型熵
- T: 温度参数，控制能量和熵的平衡

### 核心特性
- ✅ 真实CHGNet神经网络势能量计算
- ✅ 离散空间的swap-based PSO算法
- ✅ 元素数量守恒约束自动维护
- ✅ 构型熵驱动的高熵合金优化
- ✅ 模块化设计，易于扩展
- ✅ 详细的能量分解和调试功能

## 使用方法

### 基础用法 (chgnet.optimization)
```python
from chgnet.optimization import (
    PSOParticle, 
    initialize_population,
    generate_structure_from_particle_stage1,
    calculate_configurational_entropy
)

# 定义位点群
site_groups_definition = {
    "group_A": {
        "site_indices": [0, 1, 2, 3],
        "elements": [24, 25, 26, 27],  # Cr, Mn, Fe, Co
        "element_counts": {"Cr": 1, "Mn": 1, "Fe": 1, "Co": 1}
    }
}

# 初始化种群
population = initialize_population(10, site_groups_definition)

# 计算配置熵
for particle in population:
    entropy = calculate_configurational_entropy(
        particle=particle, 
        site_groups_definition=site_groups_definition
    )
    print(f"配置熵: {entropy:.6f} eV/K")
```

### 完整优化系统 (optimization/)
```python
from optimization.pso_optimizer import PSOOptimizer, createSimpleTestCase

# 创建测试案例
lattice, site_groups = createSimpleTestCase()

# 创建优化器
optimizer = PSOOptimizer(
    lattice=lattice,
    site_groups=site_groups,
    swarm_size=20,
    max_iterations=100,
    temperature=500.0
)

# 执行优化
result = optimizer.optimize()

# 保存结果
optimizer.saveResult(result, "optimization_results")
```

### 高熵结构优化基础功能
```python
from chgnet.model.dynamics import HighEntropyOptimizer

# 创建高熵优化器
optimizer = HighEntropyOptimizer()

# 识别位点群
structure_info = {'elements': ['Cr', 'Mn', 'Fe', 'Co', 'Ni']}
site_groups = optimizer.identify_site_groups(structure_info)

# 计算构型熵
entropy = optimizer.calculate_configurational_entropy(structure_info)
print(f"构型熵: {entropy:.6f} eV/K")
```

## 文件结构

```
custom_chgnet/
├── chgnet/
│   ├── model/
│   │   ├── dynamics.py          # HighEntropyOptimizer基础框架
│   │   └── __init__.py          # 导出HighEntropyOptimizer
│   └── optimization/            # PSO核心组件模块
│       ├── __init__.py          # 模块导出
│       ├── particle.py          # PSO粒子表示
│       ├── entropy.py           # 配置熵计算
│       ├── fitness.py           # 适应度函数
│       └── structure_ops.py     # 结构生成操作
└── optimization/                # 完整PSO优化系统
    ├── __init__.py              # 系统导出
    ├── chgnet_wrapper.py        # CHGNet计算器封装
    └── pso_optimizer.py         # 完整PSO优化器
```

## 技术特点

### 多层次架构
1. **基础层**: chgnet.model.dynamics.HighEntropyOptimizer
2. **核心层**: chgnet.optimization 模块
3. **应用层**: optimization/ 完整系统

### 约束处理
- 自动维护元素数量守恒
- 支持多个独立位点群
- 基于交换的离散空间搜索

### 能量计算
- 集成CHGNet神经网络势
- 支持CPU/GPU计算
- 自动异常处理和错误恢复

### 熵计算
- 基于统计力学的混合熵公式
- 位点群独立计算
- 温度效应考虑

## 扩展性

系统采用模块化设计，支持以下扩展：

1. **多目标优化**: 添加其他目标函数
2. **约束优化**: 添加额外约束条件
3. **并行计算**: 支持多进程计算
4. **可视化**: 结果可视化和分析工具

## 依赖项

- numpy >= 1.20.0
- torch >= 1.9.0
- pymatgen >= 2022.0.0
- ase >= 3.22.0
- chgnet (本项目)

## 合并说明

这个版本成功合并了三个分支的功能：
- 保留了所有核心功能
- 避免了功能重复
- 提供了多个使用层次
- 确保了向后兼容性

用户可以根据需要选择使用基础框架、核心组件或完整系统。