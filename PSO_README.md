# 高熵结构PSO优化系统

## 概述

本项目实现了基于粒子群优化算法(PSO)和CHGNet神经网络势的高熵结构优化系统。系统能够在离散原子排列空间中搜索具有最低 `E_CHGNet - T*S_config` 的原子构型。

## 主要功能

### 1. CHGNet计算器封装
- 简化的能量计算接口
- 支持pymatgen Structure和ASE Atoms
- 自动异常处理和错误恢复

### 2. 粒子编码系统
- 基于位点群的原子排列表示
- 元素数量守恒约束
- 基于交换操作的离散PSO更新

### 3. 适应度函数
- 目标函数：`F = E_CHGNet - T*S_config`
- 构型熵和混合熵计算
- 详细的能量分解分析

### 4. PSO优化器
- 完整的粒子群优化循环
- 收敛控制和早停机制
- 实时进度监控

## 安装和使用

### 环境要求
```bash
pip install numpy pymatgen torch ase matplotlib
pip install nvidia-ml-py  # 用于GPU支持
```

### 快速开始

#### 1. 运行测试
```bash
python test_pso_system.py
```

#### 2. 基本优化
```bash
# 简单测试案例
python run_pso_optimization.py --case simple --swarm-size 10 --max-iterations 20

# 高熵合金FCC结构
python run_pso_optimization.py --case heafcc --swarm-size 20 --max-iterations 50 --temperature 600

# 从CIF文件加载
python run_pso_optimization.py --case cif --cif test_data/4450.cif --swarm-size 15
```

#### 3. 完整演示
```bash
python demo_pso_system.py
```

## 目录结构

```
optimization/
├── __init__.py                 # 模块初始化
├── chgnet_wrapper.py          # CHGNet计算器封装
├── particle_encoding.py       # 粒子编码系统
├── fitness_function.py        # 适应度函数实现
└── pso_optimizer.py           # PSO优化器主逻辑

test_pso_system.py             # 系统测试脚本
run_pso_optimization.py        # 主优化程序
demo_pso_system.py             # 完整演示脚本

test_data/                     # 测试数据
├── 4450.cif
└── 4452.cif
```

## 使用示例

### Python API使用

```python
from optimization.pso_optimizer import PSOOptimizer, createSimpleTestCase
from optimization.particle_encoding import HighEntropySiteGroup
from pymatgen.core.lattice import Lattice
import numpy as np

# 创建优化案例
lattice = Lattice.cubic(4.0)
positions = [np.array([0, 0, 0]), np.array([0.5, 0.5, 0.5])]
element_counts = {"Fe": 1, "Ni": 1}
site_group = HighEntropySiteGroup("test_sites", positions, element_counts)

# 创建优化器
optimizer = PSOOptimizer(
    lattice=lattice,
    site_groups=[site_group],
    swarm_size=20,
    max_iterations=100,
    temperature=300.0
)

# 执行优化
result = optimizer.optimize()

# 保存结果
optimizer.saveResult(result, "results")
```

### 命令行使用

```bash
# 基本参数
python run_pso_optimization.py \
    --case heafcc \
    --swarm-size 30 \
    --max-iterations 100 \
    --temperature 500 \
    --output-dir my_results

# PSO参数调整
python run_pso_optimization.py \
    --case large \
    --swarm-size 25 \
    --max-iterations 80 \
    --inertia 0.7 \
    --cognitive 1.5 \
    --social 1.5 \
    --mutation 0.05 \
    --patience 15
```

## 结果分析

优化完成后会生成以下文件：

1. `best_structure.cif` - 最优晶体结构
2. `optimization_result.json` - 详细优化结果
3. `best_particle.json` - 最优粒子编码

结果包含：
- 最优适应度值
- CHGNet能量
- 构型熵
- 优化历史
- 能量分解

## 技术特点

- **真实物理模型**：基于CHGNet神经网络势的准确能量计算
- **离散优化**：适用于原子排列的swap-based PSO算法
- **熵驱动优化**：考虑构型熵对高熵合金稳定性的贡献
- **约束保持**：自动维护元素数量守恒
- **模块化设计**：易于扩展和定制

## 参数说明

### PSO参数
- `swarm_size`: 粒子群大小 (建议10-50)
- `max_iterations`: 最大迭代次数
- `inertia_weight`: 惯性权重 (0.1-0.9)
- `cognitive_factor`: 认知因子 (1.0-2.0)
- `social_factor`: 社会因子 (1.0-2.0)
- `mutation_rate`: 变异率 (0.01-0.2)

### 物理参数
- `temperature`: 优化温度 (K)
- `lattice`: 晶格参数
- `site_groups`: 位点群定义

## 故障排除

### 常见问题

1. **CHGNet导入失败**
   ```bash
   pip install torch pymatgen
   ```

2. **GPU内存不足**
   - 使用CPU模式
   - 减少粒子群大小

3. **优化不收敛**
   - 增加迭代次数
   - 调整PSO参数
   - 增加变异率

4. **能量计算失败**
   - 检查结构合理性
   - 验证元素类型

## 扩展功能

系统支持以下扩展：

1. **多目标优化**：添加其他目标函数
2. **约束优化**：添加额外约束条件
3. **并行计算**：支持多进程计算
4. **可视化**：结果可视化和分析工具

## 参考文献

- CHGNet: 基于晶体哈密顿图神经网络的通用材料势
- PSO: 粒子群优化算法原理
- 高熵合金: 构型熵在材料设计中的应用

---

**开发者**: yjbyte  
**版本**: 1.0.0  
**许可**: 按照原CHGNet项目许可  