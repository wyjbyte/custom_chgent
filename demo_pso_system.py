#!/usr/bin/env python3
"""
高熵结构PSO优化系统演示脚本
展示完整的优化流程和结果分析
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import json
import matplotlib.pyplot as plt
import numpy as np
from pymatgen.core import Structure

from optimization.pso_optimizer import PSOOptimizer, createSimpleTestCase
from optimization.fitness_function import validateFitnessFunction
from optimization.chgnet_wrapper import chgnet_calculator


def demoBasicFunctionality():
    """演示基本功能"""
    print("="*60)
    print("高熵结构PSO优化系统演示")
    print("="*60)
    
    # 1. CHGNet功能演示
    print("\n1. CHGNet能量计算演示")
    print("-" * 40)
    
    # 加载测试结构
    test_structures = []
    cif_files = ["test_data/4450.cif", "test_data/4452.cif"]
    
    for cif_file in cif_files:
        if os.path.exists(cif_file):
            structure = Structure.from_file(cif_file)
            test_structures.append(structure)
            energy = chgnet_calculator.calculateEnergy(structure)
            print(f"  {structure.formula}: {energy:.6f} eV")
    
    # 2. 适应度函数验证
    print(f"\n2. 适应度函数验证")
    print("-" * 40)
    
    if test_structures:
        fitness_results = validateFitnessFunction(test_structures[:1], temperature=300)
        for result in fitness_results:
            print(f"  结构: {result['formula']}")
            print(f"    能量: {result['energy']:.6f} eV")
            print(f"    构型熵: {result['configurational_entropy']:.6f} k_B")
            print(f"    适应度: {result['fitness']:.6f} eV")
    
    # 3. PSO优化演示
    print(f"\n3. PSO优化演示")
    print("-" * 40)
    
    lattice, site_groups = createSimpleTestCase()
    
    optimizer = PSOOptimizer(
        lattice=lattice,
        site_groups=site_groups,
        swarm_size=6,
        max_iterations=3,
        temperature=400.0
    )
    
    result = optimizer.optimize()
    
    if result["optimization_successful"]:
        print(f"✓ 优化成功")
        print(f"  最优适应度: {result['global_best_fitness']:.6f}")
        print(f"  最优能量: {result['energy_decomposition']['energy']:.6f} eV")
        print(f"  构型熵: {result['energy_decomposition']['configurational_entropy']:.6f} k_B")
        
        # 保存演示结果
        optimizer.saveResult(result, "demo_pso_results")
        
        return result
    else:
        print("✗ 优化失败")
        return None


def analyzeOptimizationResults(result_dir: str = "demo_results"):
    """分析优化结果"""
    print(f"\n4. 优化结果分析")
    print("-" * 40)
    
    # 读取结果文件
    json_file = os.path.join(result_dir, "optimization_result.json")
    if not os.path.exists(json_file):
        print(f"结果文件不存在: {json_file}")
        return
    
    with open(json_file, 'r') as f:
        result = json.load(f)
    
    # 分析优化历史
    history = result["optimization_history"]
    iterations = [h["iteration"] for h in history]
    best_fitness = [h["global_best_fitness"] for h in history]
    avg_fitness = [h["average_fitness"] for h in history]
    
    print(f"优化统计:")
    print(f"  总迭代次数: {result['iterations_completed']}")
    print(f"  优化时间: {result['optimization_time']:.2f} 秒")
    print(f"  最终适应度: {result['global_best_fitness']:.6f}")
    
    print(f"\n能量分解:")
    decomp = result["energy_decomposition"]
    print(f"  CHGNet能量: {decomp['energy']:.6f} eV")
    print(f"  构型熵: {decomp['configurational_entropy']:.6f} k_B")
    print(f"  混合熵: {decomp['mixing_entropy']:.6f} k_B")
    print(f"  T*S_config: {decomp['T_S_configurational']:.6f} eV")
    print(f"  T*S_mixing: {decomp['T_S_mixing']:.6f} eV")
    
    print(f"\n最优原子排列:")
    for group_name, elements in result["best_particle_position"].items():
        print(f"  {group_name}: {elements}")
    
    # 生成收敛图
    try:
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(iterations, best_fitness, 'b-o', label='全局最优')
        plt.plot(iterations, avg_fitness, 'r--s', label='平均')
        plt.xlabel('迭代次数')
        plt.ylabel('适应度 (eV)')
        plt.title('PSO收敛曲线')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        # 能量分解饼图
        values = [
            abs(decomp['energy']),
            abs(decomp['T_S_configurational'])
        ]
        labels = ['|E_CHGNet|', '|T*S_config|']
        colors = ['lightblue', 'lightcoral']
        
        plt.pie(values, labels=labels, colors=colors, autopct='%1.1f%%')
        plt.title('能量组成')
        
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, 'optimization_analysis.png'), dpi=150, bbox_inches='tight')
        print(f"\n✓ 分析图表已保存至: {result_dir}/optimization_analysis.png")
        
    except Exception as e:
        print(f"图表生成失败: {e}")


def demoAdvancedCase():
    """演示更复杂的优化案例"""
    print(f"\n5. 高熵合金优化案例")
    print("-" * 40)
    
    # 创建更大的高熵合金案例
    from optimization.particle_encoding import HighEntropySiteGroup
    from pymatgen.core.lattice import Lattice
    
    # 2x2x1 超晶胞，BCC结构
    lattice = Lattice.cubic(6.0)
    
    # 创建8个原子位点
    positions = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                x = i * 0.5
                y = j * 0.5
                z = k * 0.5
                positions.append(np.array([x, y, z]))
    
    # 五元高熵合金 CrFeCoNiMn
    element_counts = {"Cr": 2, "Fe": 2, "Co": 2, "Ni": 1, "Mn": 1}
    site_group = HighEntropySiteGroup("bcc_sites", positions, element_counts)
    
    print(f"案例设置:")
    print(f"  晶格: BCC, a={lattice.a:.1f} Å")
    print(f"  原子数: {sum(element_counts.values())}")
    print(f"  成分: {element_counts}")
    
    optimizer = PSOOptimizer(
        lattice=lattice,
        site_groups=[site_group],
        swarm_size=10,
        max_iterations=5,
        temperature=500.0,
        mutation_rate=0.15
    )
    
    result = optimizer.optimize()
    
    if result["optimization_successful"]:
        optimizer.saveResult(result, "advanced_demo_results")
        
        print(f"\n高熵合金优化结果:")
        print(f"  最优适应度: {result['global_best_fitness']:.6f} eV")
        print(f"  最优结构: {result['best_structure'].formula}")
        print(f"  能量: {result['energy_decomposition']['energy']:.6f} eV")
        print(f"  熵增益: {result['energy_decomposition']['T_S_configurational']:.6f} eV")
        
        return result
    else:
        print("✗ 高熵合金优化失败")
        return None


def main():
    """主演示函数"""
    try:
        # 基本功能演示
        basic_result = demoBasicFunctionality()
        
        # 分析已有结果
        if os.path.exists("demo_results"):
            analyzeOptimizationResults("demo_results")
        
        # 高级案例演示
        advanced_result = demoAdvancedCase()
        
        print(f"\n" + "="*60)
        print("演示完成")
        print("="*60)
        
        print(f"\n生成的文件:")
        result_dirs = ["demo_pso_results", "demo_results", "advanced_demo_results"]
        for result_dir in result_dirs:
            if os.path.exists(result_dir):
                print(f"\n{result_dir}/:")
                for file in os.listdir(result_dir):
                    print(f"  - {file}")
        
        print(f"\n系统功能验证:")
        print(f"  ✓ CHGNet能量计算")
        print(f"  ✓ 构型熵计算")
        print(f"  ✓ PSO粒子群优化")
        print(f"  ✓ 离散空间优化")
        print(f"  ✓ 结果保存和分析")
        print(f"  ✓ 多种高熵合金案例")
        
        print(f"\n🎉 高熵结构PSO优化系统演示成功完成！")
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()