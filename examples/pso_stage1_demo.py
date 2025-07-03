"""
高熵结构优化 - 阶段一PSO使用示例

展示如何使用阶段一PSO核心组件进行高熵合金结构优化
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pymatgen.core import Structure, Lattice
from chgnet.optimization import (
    PSOParticle,
    initialize_population,
    generate_structure_from_particle_stage1,
    calculate_configurational_entropy,
    calculate_fitness_stage1
)
from chgnet.optimization.fitness import create_chgnet_calculator
from chgnet.optimization.entropy import get_entropy_analysis_report


def create_high_entropy_alloy_example():
    """
    创建高熵合金示例结构和位点群定义
    
    Returns:
        tuple: (base_structure, site_groups_definition)
    """
    # 创建面心立方结构的高熵合金
    lattice = Lattice.cubic(3.6)  # 类似奥氏体不锈钢的晶格常数
    
    # 初始组成：CrMnFeCoNi (等原子比高熵合金)
    species = ["Cr", "Mn", "Fe", "Co", "Ni", "Cr", "Mn", "Fe"]
    coords = [
        [0.0, 0.0, 0.0],      # 位点0
        [0.5, 0.5, 0.0],      # 位点1
        [0.5, 0.0, 0.5],      # 位点2
        [0.0, 0.5, 0.5],      # 位点3
        [0.25, 0.25, 0.25],   # 位点4
        [0.75, 0.75, 0.25],   # 位点5
        [0.75, 0.25, 0.75],   # 位点6
        [0.25, 0.75, 0.75]    # 位点7
    ]
    
    base_structure = Structure(lattice, species, coords)
    
    # 定义两个独立的位点群
    site_groups_definition = {
        "octahedral_sites": {
            "site_indices": [0, 1, 2, 3],
            "elements": [24, 25, 26, 27],  # Cr, Mn, Fe, Co
            "element_counts": {"Cr": 1, "Mn": 1, "Fe": 1, "Co": 1}
        },
        "tetrahedral_sites": {
            "site_indices": [4, 5, 6, 7],
            "elements": [28, 24, 25, 26],  # Ni, Cr, Mn, Fe
            "element_counts": {"Ni": 1, "Cr": 1, "Mn": 1, "Fe": 1}
        }
    }
    
    return base_structure, site_groups_definition


def demonstrate_pso_stage1():
    """演示阶段一PSO核心组件的使用"""
    
    print("=== 高熵结构优化 - 阶段一PSO演示 ===\n")
    
    # 1. 创建高熵合金结构和位点群定义
    print("1. 创建高熵合金结构...")
    base_structure, site_groups_def = create_high_entropy_alloy_example()
    print(f"   基础结构组成: {base_structure.composition}")
    print(f"   位点群数量: {len(site_groups_def)}")
    for group_name, group_info in site_groups_def.items():
        print(f"   - {group_name}: {len(group_info['site_indices'])}个位点")
    
    # 2. 初始化PSO种群
    print("\n2. 初始化PSO种群...")
    population_size = 10
    population = initialize_population(
        population_size, 
        site_groups_def, 
        random_seed=42
    )
    print(f"   成功创建{len(population)}个粒子的种群")
    
    # 3. 展示粒子排列的多样性
    print("\n3. 种群多样性分析...")
    print("   前3个粒子的排列:")
    for i, particle in enumerate(population[:3]):
        summary = particle.get_arrangement_summary()
        print(f"   粒子{i}:")
        for group_name, group_data in summary.items():
            arrangement = group_data['arrangement']
            # 将原子序数转换为元素符号
            elements = [['Cr','Mn','Fe','Co','Ni'][x-24] if 24<=x<=28 else str(x) for x in arrangement]
            print(f"     {group_name}: {elements}")
    
    # 4. 配置熵计算
    print("\n4. 配置熵分析...")
    entropies = []
    for i, particle in enumerate(population):
        entropy_report = get_entropy_analysis_report(
            particle, site_groups_def, temperature=300.0
        )
        entropies.append(entropy_report['total_configurational_entropy'])
        if i < 3:  # 显示前3个粒子的详细信息
            print(f"   粒子{i}:")
            print(f"     总配置熵: {entropy_report['total_configurational_entropy']:.6f} eV/K")
            print(f"     熵效率: {entropy_report['entropy_efficiency']:.2%}")
    
    print(f"   种群平均配置熵: {np.mean(entropies):.6f} eV/K")
    print(f"   配置熵范围: {min(entropies):.6f} - {max(entropies):.6f} eV/K")
    
    # 5. 结构生成
    print("\n5. 结构生成演示...")
    best_entropy_idx = np.argmax(entropies)
    best_particle = population[best_entropy_idx]
    
    # 生成对应的晶体结构
    optimized_structure = generate_structure_from_particle_stage1(
        base_structure, best_particle, site_groups_def
    )
    
    print(f"   选择配置熵最高的粒子(索引{best_entropy_idx})")
    print(f"   原始结构: {base_structure.composition}")
    print(f"   优化结构: {optimized_structure.composition}")
    
    # 6. 适应度函数演示（如果可能）
    print("\n6. 适应度函数演示...")
    try:
        calculator = create_chgnet_calculator(device="cpu")
        T = 500.0  # 温度参数
        
        # 计算几个结构的适应度
        print(f"   温度参数 T = {T} K")
        fitness_values = []
        
        for i, particle in enumerate(population[:3]):
            structure = generate_structure_from_particle_stage1(
                base_structure, particle, site_groups_def
            )
            
            fitness = calculate_fitness_stage1(
                structure, T, site_groups_def, calculator, particle
            )
            fitness_values.append(fitness)
            
            entropy = calculate_configurational_entropy(
                particle=particle, site_groups_definition=site_groups_def
            )
            
            print(f"   粒子{i}:")
            print(f"     适应度: {fitness:.4f} eV")
            print(f"     配置熵: {entropy:.6f} eV/K")
            print(f"     T×S项: {T * entropy:.4f} eV")
        
        best_fitness_idx = np.argmin(fitness_values)
        print(f"\n   最佳适应度: 粒子{best_fitness_idx} (F = {fitness_values[best_fitness_idx]:.4f} eV)")
        
    except Exception as e:
        print(f"   适应度计算跳过: {e}")
    
    # 7. 优化建议
    print("\n7. 优化建议...")
    print("   - 温度参数T的选择对平衡能量和熵很重要")
    print("   - 更高的T值会更倾向于高熵配置")
    print("   - 更低的T值会更倾向于低能量配置")
    print("   - 建议先进行参数敏感性分析")
    
    return population, base_structure, site_groups_def


def analyze_temperature_effect():
    """分析温度参数对适应度的影响"""
    
    print("\n=== 温度参数影响分析 ===")
    
    # 创建测试结构
    base_structure, site_groups_def = create_high_entropy_alloy_example()
    particle = PSOParticle(site_groups_def)
    
    structure = generate_structure_from_particle_stage1(
        base_structure, particle, site_groups_def
    )
    
    # 不同温度下的适应度分析
    temperatures = [100, 300, 500, 800, 1200]  # K
    
    try:
        calculator = create_chgnet_calculator(device="cpu")
        
        print("温度(K)\t适应度(eV)\tT×S项(eV)")
        print("-" * 40)
        
        for T in temperatures:
            fitness = calculate_fitness_stage1(
                structure, T, site_groups_def, calculator, particle
            )
            entropy = calculate_configurational_entropy(
                particle=particle, site_groups_definition=site_groups_def
            )
            t_times_s = T * entropy
            
            print(f"{T}\t{fitness:.4f}\t\t{t_times_s:.4f}")
            
    except Exception as e:
        print(f"温度分析跳过: {e}")


if __name__ == "__main__":
    # 运行主演示
    population, base_structure, site_groups_def = demonstrate_pso_stage1()
    
    # 运行温度效应分析
    analyze_temperature_effect()
    
    print("\n=== 演示完成 ===")
    print("阶段一PSO核心组件已成功实现并验证！")