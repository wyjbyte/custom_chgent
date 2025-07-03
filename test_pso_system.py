#!/usr/bin/env python3
"""
PSO优化系统测试脚本
验证CHGNet集成和PSO算法功能
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
from pymatgen.core import Structure
from pymatgen.core.lattice import Lattice

from optimization.chgnet_wrapper import chgnet_calculator
from optimization.pso_optimizer import PSOOptimizer, createSimpleTestCase
from optimization.fitness_function import validateFitnessFunction


def test_chgnet_wrapper():
    """测试CHGNet封装器"""
    print("="*50)
    print("测试CHGNet封装器")
    print("="*50)
    
    # 检查CHGNet可用性
    print("检查CHGNet可用性...", end=" ")
    if chgnet_calculator.isAvailable():
        print("✓ 可用")
    else:
        print("✗ 不可用")
        return False
    
    # 测试从测试数据计算能量
    try:
        print("\n加载测试结构...")
        test_structures = []
        
        # 加载CIF文件
        cif_files = ["test_data/4450.cif", "test_data/4452.cif"]
        for cif_file in cif_files:
            if os.path.exists(cif_file):
                structure = Structure.from_file(cif_file)
                test_structures.append(structure)
                print(f"  加载: {cif_file} - {structure.formula}")
        
        if not test_structures:
            # 创建简单测试结构
            print("  创建简单测试结构...")
            lattice = Lattice.cubic(4.0)
            species = ["Fe", "Ni"]
            coords = [[0, 0, 0], [0.5, 0.5, 0.5]]
            test_structure = Structure(lattice, species, coords)
            test_structures.append(test_structure)
            print(f"  创建: {test_structure.formula}")
        
        # 测试能量计算
        print(f"\n测试能量计算 ({len(test_structures)} 个结构)...")
        for i, structure in enumerate(test_structures):
            try:
                energy = chgnet_calculator.calculateEnergy(structure)
                print(f"  结构 {i+1}: {structure.formula} - 能量: {energy:.6f} eV")
            except Exception as e:
                print(f"  结构 {i+1}: 计算失败 - {e}")
                return False
        
        print("✓ CHGNet封装器测试通过")
        return True
        
    except Exception as e:
        print(f"✗ CHGNet封装器测试失败: {e}")
        return False


def test_fitness_function():
    """测试适应度函数"""
    print("\n" + "="*50)
    print("测试适应度函数")
    print("="*50)
    
    try:
        # 创建测试用例
        lattice, site_groups = createSimpleTestCase()
        
        # 创建测试粒子
        from optimization.particle_encoding import Particle
        particle = Particle(site_groups)
        
        print(f"创建测试粒子:")
        for group_name, elements in particle.position.items():
            print(f"  {group_name}: {elements}")
        
        # 测试适应度计算
        from optimization.fitness_function import calculate_fitness_stage1, calculateEnergyDecomposition
        
        print(f"\n计算适应度...")
        fitness = calculate_fitness_stage1(particle, lattice, temperature=300.0)
        print(f"适应度: {fitness:.6f}")
        
        # 测试能量分解
        print(f"\n计算能量分解...")
        decomposition = calculateEnergyDecomposition(particle, lattice, temperature=300.0)
        
        for key, value in decomposition.items():
            print(f"  {key}: {value}")
        
        print("✓ 适应度函数测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 适应度函数测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pso_basic():
    """测试基本PSO功能"""
    print("\n" + "="*50)
    print("测试基本PSO功能")
    print("="*50)
    
    try:
        # 创建测试用例
        lattice, site_groups = createSimpleTestCase()
        
        # 创建PSO优化器（小规模测试）
        optimizer = PSOOptimizer(
            lattice=lattice,
            site_groups=site_groups,
            swarm_size=5,  # 小群体
            max_iterations=3,  # 少迭代
            temperature=300.0,
            inertia_weight=0.5,
            cognitive_factor=1.0,
            social_factor=1.0,
            mutation_rate=0.1
        )
        
        print("✓ PSO优化器创建成功")
        
        # 测试粒子初始化
        print("\n测试粒子初始化...")
        optimizer.initializeParticles()
        
        if optimizer.swarm.global_best_fitness < float('inf'):
            print(f"✓ 粒子初始化成功，全局最优: {optimizer.swarm.global_best_fitness:.6f}")
        else:
            print("✗ 粒子初始化失败")
            return False
        
        print("✓ 基本PSO功能测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 基本PSO功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_optimization():
    """测试完整优化流程"""
    print("\n" + "="*50)
    print("测试完整优化流程（小规模）")
    print("="*50)
    
    try:
        # 创建测试用例
        lattice, site_groups = createSimpleTestCase()
        
        # 创建PSO优化器
        optimizer = PSOOptimizer(
            lattice=lattice,
            site_groups=site_groups,
            swarm_size=3,  # 很小的群体
            max_iterations=2,  # 很少的迭代
            temperature=300.0,
            patience=5  # 早期停止
        )
        
        # 执行优化
        result = optimizer.optimize()
        
        # 检查结果
        if result["optimization_successful"]:
            print(f"✓ 优化成功完成")
            print(f"  迭代次数: {result['iterations_completed']}")
            print(f"  最优适应度: {result['global_best_fitness']:.6f}")
            print(f"  最优结构式: {result['best_structure'].formula}")
            
            # 保存结果
            optimizer.saveResult(result, "test_results")
            print("✓ 结果保存成功")
            
            return True
        else:
            print("✗ 优化未成功完成")
            return False
            
    except Exception as e:
        print(f"✗ 完整优化流程测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("高熵结构PSO优化系统测试")
    print("="*50)
    
    tests = [
        ("CHGNet封装器", test_chgnet_wrapper),
        ("适应度函数", test_fitness_function), 
        ("基本PSO功能", test_pso_basic),
        ("完整优化流程", test_full_optimization)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n✓ {test_name} 测试通过")
            else:
                print(f"\n✗ {test_name} 测试失败")
        except Exception as e:
            print(f"\n✗ {test_name} 测试异常: {e}")
    
    print(f"\n" + "="*50)
    print(f"测试完成: {passed}/{total} 通过")
    print("="*50)
    
    if passed == total:
        print("🎉 所有测试通过！系统可以使用。")
        return True
    else:
        print("❌ 部分测试失败，请检查系统配置。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)