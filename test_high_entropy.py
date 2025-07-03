#!/usr/bin/env python3
"""
测试高熵结构优化器功能
使用提供的测试数据验证功能实现
"""

import sys
from pathlib import Path
from pymatgen.core.structure import Structure

# 添加chgnet路径
sys.path.insert(0, str(Path(__file__).parent))

from chgnet.model.dynamics import HighEntropyOptimizer

def test_basic_functionality():
    """测试基础功能"""
    print("=" * 50)
    print("测试高熵结构优化器基础功能")
    print("=" * 50)
    
    # 初始化优化器
    print("1. 初始化高熵优化器...")
    optimizer = HighEntropyOptimizer()
    print(f"   原子半径表路径: {optimizer.radius_table_path}")
    print(f"   半径表形状: {optimizer.radius_table.shape}")
    print(f"   半径表列名: {list(optimizer.radius_table.columns)}")
    
    # 测试半径查询功能
    print("\n2. 测试离子半径查询...")
    test_elements = ["Pt", "Cs", "F", "N", "O"]
    for element in test_elements:
        for site_type in ["A", "B", "O"]:
            radius = optimizer.get_ionic_radius(element, site_type)
            print(f"   {element} 在 {site_type} 位点: {radius} pm")
    
    return optimizer

def test_with_cif_files(optimizer):
    """使用CIF文件测试"""
    print("\n" + "=" * 50)
    print("使用测试CIF文件验证功能")
    print("=" * 50)
    
    test_files = ["test_data/4450.cif", "test_data/4452.cif"]
    
    for cif_file in test_files:
        print(f"\n处理文件: {cif_file}")
        
        try:
            # 加载结构
            structure = Structure.from_file(cif_file)
            print(f"   结构公式: {structure.formula}")
            print(f"   原子数: {len(structure)}")
            print(f"   晶格参数: {structure.lattice.abc}")
            
            # 识别位点群
            site_groups = optimizer.identify_site_groups(structure)
            print(f"   位点群识别结果:")
            for site_type, indices in site_groups.items():
                if indices:
                    elements = [structure[i].species_string for i in indices]
                    element_counts = {}
                    for element in elements:
                        element_counts[element] = element_counts.get(element, 0) + 1
                    print(f"     {site_type} 位点 ({len(indices)} 个原子): {element_counts}")
                else:
                    print(f"     {site_type} 位点: 无原子")
            
            # 计算构型熵
            entropy = optimizer.calculate_configurational_entropy(structure, site_groups)
            entropy_per_atom = entropy / len(structure)
            print(f"   总构型熵: {entropy:.6f} eV/K")
            print(f"   每原子构型熵: {entropy_per_atom:.6f} eV/K")
            
        except Exception as e:
            print(f"   错误: {e}")

def test_entropy_calculation():
    """测试构型熵计算的正确性"""
    print("\n" + "=" * 50)
    print("测试构型熵计算正确性")
    print("=" * 50)
    
    optimizer = HighEntropyOptimizer()
    
    # 创建一个简单的测试结构
    from pymatgen.core.lattice import Lattice
    
    # 创建立方晶格
    lattice = Lattice.cubic(5.0)
    
    # 测试情况1: 单元素，应该没有构型熵
    species = ["Fe", "Fe", "Fe", "Fe"]
    coords = [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
    structure1 = Structure(lattice, species, coords)
    
    entropy1 = optimizer.calculate_configurational_entropy(structure1)
    print(f"单元素结构构型熵: {entropy1:.6f} eV/K (应该接近0)")
    
    # 测试情况2: 二元素等比例混合
    species = ["Fe", "Ni", "Fe", "Ni"]
    structure2 = Structure(lattice, species, coords)
    
    entropy2 = optimizer.calculate_configurational_entropy(structure2)
    entropy2_per_atom = entropy2 / len(structure2)
    
    # 理论值: S = -k_B * ln(0.5) * 2 = k_B * ln(2) * 2 (每个原子)
    theoretical_entropy_per_atom = optimizer.k_B * np.log(2)
    
    print(f"二元等比例结构:")
    print(f"   计算构型熵每原子: {entropy2_per_atom:.6f} eV/K")
    print(f"   理论构型熵每原子: {theoretical_entropy_per_atom:.6f} eV/K")
    print(f"   相对误差: {abs(entropy2_per_atom - theoretical_entropy_per_atom) / theoretical_entropy_per_atom * 100:.2f}%")

if __name__ == "__main__":
    import numpy as np
    
    # 运行所有测试
    try:
        optimizer = test_basic_functionality()
        test_with_cif_files(optimizer)
        test_entropy_calculation()
        
        print("\n" + "=" * 50)
        print("所有测试完成!")
        print("=" * 50)
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()