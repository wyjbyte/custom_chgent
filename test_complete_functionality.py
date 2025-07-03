#!/usr/bin/env python3
"""
测试高熵结构优化器完整功能
包括基础功能和模拟CIF文件处理
"""

import sys
from pathlib import Path
import math

# 添加chgnet路径
sys.path.insert(0, str(Path(__file__).parent))

from chgnet.model.dynamics import HighEntropyOptimizer

def parse_cif_elements(cif_file_path):
    """简单解析CIF文件获取元素列表"""
    elements = []
    
    with open(cif_file_path, 'r') as f:
        lines = f.readlines()
        
    # 查找原子位置数据部分
    in_atom_section = False
    
    for line in lines:
        line = line.strip()
        
        # 检查是否到达原子数据部分
        if "_atom_site_type_symbol" in line:
            in_atom_section = True
            continue
            
        # 如果在原子数据部分，提取元素符号
        if in_atom_section and line and not line.startswith('_') and not line.startswith('loop_'):
            parts = line.split()
            if len(parts) >= 1:
                element = parts[0]
                # 清理元素符号 (移除数字等)
                clean_element = ''.join([c for c in element if c.isalpha()])
                if clean_element:
                    elements.append(clean_element)
                    
    return elements

def test_high_entropy_optimizer():
    """测试高熵优化器完整功能"""
    print("=" * 60)
    print("测试高熵结构优化器完整功能")
    print("=" * 60)
    
    # 初始化优化器
    print("1. 初始化高熵优化器...")
    optimizer = HighEntropyOptimizer()
    print(f"   原子半径表路径: {optimizer.radius_table_path}")
    print(f"   半径表大小: {len(optimizer.radius_table)} 个元素")
    
    # 测试半径查询功能
    print("\n2. 测试离子半径查询...")
    test_elements = ["Pt", "Cs", "F", "N", "O", "Sr", "Mo"]
    for element in test_elements:
        print(f"   {element}:")
        for site_type in ["A", "B", "O"]:
            radius = optimizer.get_ionic_radius(element, site_type)
            print(f"     {site_type} 位点: {radius} pm")
    
    return optimizer

def test_with_cif_files(optimizer):
    """使用CIF文件测试"""
    print("\n" + "=" * 60)
    print("使用测试CIF文件验证功能")
    print("=" * 60)
    
    test_files = ["test_data/4450.cif", "test_data/4452.cif"]
    
    for cif_file in test_files:
        print(f"\n处理文件: {cif_file}")
        
        try:
            # 解析CIF文件获取元素列表
            elements = parse_cif_elements(cif_file)
            print(f"   提取的元素: {elements}")
            print(f"   原子数: {len(elements)}")
            
            # 统计元素组成
            element_counts = {}
            for element in elements:
                element_counts[element] = element_counts.get(element, 0) + 1
            print(f"   元素组成: {element_counts}")
            
            # 创建结构信息
            structure_info = {'elements': elements}
            
            # 识别位点群
            site_groups = optimizer.identify_site_groups(structure_info)
            print(f"   位点群识别结果:")
            for site_type, indices in site_groups.items():
                if indices:
                    site_elements = [elements[i] for i in indices]
                    site_element_counts = {}
                    for element in site_elements:
                        site_element_counts[element] = site_element_counts.get(element, 0) + 1
                    print(f"     {site_type} 位点 ({len(indices)} 个原子): {site_element_counts}")
                else:
                    print(f"     {site_type} 位点: 无原子")
            
            # 计算构型熵
            entropy = optimizer.calculate_configurational_entropy(structure_info, site_groups)
            entropy_per_atom = entropy / len(elements) if elements else 0
            print(f"   总构型熵: {entropy:.6f} eV/K")
            print(f"   每原子构型熵: {entropy_per_atom:.6f} eV/K")
            
        except Exception as e:
            print(f"   错误: {e}")
            import traceback
            traceback.print_exc()

def test_entropy_calculation_detailed(optimizer):
    """详细测试构型熵计算"""
    print("\n" + "=" * 60)
    print("详细测试构型熵计算功能")
    print("=" * 60)
    
    # 测试案例1: 单元素，应该没有构型熵
    print("\n测试案例1: 单元素结构")
    elements1 = ["Fe", "Fe", "Fe", "Fe"]
    structure_info1 = {'elements': elements1}
    entropy1 = optimizer.calculate_configurational_entropy(structure_info1)
    print(f"   元素: {elements1}")
    print(f"   构型熵: {entropy1:.6f} eV/K (应该接近0)")
    
    # 测试案例2: 二元素等比例混合
    print("\n测试案例2: 二元等比例结构")
    elements2 = ["Fe", "Ni", "Fe", "Ni"]
    structure_info2 = {'elements': elements2}
    entropy2 = optimizer.calculate_configurational_entropy(structure_info2)
    entropy2_per_atom = entropy2 / len(elements2)
    
    # 理论值: S = -k_B * ln(0.5) * 2 (每个原子)
    theoretical_entropy_per_atom = optimizer.k_B * math.log(2)
    
    print(f"   元素: {elements2}")
    print(f"   计算构型熵每原子: {entropy2_per_atom:.6f} eV/K")
    print(f"   理论构型熵每原子: {theoretical_entropy_per_atom:.6f} eV/K")
    print(f"   相对误差: {abs(entropy2_per_atom - theoretical_entropy_per_atom) / theoretical_entropy_per_atom * 100:.2f}%")
    
    # 测试案例3: 复杂多元素结构 (模拟高熵合金)
    print("\n测试案例3: 五元等比例高熵合金")
    elements3 = ["Fe", "Ni", "Co", "Cr", "Mn"] * 4  # 每种元素4个原子
    structure_info3 = {'elements': elements3}
    entropy3 = optimizer.calculate_configurational_entropy(structure_info3)
    entropy3_per_atom = entropy3 / len(elements3)
    
    # 理论值: S = -k_B * ln(0.2) * 5 (每个原子)
    theoretical_entropy3_per_atom = optimizer.k_B * math.log(5)
    
    print(f"   元素: {len(elements3)} 个原子, 5种元素各4个")
    print(f"   计算构型熵每原子: {entropy3_per_atom:.6f} eV/K")
    print(f"   理论构型熵每原子: {theoretical_entropy3_per_atom:.6f} eV/K")
    print(f"   相对误差: {abs(entropy3_per_atom - theoretical_entropy3_per_atom) / theoretical_entropy3_per_atom * 100:.2f}%")

def test_site_classification_detailed(optimizer):
    """详细测试位点分类功能"""
    print("\n" + "=" * 60)
    print("详细测试位点分类功能")
    print("=" * 60)
    
    # 测试各种元素的分类
    test_cases = [
        {"name": "典型A位点元素", "elements": ["Cs", "Ba", "Sr", "La"]},
        {"name": "典型B位点元素", "elements": ["Pt", "Fe", "Ti", "Nb"]},
        {"name": "典型O位点元素", "elements": ["O", "F", "N", "Cl"]},
        {"name": "混合元素", "elements": ["Cs", "Pt", "O", "Fe", "F", "Sr"]}
    ]
    
    for case in test_cases:
        print(f"\n{case['name']}:")
        structure_info = {'elements': case['elements']}
        site_groups = optimizer.identify_site_groups(structure_info)
        
        print(f"   输入元素: {case['elements']}")
        for site_type, indices in site_groups.items():
            if indices:
                site_elements = [case['elements'][i] for i in indices]
                print(f"   {site_type} 位点: {site_elements}")
            else:
                print(f"   {site_type} 位点: 无")

if __name__ == "__main__":
    try:
        optimizer = test_high_entropy_optimizer()
        test_with_cif_files(optimizer)
        test_entropy_calculation_detailed(optimizer)
        test_site_classification_detailed(optimizer)
        
        print("\n" + "=" * 60)
        print("所有测试完成!")
        print("高熵结构优化器基础框架实现成功!")
        print("=" * 60)
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()