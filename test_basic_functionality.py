#!/usr/bin/env python3
"""
测试高熵结构优化器的基本功能 (无需外部依赖)
只测试CSV解析和基础计算功能
"""

import csv
import os
import sys
from pathlib import Path
import math

def test_csv_parsing():
    """测试CSV文件解析功能"""
    print("=" * 50)
    print("测试原子半径表CSV解析功能")
    print("=" * 50)
    
    csv_path = "chgnet/data/Atomic_radius_table.csv"
    
    if not os.path.exists(csv_path):
        print(f"错误: CSV文件不存在 {csv_path}")
        return
        
    # 手动解析CSV文件
    radius_data = {}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # 读取表头
        print(f"CSV表头: {header}")
        
        for i, row in enumerate(reader):
            if len(row) >= 4:
                element = row[0]
                radius_a = row[1] if row[1] else None
                radius_b = row[2] if row[2] else None  
                radius_o = row[3] if row[3] else None
                
                radius_data[element] = {
                    'A': float(radius_a) if radius_a else None,
                    'B': float(radius_b) if radius_b else None,
                    'O': float(radius_o) if radius_o else None
                }
                
                # 只打印前10个元素
                if i < 10:
                    print(f"   {element}: A={radius_a}, B={radius_b}, O={radius_o}")
                    
    print(f"\n总共解析了 {len(radius_data)} 个元素")
    
    # 测试一些特定元素
    test_elements = ["Pt", "Cs", "F", "N", "O"]
    print(f"\n测试特定元素的半径查询:")
    
    for element in test_elements:
        if element in radius_data:
            data = radius_data[element]
            print(f"   {element}:")
            for site_type in ['A', 'B', 'O']:
                radius = data[site_type]
                if radius is not None:
                    print(f"     {site_type} 位点: {radius} pm")
                else:
                    print(f"     {site_type} 位点: 未定义")
        else:
            print(f"   {element}: 未在表中找到")
            
    return radius_data

def test_entropy_calculation():
    """测试构型熵计算的数学正确性"""
    print("\n" + "=" * 50)
    print("测试构型熵计算数学功能")
    print("=" * 50)
    
    # 玻尔兹曼常数 (eV/K)
    k_B = 8.617333262145e-5
    
    def calculate_entropy(element_counts):
        """计算给定元素计数的构型熵"""
        total_atoms = sum(element_counts.values())
        if total_atoms <= 1:
            return 0.0
            
        entropy = 0.0
        for element, count in element_counts.items():
            mole_fraction = count / total_atoms
            if mole_fraction > 1e-10:
                entropy -= mole_fraction * math.log(mole_fraction)
                
        return entropy * k_B * total_atoms
    
    # 测试案例1: 单元素 (应该为0)
    case1 = {"Fe": 4}
    entropy1 = calculate_entropy(case1)
    print(f"测试案例1 - 单元素 {case1}: {entropy1:.6f} eV/K (应该 = 0)")
    
    # 测试案例2: 二元等比例
    case2 = {"Fe": 2, "Ni": 2}
    entropy2 = calculate_entropy(case2)
    theoretical2 = k_B * math.log(2) * 4  # 4个原子，每个贡献k_B*ln(2)
    print(f"测试案例2 - 二元等比例 {case2}: {entropy2:.6f} eV/K")
    print(f"   理论值: {theoretical2:.6f} eV/K")
    print(f"   相对误差: {abs(entropy2 - theoretical2) / theoretical2 * 100:.2f}%")
    
    # 测试案例3: 三元等比例
    case3 = {"Fe": 1, "Ni": 1, "Co": 1}
    entropy3 = calculate_entropy(case3)
    theoretical3 = k_B * math.log(3) * 3
    print(f"测试案例3 - 三元等比例 {case3}: {entropy3:.6f} eV/K")
    print(f"   理论值: {theoretical3:.6f} eV/K")
    print(f"   相对误差: {abs(entropy3 - theoretical3) / theoretical3 * 100:.2f}%")

def test_site_classification():
    """测试位点分类逻辑"""
    print("\n" + "=" * 50)
    print("测试位点分类逻辑")
    print("=" * 50)
    
    # 模拟4450.cif中的元素
    elements_4450 = ["Pt", "Cs", "F", "N", "O"]
    
    # 模拟半径表数据 (简化版)
    radius_table = {
        "Pt": {"A": None, "B": 62.5, "O": None},
        "Cs": {"A": 188, "B": None, "O": None},
        "F": {"A": None, "B": None, "O": 133},
        "N": {"A": None, "B": None, "O": 146},
        "O": {"A": None, "B": None, "O": 140}
    }
    
    site_groups = {"A": [], "B": [], "O": []}
    
    for i, element in enumerate(elements_4450):
        if element in radius_table:
            data = radius_table[element]
            # 优先级: O > A > B
            if data["O"] is not None:
                site_groups["O"].append(i)
                site_type = "O"
            elif data["A"] is not None:
                site_groups["A"].append(i)
                site_type = "A"
            elif data["B"] is not None:
                site_groups["B"].append(i)
                site_type = "B"
            else:
                site_type = "未分类"
                
            print(f"   元素 {element} (索引 {i}) -> {site_type} 位点")
        else:
            print(f"   元素 {element} (索引 {i}) -> 未在表中找到")
            
    print(f"\n位点分类结果:")
    for site_type, indices in site_groups.items():
        elements_in_site = [elements_4450[i] for i in indices]
        print(f"   {site_type} 位点: {elements_in_site}")

if __name__ == "__main__":
    try:
        radius_data = test_csv_parsing()
        test_entropy_calculation()
        test_site_classification()
        
        print("\n" + "=" * 50)
        print("所有基础功能测试完成!")
        print("=" * 50)
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()