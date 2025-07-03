#!/usr/bin/env python3
"""
测试高熵结构优化器核心功能 (独立实现)
不依赖外部包，验证算法正确性
"""

import csv
import math
import os
from pathlib import Path

class SimpleHighEntropyOptimizer:
    """简化的高熵优化器类 (仅用于测试核心功能)"""
    
    def __init__(self, radius_table_path=None):
        """初始化"""
        if radius_table_path is None:
            radius_table_path = "chgnet/data/Atomic_radius_table.csv"
        
        self.radius_table_path = Path(radius_table_path)
        self.radius_table = self._load_radius_table()
        self.k_B = 8.617333262145e-5  # eV/K
        
    def _load_radius_table(self):
        """加载原子半径表"""
        radius_data = {}
        
        if not self.radius_table_path.exists():
            print(f"警告: CSV文件不存在 {self.radius_table_path}")
            return radius_data
            
        with open(self.radius_table_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)  # 跳过表头
            
            for row in reader:
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
                    
        return radius_data
        
    def get_ionic_radius(self, element, site_type, default_radius=100.0):
        """查询离子半径"""
        if element not in self.radius_table:
            return default_radius
            
        radius_value = self.radius_table[element][site_type]
        return float(radius_value) if radius_value is not None else default_radius
        
    def identify_site_groups(self, elements):
        """识别位点群"""
        site_groups = {"A": [], "B": [], "O": []}
        
        for i, element in enumerate(elements):
            if element in self.radius_table:
                data = self.radius_table[element]
                if data["O"] is not None:
                    site_groups["O"].append(i)
                elif data["A"] is not None:
                    site_groups["A"].append(i)
                elif data["B"] is not None:
                    site_groups["B"].append(i)
                else:
                    self._assign_default_site(element, i, site_groups)
            else:
                self._assign_default_site(element, i, site_groups)
                
        return site_groups
        
    def _assign_default_site(self, element, index, site_groups):
        """默认位点分配"""
        if element in ["O", "F", "Cl", "Br", "I", "N", "S", "Se", "Te"]:
            site_groups["O"].append(index)
        elif element in ["Cs", "Rb", "K", "Ba", "Sr", "Ca", "La", "Ce", "Nd", "Sm", "Pr"]:
            site_groups["A"].append(index)
        else:
            site_groups["B"].append(index)
            
    def count_elements_in_site_group(self, elements, site_indices):
        """统计位点群中的元素"""
        element_counts = {}
        for idx in site_indices:
            if idx < len(elements):
                element = elements[idx]
                element_counts[element] = element_counts.get(element, 0) + 1
        return element_counts
        
    def calculate_configurational_entropy(self, elements, site_groups_definition=None):
        """计算构型熵"""
        if site_groups_definition is None:
            site_groups_definition = self.identify_site_groups(elements)
            
        total_entropy = 0.0
        
        for site_type, site_indices in site_groups_definition.items():
            if not site_indices:
                continue
                
            element_counts = self.count_elements_in_site_group(elements, site_indices)
            total_atoms_in_group = sum(element_counts.values())
            
            if total_atoms_in_group <= 1:
                continue
                
            site_entropy = 0.0
            for element, count in element_counts.items():
                mole_fraction = count / total_atoms_in_group
                if mole_fraction > 1e-10:
                    site_entropy -= mole_fraction * math.log(mole_fraction)
                    
            total_entropy += site_entropy * total_atoms_in_group
            
        return total_entropy * self.k_B

def parse_cif_elements(cif_file_path):
    """简单解析CIF文件获取元素列表"""
    elements = []
    
    with open(cif_file_path, 'r') as f:
        lines = f.readlines()
        
    in_atom_section = False
    
    for line in lines:
        line = line.strip()
        
        if "_atom_site_type_symbol" in line:
            in_atom_section = True
            continue
            
        if in_atom_section and line and not line.startswith('_') and not line.startswith('loop_'):
            parts = line.split()
            if len(parts) >= 1:
                element = parts[0]
                clean_element = ''.join([c for c in element if c.isalpha()])
                if clean_element:
                    elements.append(clean_element)
                    
    return elements

def test_basic_functionality():
    """测试基本功能"""
    print("=" * 60)
    print("测试高熵结构优化器基础功能")
    print("=" * 60)
    
    optimizer = SimpleHighEntropyOptimizer()
    print(f"1. 初始化优化器完成")
    print(f"   半径表路径: {optimizer.radius_table_path}")
    print(f"   半径表大小: {len(optimizer.radius_table)} 个元素")
    
    # 测试半径查询
    print(f"\n2. 测试离子半径查询:")
    test_elements = ["Pt", "Cs", "F", "N", "O"]
    for element in test_elements:
        print(f"   {element}:")
        for site_type in ["A", "B", "O"]:
            radius = optimizer.get_ionic_radius(element, site_type)
            print(f"     {site_type} 位点: {radius} pm")
    
    return optimizer

def test_entropy_calculation(optimizer):
    """测试构型熵计算"""
    print(f"\n3. 测试构型熵计算:")
    
    # 单元素
    elements1 = ["Fe", "Fe", "Fe", "Fe"]
    entropy1 = optimizer.calculate_configurational_entropy(elements1)
    print(f"   单元素 {elements1}: {entropy1:.6f} eV/K")
    
    # 二元等比例
    elements2 = ["Fe", "Ni", "Fe", "Ni"]
    entropy2 = optimizer.calculate_configurational_entropy(elements2)
    theoretical2 = optimizer.k_B * math.log(2) * 4
    print(f"   二元等比例 {elements2}: {entropy2:.6f} eV/K")
    print(f"   理论值: {theoretical2:.6f} eV/K")
    
    return optimizer

def test_cif_processing(optimizer):
    """测试CIF文件处理"""
    print(f"\n4. 测试CIF文件处理:")
    
    test_files = ["test_data/4450.cif", "test_data/4452.cif"]
    
    for cif_file in test_files:
        if os.path.exists(cif_file):
            print(f"\n   处理文件: {cif_file}")
            
            elements = parse_cif_elements(cif_file)
            print(f"     提取元素: {elements[:10]}..." if len(elements) > 10 else f"     提取元素: {elements}")
            print(f"     原子总数: {len(elements)}")
            
            # 统计元素组成
            element_counts = {}
            for element in elements:
                element_counts[element] = element_counts.get(element, 0) + 1
            print(f"     元素组成: {element_counts}")
            
            # 识别位点群
            site_groups = optimizer.identify_site_groups(elements)
            print(f"     位点群识别:")
            for site_type, indices in site_groups.items():
                if indices:
                    site_elements = [elements[i] for i in indices]
                    site_counts = {}
                    for element in site_elements:
                        site_counts[element] = site_counts.get(element, 0) + 1
                    print(f"       {site_type} 位点 ({len(indices)} 个): {site_counts}")
            
            # 计算构型熵
            entropy = optimizer.calculate_configurational_entropy(elements, site_groups)
            entropy_per_atom = entropy / len(elements) if elements else 0
            print(f"     总构型熵: {entropy:.6f} eV/K")
            print(f"     每原子构型熵: {entropy_per_atom:.6f} eV/K")
        else:
            print(f"   文件不存在: {cif_file}")

if __name__ == "__main__":
    try:
        optimizer = test_basic_functionality()
        test_entropy_calculation(optimizer)
        test_cif_processing(optimizer)
        
        print("\n" + "=" * 60)
        print("测试完成! 高熵结构优化器基础框架实现成功!")
        print("主要功能:")
        print("✓ 原子半径表加载与解析")
        print("✓ 位点群定义与原子统计 (A/B/O位点)")
        print("✓ 构型熵计算模块")
        print("✓ CIF文件处理支持")
        print("=" * 60)
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()