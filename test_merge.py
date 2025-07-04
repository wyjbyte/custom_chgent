#!/usr/bin/env python3
"""
测试合并后的PSO优化系统
验证三个分支的功能都能正常工作
"""

def test_basic_imports():
    """测试基础导入功能"""
    print("=== 测试基础导入 ===")
    
    try:
        from chgnet.model.dynamics import HighEntropyOptimizer
        print("✓ HighEntropyOptimizer 导入成功")
    except Exception as e:
        print(f"✗ HighEntropyOptimizer 导入失败: {e}")
    
    try:
        from chgnet.optimization import PSOParticle
        print("✓ PSOParticle 导入成功")
    except Exception as e:
        print(f"✗ PSOParticle 导入失败: {e}")
    
    try:
        from optimization.chgnet_wrapper import CHGNetEnergyCalculator
        print("✓ CHGNetEnergyCalculator 导入成功")
    except Exception as e:
        print(f"✗ CHGNetEnergyCalculator 导入失败: {e}")

def test_structure_functionality():
    """测试结构功能（不依赖外部库）"""
    print("\n=== 测试结构功能 ===")
    
    # 测试位点群定义
    site_groups_definition = {
        "test_group": {
            "site_indices": [0, 1, 2, 3],
            "elements": [24, 25, 26, 27],  # Cr, Mn, Fe, Co
            "element_counts": {"Cr": 1, "Mn": 1, "Fe": 1, "Co": 1}
        }
    }
    
    try:
        # 测试高熵优化器基础功能
        from chgnet.model.dynamics import HighEntropyOptimizer
        
        # 创建优化器实例
        optimizer = HighEntropyOptimizer()
        print("✓ HighEntropyOptimizer 实例创建成功")
        
        # 测试位点群识别
        structure_info = {'elements': ['Cr', 'Mn', 'Fe', 'Co']}
        site_groups = optimizer.identify_site_groups(structure_info)
        print(f"✓ 位点群识别成功: {list(site_groups.keys())}")
        
    except Exception as e:
        print(f"✗ 高熵优化器测试失败: {e}")

def test_file_structure():
    """测试文件结构"""
    print("\n=== 测试文件结构 ===")
    
    import os
    
    expected_files = [
        "chgnet/model/dynamics.py",
        "chgnet/optimization/__init__.py", 
        "chgnet/optimization/particle.py",
        "chgnet/optimization/entropy.py",
        "chgnet/optimization/fitness.py",
        "chgnet/optimization/structure_ops.py",
        "optimization/__init__.py",
        "optimization/chgnet_wrapper.py",
        "MERGED_PSO_README.md",
        ".gitignore"
    ]
    
    for file_path in expected_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path} 存在")
        else:
            print(f"✗ {file_path} 缺失")

def main():
    """主测试函数"""
    print("高熵结构PSO优化系统 - 合并测试")
    print("=" * 50)
    
    test_basic_imports()
    test_structure_functionality() 
    test_file_structure()
    
    print("\n" + "=" * 50)
    print("测试完成！")
    print("注意: 部分功能需要安装 numpy, torch, pymatgen 等依赖才能完全测试")

if __name__ == "__main__":
    main()