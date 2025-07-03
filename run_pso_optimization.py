#!/usr/bin/env python3
"""
高熵结构PSO优化主程序
实现阶段一PSO主循环与CHGNet集成
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
from pymatgen.core import Structure
from pymatgen.core.lattice import Lattice

from optimization.pso_optimizer import PSOOptimizer, createSimpleTestCase
from optimization.particle_encoding import HighEntropySiteGroup


def createHighEntropyAlloyCase() -> tuple:
    """
    创建高熵合金优化案例
    
    Returns:
        tuple: (lattice, site_groups, description)
    """
    # 创建FCC晶格 (面心立方)
    lattice = Lattice.from_parameters(a=3.6, b=3.6, c=3.6, alpha=90, beta=90, gamma=90)
    
    # FCC结构的4个原子位点 (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
    positions = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.5, 0.5, 0.0]),
        np.array([0.5, 0.0, 0.5]),
        np.array([0.0, 0.5, 0.5])
    ]
    
    # 定义高熵合金成分：CrFeMnNi (等摩尔比)
    element_counts = {"Cr": 1, "Fe": 1, "Mn": 1, "Ni": 1}
    
    site_group = HighEntropySiteGroup("fcc_sites", positions, element_counts)
    
    description = "CrFeMnNi高熵合金FCC结构优化"
    
    return lattice, [site_group], description


def createLargerTestCase() -> tuple:
    """
    创建更大的测试案例
    
    Returns:
        tuple: (lattice, site_groups, description)
    """
    # 创建2x2x2超晶胞
    lattice = Lattice.cubic(8.0)
    
    # 创建32个原子位点 (4x4x2简化排列)
    positions = []
    for i in range(4):
        for j in range(4):
            for k in range(2):
                x = i * 0.25
                y = j * 0.25  
                z = k * 0.5
                positions.append(np.array([x, y, z]))
    
    # 定义5元高熵合金成分
    total_sites = len(positions)
    sites_per_element = total_sites // 5
    remainder = total_sites % 5
    
    element_counts = {
        "Fe": sites_per_element + (1 if remainder > 0 else 0),
        "Ni": sites_per_element + (1 if remainder > 1 else 0),
        "Cr": sites_per_element + (1 if remainder > 2 else 0),
        "Co": sites_per_element + (1 if remainder > 3 else 0),
        "Mn": sites_per_element
    }
    
    site_group = HighEntropySiteGroup("supercell_sites", positions, element_counts)
    
    description = f"FeCrNiCoMn高熵合金超晶胞优化 ({total_sites}原子)"
    
    return lattice, [site_group], description


def loadFromCIF(cif_file: str) -> tuple:
    """
    从CIF文件加载结构并设置为优化起点
    
    Args:
        cif_file: CIF文件路径
        
    Returns:
        tuple: (lattice, site_groups, description)
    """
    structure = Structure.from_file(cif_file)
    lattice = structure.lattice
    
    # 提取所有原子位点
    positions = []
    elements = []
    
    for site in structure.sites:
        positions.append(site.frac_coords)
        elements.append(str(site.specie))
    
    # 统计元素数量
    element_counts = {}
    for element in elements:
        element_counts[element] = element_counts.get(element, 0) + 1
    
    site_group = HighEntropySiteGroup("loaded_sites", positions, element_counts)
    
    description = f"从{cif_file}加载的结构优化 - {structure.formula}"
    
    return lattice, [site_group], description


def parseArguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="高熵结构PSO优化")
    
    parser.add_argument("--case", type=str, default="simple",
                       choices=["simple", "heafcc", "large", "cif"],
                       help="优化案例类型")
    
    parser.add_argument("--cif", type=str, 
                       help="CIF文件路径 (当case=cif时必需)")
    
    parser.add_argument("--swarm-size", type=int, default=20,
                       help="粒子群大小")
    
    parser.add_argument("--max-iterations", type=int, default=100,
                       help="最大迭代次数")
    
    parser.add_argument("--temperature", type=float, default=300.0,
                       help="温度 (K)")
    
    parser.add_argument("--output-dir", type=str, default="pso_results",
                       help="结果输出目录")
    
    # PSO参数
    parser.add_argument("--inertia", type=float, default=0.5,
                       help="惯性权重")
    parser.add_argument("--cognitive", type=float, default=1.0,
                       help="认知因子")
    parser.add_argument("--social", type=float, default=1.0,
                       help="社会因子")
    parser.add_argument("--mutation", type=float, default=0.1,
                       help="变异率")
    parser.add_argument("--patience", type=int, default=10,
                       help="早停耐心值")
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parseArguments()
    
    print("="*60)
    print("高熵结构PSO优化系统")
    print("阶段一：CHGNet集成与基础优化")
    print("="*60)
    
    # 创建优化案例
    try:
        if args.case == "simple":
            lattice, site_groups, description = createSimpleTestCase()
        elif args.case == "heafcc":
            lattice, site_groups, description = createHighEntropyAlloyCase()
        elif args.case == "large":
            lattice, site_groups, description = createLargerTestCase()
        elif args.case == "cif":
            if not args.cif:
                print("错误：使用CIF模式时必须指定--cif参数")
                return 1
            if not os.path.exists(args.cif):
                print(f"错误：CIF文件不存在: {args.cif}")
                return 1
            lattice, site_groups, description = loadFromCIF(args.cif)
        else:
            print(f"错误：未知的案例类型: {args.case}")
            return 1
            
        print(f"优化案例: {description}")
        print(f"晶格参数: a={lattice.a:.3f}, b={lattice.b:.3f}, c={lattice.c:.3f}")
        
        total_atoms = sum(group.num_sites for group in site_groups)
        print(f"总原子数: {total_atoms}")
        
        for group in site_groups:
            print(f"位点群 '{group.name}': {group.num_sites} 个位点")
            for element, count in group.element_counts.items():
                print(f"  {element}: {count} 个")
                
    except Exception as e:
        print(f"错误：创建优化案例失败: {e}")
        return 1
    
    # 创建PSO优化器
    try:
        print(f"\n创建PSO优化器...")
        optimizer = PSOOptimizer(
            lattice=lattice,
            site_groups=site_groups,
            swarm_size=args.swarm_size,
            max_iterations=args.max_iterations,
            temperature=args.temperature,
            inertia_weight=args.inertia,
            cognitive_factor=args.cognitive,
            social_factor=args.social,
            mutation_rate=args.mutation,
            patience=args.patience
        )
        
    except Exception as e:
        print(f"错误：PSO优化器创建失败: {e}")
        return 1
    
    # 执行优化
    try:
        print(f"\n开始优化...")
        result = optimizer.optimize()
        
        if result["optimization_successful"]:
            print(f"\n✓ 优化成功完成！")
            
            # 保存结果
            optimizer.saveResult(result, args.output_dir)
            
            return 0
        else:
            print(f"\n✗ 优化失败")
            return 1
            
    except KeyboardInterrupt:
        print(f"\n\n用户中断优化")
        return 1
    except Exception as e:
        print(f"\n错误：优化过程失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)