#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用PSO方法对晶体结构进行批量优化的脚本。
此脚本可以处理单个CIF文件或文件夹中的所有CIF文件。
"""

import os
import time
import argparse
import glob
import numpy as np
from pymatgen.core import Structure
from chgnet.model.dynamics import StructOptimizer


def compare_structures(original: Structure, optimized: Structure, model) -> dict:
    """
    比较优化前后的结构，返回相关信息。

    Args:
        original: 原始结构
        optimized: 优化后的结构
        model: CHGNet模型用于能量计算

    Returns:
        包含比较信息的字典
    """
    # 计算能量
    orig_pred = model.predict_structure(original, task='e')
    opt_pred = model.predict_structure(optimized, task='e')

    orig_energy = float(orig_pred['e'])
    opt_energy = float(opt_pred['e'])
    energy_diff = opt_energy - orig_energy

    # 计算原子位移
    orig_coords = original.cart_coords
    opt_coords = optimized.cart_coords

    # 计算每个原子的位移距离
    displacements = np.linalg.norm(opt_coords - orig_coords, axis=1)
    max_disp = np.max(displacements)
    avg_disp = np.mean(displacements)

    # 计算晶格参数的变化
    orig_latt = original.lattice.abc
    opt_latt = optimized.lattice.abc

    return {
        "original_energy": orig_energy,
        "optimized_energy": opt_energy,
        "energy_difference": energy_diff,
        "energy_improvement_percentage": (energy_diff / abs(orig_energy)) * 100 if orig_energy != 0 else 0,
        "max_atomic_displacement": max_disp,
        "average_atomic_displacement": avg_disp,
        "original_lattice": orig_latt,
        "optimized_lattice": opt_latt
    }


def optimize_structure(input_file, output_file, optimizer, pso_params):
    """
    优化单个结构文件并保存结果

    Args:
        input_file: 输入CIF文件路径
        output_file: 输出CIF文件路径
        optimizer: 结构优化器实例
        pso_params: PSO算法参数字典

    Returns:
        比较结果字典
    """
    try:
        # 加载初始结构
        print(f"\n处理文件: {input_file}")
        initial_structure = Structure.from_file(input_file)
        print(f"结构信息: {len(initial_structure)}个原子, 化学式: {initial_structure.composition.formula}")

        # 执行优化
        print("开始进行PSO结构优化...")
        start_time = time.time()

        final_structure = optimizer.relax_pso(
            initial_structure,
            n_particles=pso_params['n_particles'],
            max_iter=pso_params['max_iter'],
            c1=pso_params['c1'],
            c2=pso_params['c2'],
            w=pso_params['w']
        )

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"优化完成！用时: {elapsed_time:.2f}秒")

        # 比较优化前后的结构
        comparison = compare_structures(initial_structure, final_structure, optimizer.calculator.model)

        print(f"原始能量: {comparison['original_energy']:.6f} eV/atom")
        print(f"优化后能量: {comparison['optimized_energy']:.6f} eV/atom")
        print(
            f"能量变化: {comparison['energy_difference']:.6f} eV/atom ({comparison['energy_improvement_percentage']:.2f}%)")
        print(f"最大原子位移: {comparison['max_atomic_displacement']:.6f} Å")
        print(f"平均原子位移: {comparison['average_atomic_displacement']:.6f} Å")

        # 保存优化后的结构
        final_structure.to(filename=output_file)
        print(f"优化后的结构已保存到: {output_file}")

        return comparison

    except Exception as e:
        print(f"处理文件 {input_file} 时出错: {str(e)}")
        return None


def process_input_path(input_path, output_dir, optimizer, pso_params):
    """
    处理输入路径(可以是单个文件或目录)

    Args:
        input_path: 输入文件或目录路径
        output_dir: 输出目录
        optimizer: 结构优化器实例
        pso_params: PSO算法参数字典
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 如果输入是目录，处理目录下所有CIF文件
    if os.path.isdir(input_path):
        cif_files = glob.glob(os.path.join(input_path, "*.cif"))
        if not cif_files:
            print(f"警告：在目录 {input_path} 中未找到CIF文件")
            return

        print(f"在目录 {input_path} 中找到 {len(cif_files)} 个CIF文件")

        # 创建结果摘要
        summary = []

        # 处理每个CIF文件
        for cif_file in cif_files:
            base_name = os.path.splitext(os.path.basename(cif_file))[0]
            output_file = os.path.join(output_dir, f"{base_name}_optimized.cif")

            # 优化结构
            result = optimize_structure(cif_file, output_file, optimizer, pso_params)
            if result:
                result['input_file'] = cif_file
                result['output_file'] = output_file
                summary.append(result)

        # 打印批处理摘要
        if summary:
            print("\n==== 批处理摘要 ====")
            print(f"成功优化: {len(summary)}/{len(cif_files)} 个结构")

            # 找出能量改善最大的结构
            best_improvement = min(summary, key=lambda x: x['energy_difference'])
            print(f"\n能量改善最大的结构: {os.path.basename(best_improvement['input_file'])}")
            print(
                f"能量改善: {best_improvement['energy_difference']:.6f} eV/atom ({best_improvement['energy_improvement_percentage']:.2f}%)")

    # 如果输入是单个文件，只处理该文件
    elif os.path.isfile(input_path) and input_path.lower().endswith('.cif'):
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_file = os.path.join(output_dir, f"{base_name}_optimized.cif")

        optimize_structure(input_path, output_file, optimizer, pso_params)

    else:
        print(f"错误: 输入路径 {input_path} 不是有效的CIF文件或目录")


def main():
    """主函数，执行PSO结构优化流程"""

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='使用PSO方法优化晶体结构')
    parser.add_argument('--input', '-i', type=str, default='./chgnet/test_data/diffcsp',help='输入CIF文件或包含CIF文件的目录路径')
    # parser.add_argument('--output', '-o', type=str, default='demo6_03_output',help='输出目录，用于保存优化后的结构')
    parser.add_argument('--output', '-o', type=str, default='demo6_02_output',help='输出目录，用于保存优化后的结构')
    parser.add_argument('--particles', '-p', type=int, default=10,help='PSO算法的粒子数量 (默认: 10)')
    parser.add_argument('--iterations', '-n', type=int, default=100,help='PSO算法的最大迭代次数 (默认: 50)')
    parser.add_argument('--c1', type=float, default=0.5,help='PSO认知参数 (默认: 0.5)')
    parser.add_argument('--c2', type=float, default=0.5,help='PSO社会参数 (默认: 0.5)')
    parser.add_argument('--w', type=float, default=0.9,help='PSO惯性权重 (默认: 0.9)')

    args = parser.parse_args()

    # 初始化优化器
    print("初始化CHGNet优化器...")
    optimizer = StructOptimizer()

    # 设置PSO参数
    pso_params = {
        'n_particles': args.particles,
        'max_iter': args.iterations,
        'c1': args.c1,
        'c2': args.c2,
        'w': args.w
    }

    print(f"PSO参数: 粒子数={pso_params['n_particles']}, "
          f"迭代次数={pso_params['max_iter']}, "
          f"c1={pso_params['c1']}, c2={pso_params['c2']}, w={pso_params['w']}")

    # 处理输入路径
    process_input_path(args.input, args.output, optimizer, pso_params)

    print("\n所有优化任务完成！")


if __name__ == "__main__":
    main()