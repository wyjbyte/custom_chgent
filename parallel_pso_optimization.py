#!/usr/bin/env python3

"""
使用PSO方法对晶体结构进行批量并行优化的脚本。
此脚本可以处理单个CIF文件或文件夹中的所有CIF文件。
支持基于构型熵的优化。
支持多进程并行处理，大幅提高处理速度。
"""

import os
import time
import argparse
import glob
import numpy as np
from pymatgen.core import Structure
from chgnet.model.dynamics import StructOptimizer, calculate_distance_entropy
from multiprocessing import Pool, cpu_count
import tqdm
import signal
import sys


def compare_structures(original: Structure, optimized: Structure, model, use_entropy: bool = False,
                       entropy_bin_width: float = 0.2, entropy_cutoff: float = 10.0) -> dict:
    """
    比较优化前后的结构，返回相关信息。

    Args:
        original: 原始结构
        optimized: 优化后的结构
        model: CHGNet模型用于能量计算
        use_entropy: 是否计算并比较构型熵
        entropy_bin_width: 构型熵计算中使用的分箱宽度
        entropy_cutoff: 构型熵计算中使用的距离截断

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

    result = {
        "original_energy": orig_energy,
        "optimized_energy": opt_energy,
        "energy_difference": energy_diff,
        "energy_improvement_percentage": (energy_diff / abs(orig_energy)) * 100 if orig_energy != 0 else 0,
        "max_atomic_displacement": max_disp,
        "average_atomic_displacement": avg_disp,
        "original_lattice": orig_latt,
        "optimized_lattice": opt_latt
    }

    # 如果启用熵计算，计算并添加熵值信息
    if use_entropy:
        orig_entropy = calculate_distance_entropy(original, bin_width=entropy_bin_width, cutoff=entropy_cutoff)
        opt_entropy = calculate_distance_entropy(optimized, bin_width=entropy_bin_width, cutoff=entropy_cutoff)
        entropy_diff = opt_entropy - orig_entropy

        result.update({
            "original_entropy": orig_entropy,
            "optimized_entropy": opt_entropy,
            "entropy_difference": entropy_diff,
            "entropy_change_percentage": (entropy_diff / abs(orig_entropy)) * 100 if orig_entropy != 0 else 0
        })

    return result


def optimize_structure_task(args):
    """
    并行优化任务的工作函数
    
    Args:
        args: 包含任务参数的元组(input_file, output_file, pso_params, entropy_params)
        
    Returns:
        包含优化结果和文件信息的字典
    """
    input_file, output_file, pso_params, entropy_params = args
    
    try:
        # 为每个进程单独创建优化器实例，避免共享资源冲突
        optimizer = StructOptimizer()
        
        # 加载初始结构
        initial_structure = Structure.from_file(input_file)
        
        # 执行优化
        start_time = time.time()
        
        final_structure = optimizer.relax_pso(
            initial_structure,
            n_particles=pso_params['n_particles'],
            max_iter=pso_params['max_iter'],
            c1=pso_params['c1'],
            c2=pso_params['c2'],
            w=pso_params['w'],
            use_entropy=entropy_params['use_entropy'],
            temperature=entropy_params['temperature'],
            entropy_bin_width=entropy_params['entropy_bin_width'],
            entropy_cutoff=entropy_params['entropy_cutoff']
        )
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # 比较优化前后的结构
        comparison = compare_structures(
            initial_structure,
            final_structure,
            optimizer.calculator.model,
            use_entropy=entropy_params['use_entropy'],
            entropy_bin_width=entropy_params['entropy_bin_width'],
            entropy_cutoff=entropy_params['entropy_cutoff']
        )
        
        # 保存优化后的结构
        final_structure.to(filename=output_file)
        
        # 添加文件信息和时间信息
        comparison.update({
            'input_file': input_file,
            'output_file': output_file,
            'elapsed_time': elapsed_time,
            'formula': initial_structure.composition.formula,
            'n_atoms': len(initial_structure),
            'status': 'success'
        })
        
        return comparison
        
    except Exception as e:
        # 返回错误信息
        return {
            'input_file': input_file,
            'output_file': output_file,
            'error': str(e),
            'status': 'error'
        }


def init_worker():
    """初始化工作进程，设置信号处理"""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def process_input_path(input_path, output_dir, pso_params, entropy_params, n_workers):
    """
    处理输入路径(可以是单个文件或目录)，并行处理多个文件

    Args:
        input_path: 输入文件或目录路径
        output_dir: 输出目录
        pso_params: PSO算法参数字典
        entropy_params: 熵计算参数字典
        n_workers: 并行处理的工作进程数
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 收集需要处理的文件列表
    files_to_process = []
    
    # 如果输入是目录，收集目录下所有CIF文件
    if os.path.isdir(input_path):
        cif_files = glob.glob(os.path.join(input_path, "*.cif"))
        if not cif_files:
            print(f"警告：在目录 {input_path} 中未找到CIF文件")
            return
        
        files_to_process = cif_files
        print(f"在目录 {input_path} 中找到 {len(files_to_process)} 个CIF文件")
        
    # 如果输入是单个文件，只处理该文件
    elif os.path.isfile(input_path) and input_path.lower().endswith('.cif'):
        files_to_process = [input_path]
        print(f"将处理单个文件: {input_path}")
        
    else:
        print(f"错误: 输入路径 {input_path} 不是有效的CIF文件或目录")
        return

    # 准备并行任务参数
    tasks = []
    for cif_file in files_to_process:
        base_name = os.path.splitext(os.path.basename(cif_file))[0]
        output_file = os.path.join(output_dir, f"{base_name}_optimized.cif")
        tasks.append((cif_file, output_file, pso_params, entropy_params))
    
    # 设置工作进程数量
    if n_workers <= 0:
        n_workers = cpu_count()  # 使用所有可用CPU
    else:
        n_workers = min(n_workers, cpu_count())  # 不超过可用CPU数量
    
    print(f"\n启动并行处理，使用 {n_workers} 个工作进程...")
    print(f"待处理文件总数: {len(tasks)}")
    
    # 估计完成时间
    estimated_time_per_file = 10  # 秒，按照用户提供的信息
    estimated_total_time = (len(tasks) * estimated_time_per_file) / n_workers
    print(f"估计完成时间: {estimated_total_time:.1f} 秒 "
          f"(约 {estimated_total_time/60:.1f} 分钟 或 {estimated_total_time/3600:.2f} 小时)")
    
    results = []
    try:
        # 创建进程池并行执行任务
        with Pool(processes=n_workers, initializer=init_worker) as pool:
            # 使用tqdm显示进度条
            for result in tqdm.tqdm(
                pool.imap_unordered(optimize_structure_task, tasks),
                total=len(tasks),
                desc="优化进度"
            ):
                results.append(result)
                
                # 简要打印每个完成的任务状态
                if result['status'] == 'success':
                    base_name = os.path.basename(result['input_file'])
                    print(f"\n完成: {base_name} - "
                          f"能量变化: {result.get('energy_difference', 'N/A'):.6f} eV/atom, "
                          f"用时: {result['elapsed_time']:.1f}秒")
                else:
                    base_name = os.path.basename(result['input_file'])
                    print(f"\n错误: {base_name} - {result.get('error', '未知错误')}")
        
    except KeyboardInterrupt:
        print("\n用户中断，正在退出...")
        return
    
    # 打印处理结果摘要
    successful_results = [r for r in results if r.get('status') == 'success']
    
    print("\n==== 处理结果摘要 ====")
    print(f"成功优化: {len(successful_results)}/{len(tasks)} 个结构")
    
    if successful_results:
        # 统计性能信息
        total_time = sum(r['elapsed_time'] for r in successful_results)
        avg_time = total_time / len(successful_results)
        max_time = max(r['elapsed_time'] for r in successful_results)
        min_time = min(r['elapsed_time'] for r in successful_results)
        
        print(f"\n性能统计:")
        print(f"总处理时间: {total_time:.1f} 秒")
        print(f"平均每个结构处理时间: {avg_time:.1f} 秒")
        print(f"最长单个处理时间: {max_time:.1f} 秒")
        print(f"最短单个处理时间: {min_time:.1f} 秒")
        print(f"加速比(与串行相比): {total_time/(avg_time*len(successful_results)):.1f}x")
        
        # 找出能量改善最大的结构
        best_improvement = min(successful_results, key=lambda x: x['energy_difference'])
        print(f"\n能量改善最大的结构: {os.path.basename(best_improvement['input_file'])}")
        print(f"能量改善: {best_improvement['energy_difference']:.6f} eV/atom "
              f"({best_improvement['energy_improvement_percentage']:.2f}%)")
        
        # 如果启用熵优化，显示熵变化最大的结构
        if entropy_params['use_entropy']:
            max_entropy_change = max(successful_results, key=lambda x: x['entropy_difference'])
            print(f"\n熵增加最大的结构: {os.path.basename(max_entropy_change['input_file'])}")
            print(f"熵增加: {max_entropy_change['entropy_difference']:.6f} "
                  f"({max_entropy_change['entropy_change_percentage']:.2f}%)")
    
    # 打印错误文件列表
    error_results = [r for r in results if r.get('status') == 'error']
    if error_results:
        print(f"\n处理失败的文件 ({len(error_results)}):")
        for err in error_results:
            print(f"- {os.path.basename(err['input_file'])}: {err.get('error', '未知错误')}")


def main():
    """主函数，执行PSO结构优化流程"""

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='使用PSO方法并行优化晶体结构')
    # parser.add_argument('--input', '-i', type=str, default='./cif/mpts_52',help='输入CIF文件或包含CIF文件的目录路径')
    # parser.add_argument('--output', '-o', type=str, default='./cif/optimizer_mpts_52',help='输出目录，用于保存优化后的结构')

    parser.add_argument('--input', '-i', type=str, default='./sort_cif/test_data', help='输入CIF文件或包含CIF文件的目录路径')
    parser.add_argument('--output', '-o', type=str, default='./sort_cif/optimizer',help='输出目录，用于保存优化后的结构')

    # PSO相关参数
    parser.add_argument('--particles', '-p', type=int, default=10,help='PSO算法的粒子数量 (默认: 10)')
    parser.add_argument('--iterations', '-n', type=int, default=100,help='PSO算法的最大迭代次数 (默认: 100)')
    parser.add_argument('--c1', type=float, default=0.5,help='PSO认知参数 (默认: 0.5)')
    parser.add_argument('--c2', type=float, default=0.5,help='PSO社会参数 (默认: 0.5)')
    parser.add_argument('--w', type=float, default=0.9,help='PSO惯性权重 (默认: 0.9)')

    # 熵相关参数
    parser.add_argument('--use_entropy', default=True, action='store_true', help='是否启用构型熵优化')
    parser.add_argument('--temperature', type=float, default=0.1,  help='熵贡献权重因子 (默认: 0.1)')
    parser.add_argument('--bin_width', type=float, default=0.2, help='熵计算的分箱宽度 (默认: 0.2)')
    parser.add_argument('--cutoff', type=float, default=10.0, help='熵计算的距离截断 (默认: 10.0)')
                      
    # 并行处理参数
    parser.add_argument('--workers', '-w', type=int, default=16,help='并行处理的工作进程数 (默认: 4, 0表示使用所有可用CPU核心)')
    parser.add_argument('--batch_size', '-b', type=int, default=0, help='每批次处理的文件数量 (默认: 0, 表示一次处理所有文件)')

    args = parser.parse_args()

    # 设置PSO参数
    pso_params = {
        'n_particles': args.particles,
        'max_iter': args.iterations,
        'c1': args.c1,
        'c2': args.c2,
        'w': args.w
    }

    # 设置熵相关参数
    entropy_params = {
        'use_entropy': args.use_entropy,
        'temperature': args.temperature,
        'entropy_bin_width': args.bin_width,
        'entropy_cutoff': args.cutoff
    }

    print("===== 并行PSO晶体结构优化 =====")
    print(f"并行工作进程数: {args.workers} (0表示使用所有可用CPU)")
    print(f"PSO参数: 粒子数={pso_params['n_particles']}, "
          f"迭代次数={pso_params['max_iter']}, "
          f"c1={pso_params['c1']}, c2={pso_params['c2']}, w={pso_params['w']}")

    if entropy_params['use_entropy']:
        print(f"熵优化已启用: T={entropy_params['temperature']}, "
              f"bin_width={entropy_params['entropy_bin_width']}, "
              f"cutoff={entropy_params['entropy_cutoff']}")
    else:
        print("熵优化未启用")
    
    # 处理输入路径
    overall_start_time = time.time()
    
    # 如果指定了批次大小，分批处理文件
    if args.batch_size > 0:
        # 收集所有需要处理的文件
        all_files = []
        if os.path.isdir(args.input):
            all_files = glob.glob(os.path.join(args.input, "*.cif"))
        elif os.path.isfile(args.input) and args.input.lower().endswith('.cif'):
            all_files = [args.input]
            
        # 计算需要处理的批次数
        total_files = len(all_files)
        if total_files == 0:
            print(f"错误: 在 {args.input} 中未找到CIF文件")
            return
            
        batch_count = (total_files + args.batch_size - 1) // args.batch_size  # 向上取整
        print(f"将 {total_files} 个文件分为 {batch_count} 批处理，每批 {args.batch_size} 个文件")
        
        for i in range(batch_count):
            start_idx = i * args.batch_size
            end_idx = min((i + 1) * args.batch_size, total_files)
            batch_files = all_files[start_idx:end_idx]
            
            # 创建临时目录存储当前批次的文件
            batch_dir = os.path.join(os.path.dirname(args.input), f"temp_batch_{i}")
            os.makedirs(batch_dir, exist_ok=True)
            
            # 创建符号链接，避免复制文件
            for file in batch_files:
                link_path = os.path.join(batch_dir, os.path.basename(file))
                if os.path.exists(link_path):
                    os.remove(link_path)
                os.symlink(os.path.abspath(file), link_path)
            
            print(f"\n=== 处理第 {i+1}/{batch_count} 批 ({start_idx+1}-{end_idx}/{total_files}) ===")
            process_input_path(batch_dir, args.output, pso_params, entropy_params, args.workers)
            
            # 清理临时目录
            for file in os.listdir(batch_dir):
                os.remove(os.path.join(batch_dir, file))
            os.rmdir(batch_dir)
    else:
        # 一次处理所有文件
        process_input_path(args.input, args.output, pso_params, entropy_params, args.workers)
    
    overall_end_time = time.time()
    overall_elapsed = overall_end_time - overall_start_time
    
    print(f"\n所有优化任务完成！总耗时: {overall_elapsed:.1f} 秒 "
          f"({overall_elapsed/60:.1f} 分钟 或 {overall_elapsed/3600:.2f} 小时)")


if __name__ == "__main__":
    main()