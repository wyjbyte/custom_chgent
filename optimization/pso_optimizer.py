"""
PSO优化器主模块 - 实现完整的粒子群优化算法
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pymatgen.core import Structure
from pymatgen.core.lattice import Lattice

from optimization.chgnet_wrapper import chgnet_calculator
from optimization.particle_encoding import ParticleSwarm, HighEntropySiteGroup, Particle
from optimization.fitness_function import calculate_fitness_stage1, calculateEnergyDecomposition


class PSOOptimizer:
    """
    PSO优化器类 - 实现阶段一的完整PSO优化循环
    """
    
    def __init__(self, lattice: Lattice, site_groups: List[HighEntropySiteGroup],
                 swarm_size: int = 20, max_iterations: int = 100,
                 temperature: float = 300.0, **kwargs):
        """
        初始化PSO优化器
        
        Args:
            lattice: 晶格参数
            site_groups: 位点群列表
            swarm_size: 粒子群大小
            max_iterations: 最大迭代次数
            temperature: 温度 (K)
            **kwargs: 其他PSO参数
        """
        self.lattice = lattice
        self.site_groups = site_groups
        self.swarm_size = swarm_size
        self.max_iterations = max_iterations
        self.temperature = temperature
        
        # PSO参数
        self.inertia_weight = kwargs.get('inertia_weight', 0.5)
        self.cognitive_factor = kwargs.get('cognitive_factor', 1.0)
        self.social_factor = kwargs.get('social_factor', 1.0)
        self.mutation_rate = kwargs.get('mutation_rate', 0.1)
        
        # 收敛控制
        self.convergence_threshold = kwargs.get('convergence_threshold', 1e-6)
        self.patience = kwargs.get('patience', 10)  # 连续多少代无改善停止
        
        # 初始化粒子群
        self.swarm = ParticleSwarm(site_groups, swarm_size)
        
        # 优化历史
        self.optimization_history = []
        self.current_iteration = 0
        self.best_fitness_history = []
        self.convergence_counter = 0
        
        # 验证CHGNet可用性
        if not chgnet_calculator.isAvailable():
            raise RuntimeError("CHGNet计算器不可用，请检查环境配置")
        
        print(f"PSO优化器初始化完成:")
        print(f"  粒子群大小: {swarm_size}")
        print(f"  最大迭代次数: {max_iterations}")
        print(f"  温度: {temperature} K")
        print(f"  位点群数量: {len(site_groups)}")
    
    def initializeParticles(self):
        """初始化阶段 - 计算所有粒子的初始适应度"""
        print("\n=== 初始化粒子群 ===")
        
        for i, particle in enumerate(self.swarm.particles):
            print(f"初始化粒子 {i+1}/{self.swarm_size}...", end=" ")
            
            try:
                fitness = calculate_fitness_stage1(particle, self.lattice, self.temperature)
                particle.fitness = fitness
                particle.updateBest(fitness)
                print(f"适应度: {fitness:.6f}")
                
            except Exception as e:
                print(f"失败: {e}")
                particle.fitness = float('inf')
                particle.updateBest(float('inf'))
        
        # 更新全局最优
        self.swarm.updateGlobalBest()
        
        if self.swarm.global_best_fitness < float('inf'):
            print(f"\n初始全局最优适应度: {self.swarm.global_best_fitness:.6f}")
        else:
            print("\n警告: 所有粒子初始化失败")
    
    def optimize(self) -> Dict[str, Any]:
        """
        执行完整的PSO优化过程
        
        Returns:
            Dict[str, Any]: 优化结果
        """
        print("\n" + "="*50)
        print("开始PSO优化")
        print("="*50)
        
        start_time = time.time()
        
        # 初始化粒子群
        self.initializeParticles()
        
        # 记录初始状态
        self._recordIteration()
        
        # 主优化循环
        for iteration in range(self.max_iterations):
            self.current_iteration = iteration + 1
            
            print(f"\n=== 迭代 {self.current_iteration}/{self.max_iterations} ===")
            
            # 更新粒子
            self._updateParticles()
            
            # 评估适应度
            self._evaluateFitness()
            
            # 更新全局最优
            previous_best = self.swarm.global_best_fitness
            self.swarm.updateGlobalBest()
            
            # 记录迭代信息
            self._recordIteration()
            
            # 输出进度
            self._printProgress()
            
            # 检查收敛
            if self._checkConvergence(previous_best):
                print(f"\n在第 {self.current_iteration} 代达到收敛条件，提前停止")
                break
        
        # 生成最终结果
        optimization_time = time.time() - start_time
        result = self._generateFinalResult(optimization_time)
        
        print("\n" + "="*50)
        print("PSO优化完成")
        print("="*50)
        
        return result
    
    def _updateParticles(self):
        """更新所有粒子的速度和位置"""
        for particle in self.swarm.particles:
            # 更新速度
            particle.updateVelocity(
                self.swarm.global_best_position,
                w=self.inertia_weight,
                c1=self.cognitive_factor,
                c2=self.social_factor
            )
            
            # 更新位置
            particle.updatePosition()
            
            # 添加随机变异
            particle.addRandomMutation(self.mutation_rate)
    
    def _evaluateFitness(self):
        """评估所有粒子的适应度"""
        for i, particle in enumerate(self.swarm.particles):
            try:
                fitness = calculate_fitness_stage1(particle, self.lattice, self.temperature)
                particle.fitness = fitness
                particle.updateBest(fitness)
                
            except Exception as e:
                print(f"粒子 {i+1} 适应度计算失败: {e}")
                particle.fitness = float('inf')
    
    def _recordIteration(self):
        """记录当前迭代的信息"""
        # 计算统计信息
        fitnesses = [p.fitness for p in self.swarm.particles if p.fitness < float('inf')]
        
        if fitnesses:
            avg_fitness = np.mean(fitnesses)
            std_fitness = np.std(fitnesses)
            min_fitness = min(fitnesses)
            max_fitness = max(fitnesses)
        else:
            avg_fitness = std_fitness = min_fitness = max_fitness = float('inf')
        
        iteration_info = {
            "iteration": self.current_iteration,
            "global_best_fitness": self.swarm.global_best_fitness,
            "average_fitness": avg_fitness,
            "std_fitness": std_fitness,
            "min_fitness": min_fitness,
            "max_fitness": max_fitness,
            "valid_particles": len(fitnesses),
            "temperature": self.temperature
        }
        
        self.optimization_history.append(iteration_info)
        self.best_fitness_history.append(self.swarm.global_best_fitness)
    
    def _printProgress(self):
        """打印优化进度"""
        current_info = self.optimization_history[-1]
        
        print(f"全局最优适应度: {current_info['global_best_fitness']:.6f}")
        print(f"平均适应度: {current_info['average_fitness']:.6f}")
        print(f"有效粒子数: {current_info['valid_particles']}/{self.swarm_size}")
        
        # 打印能量分解信息
        if self.swarm.global_best_particle:
            decomposition = calculateEnergyDecomposition(
                self.swarm.global_best_particle, 
                self.lattice, 
                self.temperature
            )
            print(f"  能量: {decomposition['energy']:.6f} eV")
            print(f"  构型熵: {decomposition['configurational_entropy']:.6f} k_B")
            print(f"  T*S: {decomposition['T_S_configurational']:.6f} eV")
    
    def _checkConvergence(self, previous_best: float) -> bool:
        """检查收敛条件"""
        current_best = self.swarm.global_best_fitness
        
        # 检查适应度改善
        if abs(current_best - previous_best) < self.convergence_threshold:
            self.convergence_counter += 1
        else:
            self.convergence_counter = 0
        
        # 达到patience次数无改善则收敛
        return self.convergence_counter >= self.patience
    
    def _generateFinalResult(self, optimization_time: float) -> Dict[str, Any]:
        """生成最终优化结果"""
        if self.swarm.global_best_particle is None:
            raise RuntimeError("优化失败：未找到有效解")
        
        # 获取最优结构
        best_structure = self.swarm.getGlobalBestStructure(self.lattice)
        
        # 计算最终能量分解
        energy_decomposition = calculateEnergyDecomposition(
            self.swarm.global_best_particle,
            self.lattice,
            self.temperature
        )
        
        result = {
            "optimization_successful": True,
            "iterations_completed": self.current_iteration,
            "optimization_time": optimization_time,
            "global_best_fitness": self.swarm.global_best_fitness,
            "best_structure": best_structure,
            "best_particle_position": self.swarm.global_best_position,
            "energy_decomposition": energy_decomposition,
            "optimization_history": self.optimization_history,
            "parameters": {
                "swarm_size": self.swarm_size,
                "max_iterations": self.max_iterations,
                "temperature": self.temperature,
                "inertia_weight": self.inertia_weight,
                "cognitive_factor": self.cognitive_factor,
                "social_factor": self.social_factor,
                "mutation_rate": self.mutation_rate
            }
        }
        
        return result
    
    def saveResult(self, result: Dict[str, Any], output_dir: str = "pso_results"):
        """
        保存优化结果
        
        Args:
            result: 优化结果字典
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 保存最优结构为CIF文件
        cif_file = output_path / "best_structure.cif"
        result["best_structure"].to(filename=str(cif_file))
        print(f"最优结构已保存至: {cif_file}")
        
        # 保存结果JSON（排除Structure对象）
        json_result = result.copy()
        del json_result["best_structure"]  # Structure不能直接序列化
        
        json_file = output_path / "optimization_result.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_result, f, indent=2, ensure_ascii=False, default=str)
        print(f"优化结果已保存至: {json_file}")
        
        # 保存最优粒子编码
        particle_file = output_path / "best_particle.json"
        with open(particle_file, 'w', encoding='utf-8') as f:
            json.dump(result["best_particle_position"], f, indent=2, ensure_ascii=False)
        print(f"最优粒子编码已保存至: {particle_file}")
        
        # 打印最终结果摘要
        print(f"\n{'='*50}")
        print("优化结果摘要")
        print(f"{'='*50}")
        print(f"迭代次数: {result['iterations_completed']}")
        print(f"优化时间: {result['optimization_time']:.2f} 秒")
        print(f"最优适应度: {result['global_best_fitness']:.6f}")
        print(f"最优能量: {result['energy_decomposition']['energy']:.6f} eV")
        print(f"构型熵: {result['energy_decomposition']['configurational_entropy']:.6f} k_B")
        print(f"T*S项: {result['energy_decomposition']['T_S_configurational']:.6f} eV")
        print(f"最优结构式: {result['best_structure'].formula}")


def createSimpleTestCase() -> Tuple[Lattice, List[HighEntropySiteGroup]]:
    """
    创建简单的测试用例
    
    Returns:
        Tuple[Lattice, List[HighEntropySiteGroup]]: 晶格和位点群
    """
    # 创建简单立方晶格
    lattice = Lattice.cubic(4.0)
    
    # 创建一个位点群，包含8个角位点
    positions = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.5, 0.0, 0.0]),
        np.array([0.0, 0.5, 0.0]),
        np.array([0.0, 0.0, 0.5]),
        np.array([0.5, 0.5, 0.0]),
        np.array([0.5, 0.0, 0.5]),
        np.array([0.0, 0.5, 0.5]),
        np.array([0.5, 0.5, 0.5])
    ]
    
    # 定义高熵合金成分（例如：Fe, Ni, Cr, Co各2个原子）
    element_counts = {"Fe": 2, "Ni": 2, "Cr": 2, "Co": 2}
    
    site_group = HighEntropySiteGroup("corner_sites", positions, element_counts)
    
    return lattice, [site_group]