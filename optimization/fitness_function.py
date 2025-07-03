"""
适应度函数模块 - 计算目标函数 F = E_CHGNet - T*S_config
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
from pymatgen.core import Structure

from optimization.chgnet_wrapper import chgnet_calculator


def calculateConfigurationalEntropy(particle_position: Dict[str, List[str]]) -> float:
    """
    计算构型熵 S_config
    
    基于理想溶液模型：S_config = -k_B * Σ(x_i * ln(x_i))
    其中 x_i 是元素i的摩尔分数
    
    Args:
        particle_position: 粒子的原子排列字典 {site_group: [elements]}
        
    Returns:
        float: 构型熵值 (k_B单位)
    """
    total_entropy = 0.0
    
    for site_group_name, elements in particle_position.items():
        # 统计每种元素的数量
        element_counts = {}
        for element in elements:
            element_counts[element] = element_counts.get(element, 0) + 1
        
        total_sites = len(elements)
        if total_sites == 0:
            continue
            
        # 计算该位点群的构型熵
        site_entropy = 0.0
        for count in element_counts.values():
            if count > 0:
                x_i = count / total_sites  # 摩尔分数
                site_entropy -= x_i * math.log(x_i)
        
        total_entropy += site_entropy * total_sites  # 乘以位点数量
    
    return total_entropy


def calculateMixingEntropy(particle_position: Dict[str, List[str]]) -> float:
    """
    计算混合熵（另一种计算方法）
    
    使用多项式混合熵公式：S_mix = k_B * ln(N! / (N1! * N2! * ... * Nm!))
    使用Stirling近似：ln(N!) ≈ N*ln(N) - N
    
    Args:
        particle_position: 粒子的原子排列字典
        
    Returns:
        float: 混合熵值 (k_B单位)
    """
    total_entropy = 0.0
    
    for site_group_name, elements in particle_position.items():
        # 统计每种元素的数量
        element_counts = {}
        for element in elements:
            element_counts[element] = element_counts.get(element, 0) + 1
        
        total_sites = len(elements)
        if total_sites <= 1:
            continue
            
        # 使用Stirling近似计算混合熵
        # S = ln(N!) - Σ ln(Ni!)
        entropy = total_sites * math.log(total_sites) - total_sites
        
        for count in element_counts.values():
            if count > 1:
                entropy -= count * math.log(count) - count
        
        total_entropy += entropy
    
    return total_entropy


def calculate_fitness_stage1(particle, lattice, temperature: float = 300.0, 
                           entropy_method: str = "configurational") -> float:
    """
    计算阶段一适应度函数：F = E_CHGNet - T*S_config
    
    Args:
        particle: 粒子对象
        lattice: 晶格参数 
        temperature: 温度 (K)
        entropy_method: 熵计算方法 ("configurational" 或 "mixing")
        
    Returns:
        float: 适应度值（越小越好）
        
    Raises:
        RuntimeError: 当能量计算失败时
        ValueError: 当参数无效时
    """
    if temperature <= 0:
        raise ValueError("温度必须大于0")
    
    try:
        # 生成结构
        from optimization.particle_encoding import generate_structure_from_particle_stage1
        structure = generate_structure_from_particle_stage1(particle, lattice)
        
        # 计算CHGNet能量
        energy = chgnet_calculator.calculateEnergy(structure)
        
        # 计算构型熵
        if entropy_method == "configurational":
            entropy = calculateConfigurationalEntropy(particle.position)
        elif entropy_method == "mixing":
            entropy = calculateMixingEntropy(particle.position)
        else:
            raise ValueError(f"未知的熵计算方法: {entropy_method}")
        
        # Boltzmann常数 (eV/K)
        k_B = 8.617333262145e-5  
        
        # 计算目标函数
        fitness = energy - temperature * k_B * entropy
        
        return fitness
        
    except Exception as e:
        print(f"适应度计算失败: {e}")
        return float('inf')  # 返回无穷大表示失败


def calculateEnergyDecomposition(particle, lattice, temperature: float = 300.0) -> Dict[str, float]:
    """
    计算能量分解，用于调试和分析
    
    Args:
        particle: 粒子对象
        lattice: 晶格参数
        temperature: 温度 (K)
        
    Returns:
        Dict[str, float]: 包含各个能量项的字典
    """
    try:
        from optimization.particle_encoding import generate_structure_from_particle_stage1
        structure = generate_structure_from_particle_stage1(particle, lattice)
        
        # 计算能量
        energy = chgnet_calculator.calculateEnergy(structure)
        
        # 计算熵
        config_entropy = calculateConfigurationalEntropy(particle.position)
        mixing_entropy = calculateMixingEntropy(particle.position)
        
        # Boltzmann常数
        k_B = 8.617333262145e-5
        
        # 计算各项
        t_s_config = temperature * k_B * config_entropy
        t_s_mixing = temperature * k_B * mixing_entropy
        
        fitness_config = energy - t_s_config
        fitness_mixing = energy - t_s_mixing
        
        return {
            "energy": energy,
            "configurational_entropy": config_entropy,
            "mixing_entropy": mixing_entropy,
            "T_S_configurational": t_s_config,
            "T_S_mixing": t_s_mixing,
            "fitness_configurational": fitness_config,
            "fitness_mixing": fitness_mixing,
            "temperature": temperature
        }
        
    except Exception as e:
        print(f"能量分解计算失败: {e}")
        return {
            "energy": float('inf'),
            "configurational_entropy": 0.0,
            "mixing_entropy": 0.0,
            "T_S_configurational": 0.0,
            "T_S_mixing": 0.0,
            "fitness_configurational": float('inf'),
            "fitness_mixing": float('inf'),
            "temperature": temperature
        }


def validateFitnessFunction(test_structures: List[Structure], 
                          temperature: float = 300.0) -> List[Dict[str, float]]:
    """
    验证适应度函数的计算
    
    Args:
        test_structures: 测试结构列表
        temperature: 温度
        
    Returns:
        List[Dict[str, float]]: 每个结构的适应度分解
    """
    results = []
    
    for i, structure in enumerate(test_structures):
        print(f"验证结构 {i+1}/{len(test_structures)}")
        
        try:
            # 计算能量
            energy = chgnet_calculator.calculateEnergy(structure)
            
            # 从结构重构粒子位置（简化版本）
            # 这里假设只有一个位点群
            elements = [str(site.specie) for site in structure.sites]
            particle_position = {"default": elements}
            
            # 计算熵
            config_entropy = calculateConfigurationalEntropy(particle_position)
            
            # 计算适应度
            k_B = 8.617333262145e-5
            fitness = energy - temperature * k_B * config_entropy
            
            result = {
                "structure_index": i,
                "energy": energy,
                "configurational_entropy": config_entropy,
                "T_S": temperature * k_B * config_entropy,
                "fitness": fitness,
                "formula": structure.formula
            }
            
            results.append(result)
            print(f"  能量: {energy:.6f} eV")
            print(f"  构型熵: {config_entropy:.6f} k_B")
            print(f"  适应度: {fitness:.6f} eV")
            
        except Exception as e:
            print(f"  计算失败: {e}")
            results.append({
                "structure_index": i,
                "energy": float('inf'),
                "configurational_entropy": 0.0,
                "T_S": 0.0,
                "fitness": float('inf'),
                "formula": structure.formula,
                "error": str(e)
            })
    
    return results