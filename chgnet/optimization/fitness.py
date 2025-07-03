"""
适应度函数模块

实现阶段一PSO的适应度评估核心功能，目标函数为 F = E_CHGNet - T * S_config
"""

from __future__ import annotations
from typing import Dict, Optional
import numpy as np
from pymatgen.core import Structure
from chgnet.model import CHGNet
from chgnet.model.dynamics import CHGNetCalculator
from chgnet.optimization.particle import PSOParticle
from chgnet.optimization.structure_ops import generate_structure_from_particle_stage1
from chgnet.optimization.entropy import calculate_configurational_entropy


def calculate_fitness_stage1(
    structure: Structure,
    T: float,
    site_groups_definition: Dict[str, Dict],
    chgnet_calculator: CHGNetCalculator,
    particle: Optional[PSOParticle] = None
) -> float:
    """
    计算阶段一适应度函数值：F = E_CHGNet - T * S_config
    
    Args:
        structure (Structure): 由PSO粒子生成的当前晶体结构
        T (float): 温度超参数 (K)
        site_groups_definition (Dict[str, Dict]): 位点群定义
        chgnet_calculator (CHGNetCalculator): CHGNet计算器实例
        particle (PSOParticle, optional): 对应的PSO粒子
        
    Returns:
        float: 适应度值 F (eV)
    """
    try:
        # 计算CHGNet能量
        energy_chgnet = _calculate_chgnet_energy(structure, chgnet_calculator)
        
        # 计算配置熵
        s_config = calculate_configurational_entropy(
            particle=particle,
            structure=structure,
            site_groups_definition=site_groups_definition
        )
        
        # 计算适应度：F = E - T*S
        fitness = energy_chgnet - T * s_config
        
        return fitness
        
    except Exception as e:
        print(f"适应度计算出错: {e}")
        return float('inf')


def _calculate_chgnet_energy(
    structure: Structure,
    chgnet_calculator: CHGNetCalculator
) -> float:
    """
    使用CHGNet计算结构能量
    
    Args:
        structure (Structure): 晶体结构
        chgnet_calculator (CHGNetCalculator): CHGNet计算器
        
    Returns:
        float: 结构能量 (eV)
    """
    try:
        # CHGNet需要ASE Atoms对象
        from chgnet.optimization.structure_ops import structure_to_ase_atoms
        atoms = structure_to_ase_atoms(structure)
        
        # 使用CHGNet计算能量
        atoms.calc = chgnet_calculator
        energy = atoms.get_potential_energy()
        
        return float(energy)
        
    except Exception as e:
        print(f"CHGNet能量计算出错: {e}")
        return float('inf')


def evaluate_particle_fitness(
    particle: PSOParticle,
    base_structure: Structure,
    site_groups_definition: Dict[str, Dict],
    T: float,
    chgnet_calculator: CHGNetCalculator
) -> float:
    """
    评估单个PSO粒子的适应度
    
    Args:
        particle (PSOParticle): PSO粒子
        base_structure (Structure): 基础结构
        site_groups_definition (Dict[str, Dict]): 位点群定义
        T (float): 温度参数
        chgnet_calculator (CHGNetCalculator): CHGNet计算器
        
    Returns:
        float: 适应度值
    """
    try:
        # 从粒子生成结构
        structure = generate_structure_from_particle_stage1(
            base_structure, particle, site_groups_definition
        )
        
        # 计算适应度
        fitness = calculate_fitness_stage1(
            structure, T, site_groups_definition, chgnet_calculator, particle
        )
        
        # 更新粒子适应度
        particle.fitness = fitness
        particle.update_best_position()
        
        return fitness
        
    except Exception as e:
        print(f"粒子适应度评估出错: {e}")
        particle.fitness = float('inf')
        return float('inf')


def create_chgnet_calculator(
    model_path: Optional[str] = None,
    device: str = "cpu"
) -> CHGNetCalculator:
    """
    创建CHGNet计算器实例
    
    Args:
        model_path (str, optional): 预训练模型路径，None则使用默认模型
        device (str): 计算设备 ("cpu" 或 "cuda")
        
    Returns:
        CHGNetCalculator: CHGNet计算器实例
    """
    if model_path is None:
        model = CHGNet.load()
    else:
        model = CHGNet.load(path=model_path)
    
    calculator = CHGNetCalculator(model=model, use_device=device)
    return calculator