"""
配置熵计算模块

实现高熵合金结构的配置熵计算核心功能
"""

from __future__ import annotations
from typing import Dict, Optional
import numpy as np
import math
from collections import Counter
from pymatgen.core import Structure
from chgnet.optimization.particle import PSOParticle

# 玻尔兹曼常数 (eV/K)
K_B = 8.617333262145e-5


def calculate_configurational_entropy(
    particle: PSOParticle = None,
    structure: Structure = None,
    site_groups_definition: Dict[str, Dict] = None,
    temperature: float = 300.0
) -> float:
    """
    计算配置熵 S_config
    
    Args:
        particle (PSOParticle, optional): PSO粒子
        structure (Structure, optional): 晶体结构  
        site_groups_definition (Dict[str, Dict], optional): 位点群定义
        temperature (float): 温度 (K)
        
    Returns:
        float: 配置熵值 (eV/K)
    """
    if particle is not None:
        return _calculate_entropy_from_particle(particle, site_groups_definition)
    elif structure is not None:
        return _calculate_entropy_from_structure(structure, site_groups_definition)
    else:
        raise ValueError("必须提供particle或structure中的一个")


def _calculate_entropy_from_particle(
    particle: PSOParticle,
    site_groups_definition: Dict[str, Dict]
) -> float:
    """从PSO粒子计算配置熵"""
    total_entropy = 0.0
    
    for group_name, arrangement in particle.site_groups_arrangement.items():
        group_entropy = _calculate_group_configurational_entropy(arrangement)
        total_entropy += group_entropy
    
    return total_entropy


def _calculate_entropy_from_structure(
    structure: Structure,
    site_groups_definition: Dict[str, Dict]
) -> float:
    """从晶体结构计算配置熵"""
    total_entropy = 0.0
    
    for group_name, group_info in site_groups_definition.items():
        site_indices = group_info["site_indices"]
        
        # 提取该位点群的原子种类
        group_elements = []
        for site_idx in site_indices:
            element = structure[site_idx].specie.Z
            group_elements.append(element)
        
        arrangement = np.array(group_elements)
        group_entropy = _calculate_group_configurational_entropy(arrangement)
        total_entropy += group_entropy
    
    return total_entropy


def _calculate_group_configurational_entropy(arrangement: np.ndarray) -> float:
    """
    计算单个位点群的配置熵
    
    使用混合熵公式：S = -k_B * Σ(x_i * ln(x_i))
    """
    if len(arrangement) == 0:
        return 0.0
    
    # 统计每种元素的数量
    element_counts = Counter(arrangement)
    total_atoms = len(arrangement)
    
    entropy = 0.0
    for count in element_counts.values():
        if count > 0:
            x_i = count / total_atoms
            entropy -= x_i * math.log(x_i)
    
    return entropy * K_B