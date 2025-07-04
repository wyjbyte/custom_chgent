"""
PSO粒子表示和种群管理模块

实现阶段一PSO的粒子数据结构和种群初始化功能
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Any
import numpy as np
import copy


class PSOParticle:
    """
    PSO粒子类，用于表示晶体结构中原子在可交换位点群内的排列
    
    粒子编码原子在位点群内的具体排列，确保每个位点群内的元素种类和数量保持不变
    
    Attributes:
        site_groups_arrangement (Dict[str, np.ndarray]): 位点群排列字典
            键为位点群名称，值为该位点群内的原子排列数组
            数组索引对应位点索引，数组值对应原子种类（用原子序数表示）
        site_groups_definition (Dict[str, Dict]): 位点群定义信息
        fitness (float): 粒子适应度值
        velocity (Dict[str, np.ndarray]): 粒子速度（用于PSO更新）
        best_position (Dict[str, np.ndarray]): 粒子历史最佳位置
        best_fitness (float): 粒子历史最佳适应度
    """
    
    def __init__(self, site_groups_definition: Dict[str, Dict]):
        """
        初始化PSO粒子
        
        Args:
            site_groups_definition (Dict[str, Dict]): 位点群定义
                格式: {
                    "group_name": {
                        "site_indices": [0, 1, 2, ...],  # 该群组包含的位点索引
                        "elements": [6, 6, 14, 14, ...],  # 该群组包含的原子种类（原子序数）
                        "element_counts": {"C": 2, "Si": 2}  # 元素计数
                    }
                }
        """
        self.site_groups_definition = copy.deepcopy(site_groups_definition)
        self.site_groups_arrangement = {}
        self.fitness = float('inf')
        self.velocity = {}
        self.best_position = {}
        self.best_fitness = float('inf')
        
        # 为每个位点群初始化随机排列
        self._initialize_random_arrangement()
        
    def _initialize_random_arrangement(self):
        """为每个位点群初始化随机的原子排列"""
        for group_name, group_info in self.site_groups_definition.items():
            elements = group_info["elements"]
            # 创建随机排列
            arrangement = np.array(elements)
            np.random.shuffle(arrangement)
            self.site_groups_arrangement[group_name] = arrangement
            
    def get_arrangement_for_group(self, group_name: str) -> np.ndarray:
        """
        获取指定位点群的原子排列
        
        Args:
            group_name (str): 位点群名称
            
        Returns:
            np.ndarray: 该位点群的原子排列数组
        """
        return self.site_groups_arrangement.get(group_name, np.array([]))
    
    def set_arrangement_for_group(self, group_name: str, arrangement: np.ndarray):
        """
        设置指定位点群的原子排列
        
        Args:
            group_name (str): 位点群名称
            arrangement (np.ndarray): 新的原子排列
        """
        if not self._validate_arrangement(group_name, arrangement):
            raise ValueError(f"无效的排列：不满足位点群 {group_name} 的约束条件")
        self.site_groups_arrangement[group_name] = arrangement.copy()
    
    def _validate_arrangement(self, group_name: str, arrangement: np.ndarray) -> bool:
        """
        验证给定排列是否满足位点群约束
        
        Args:
            group_name (str): 位点群名称
            arrangement (np.ndarray): 待验证的排列
            
        Returns:
            bool: 是否满足约束
        """
        if group_name not in self.site_groups_definition:
            return False
            
        expected_elements = sorted(self.site_groups_definition[group_name]["elements"])
        actual_elements = sorted(arrangement.tolist())
        
        return expected_elements == actual_elements
    
    def swap_atoms_in_group(self, group_name: str, index1: int, index2: int):
        """
        在指定位点群内交换两个原子的位置
        
        Args:
            group_name (str): 位点群名称
            index1, index2 (int): 要交换的原子在该位点群内的索引
        """
        if group_name not in self.site_groups_arrangement:
            raise ValueError(f"位点群 {group_name} 不存在")
            
        arrangement = self.site_groups_arrangement[group_name]
        if index1 >= len(arrangement) or index2 >= len(arrangement):
            raise ValueError("索引超出范围")
            
        arrangement[index1], arrangement[index2] = arrangement[index2], arrangement[index1]
    
    def copy(self) -> 'PSOParticle':
        """创建粒子的深拷贝"""
        new_particle = PSOParticle(self.site_groups_definition)
        new_particle.site_groups_arrangement = {
            k: v.copy() for k, v in self.site_groups_arrangement.items()
        }
        new_particle.fitness = self.fitness
        new_particle.best_fitness = self.best_fitness
        new_particle.best_position = {
            k: v.copy() for k, v in self.best_position.items()
        }
        return new_particle
    
    def update_best_position(self):
        """更新粒子的历史最佳位置"""
        if self.fitness < self.best_fitness:
            self.best_fitness = self.fitness
            self.best_position = {
                k: v.copy() for k, v in self.site_groups_arrangement.items()
            }
    
    def get_total_sites(self) -> int:
        """获取所有位点群的总位点数"""
        return sum(len(arr) for arr in self.site_groups_arrangement.values())
    
    def get_arrangement_summary(self) -> Dict[str, Dict]:
        """
        获取排列的摘要信息
        
        Returns:
            Dict: 包含每个位点群排列信息的字典
        """
        summary = {}
        for group_name, arrangement in self.site_groups_arrangement.items():
            unique, counts = np.unique(arrangement, return_counts=True)
            summary[group_name] = {
                "arrangement": arrangement.tolist(),
                "element_distribution": dict(zip(unique.tolist(), counts.tolist()))
            }
        return summary
    
    def __str__(self) -> str:
        """粒子的字符串表示"""
        return f"PSOParticle(fitness={self.fitness:.4f}, groups={list(self.site_groups_arrangement.keys())})"
    
    def __repr__(self) -> str:
        return self.__str__()


# 种群管理函数

def initialize_population(
    population_size: int,
    site_groups_definition: Dict[str, Dict],
    random_seed: int = None
) -> List[PSOParticle]:
    """
    初始化PSO粒子种群
    
    Args:
        population_size (int): 种群大小
        site_groups_definition (Dict[str, Dict]): 位点群定义
        random_seed (int, optional): 随机种子
        
    Returns:
        List[PSOParticle]: 初始化的粒子种群列表
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    population = []
    
    for i in range(population_size):
        particle = PSOParticle(site_groups_definition)
        
        # 应用不同的随机化策略增加多样性
        for group_name in particle.site_groups_arrangement.keys():
            if i % 3 == 0:
                _randomize_group_arrangement(particle, group_name)
            elif i % 3 == 1:
                _local_shuffle_group(particle, group_name)
            else:
                _segment_shuffle_group(particle, group_name)
        
        population.append(particle)
    
    return population


def _randomize_group_arrangement(particle: PSOParticle, group_name: str):
    """对位点群进行完全随机排列"""
    arrangement = particle.site_groups_arrangement[group_name].copy()
    np.random.shuffle(arrangement)
    particle.site_groups_arrangement[group_name] = arrangement


def _local_shuffle_group(particle: PSOParticle, group_name: str):
    """对位点群进行局部随机交换"""
    arrangement = particle.site_groups_arrangement[group_name].copy()
    n = len(arrangement)
    
    num_swaps = max(1, n // 3)
    for _ in range(num_swaps):
        i = np.random.randint(0, n)
        j = np.random.randint(max(0, i-2), min(n, i+3))
        arrangement[i], arrangement[j] = arrangement[j], arrangement[i]
    
    particle.site_groups_arrangement[group_name] = arrangement


def _segment_shuffle_group(particle: PSOParticle, group_name: str):
    """对位点群进行分段随机排列"""
    arrangement = particle.site_groups_arrangement[group_name].copy()
    n = len(arrangement)
    
    if n <= 2:
        np.random.shuffle(arrangement)
    else:
        num_segments = np.random.randint(2, min(4, n+1))
        segment_boundaries = np.sort(np.random.choice(range(1, n), num_segments-1, replace=False))
        segment_boundaries = np.concatenate([[0], segment_boundaries, [n]])
        
        for i in range(len(segment_boundaries) - 1):
            start, end = segment_boundaries[i], segment_boundaries[i+1]
            segment = arrangement[start:end].copy()
            np.random.shuffle(segment)
            arrangement[start:end] = segment
    
    particle.site_groups_arrangement[group_name] = arrangement