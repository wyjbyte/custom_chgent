"""
粒子编码系统 - 用于表示原子排列的粒子
实现基于交换的离散PSO算法
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pymatgen.core import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.core.sites import PeriodicSite


class SiteGroup:
    """
    位点群类 - 表示具有相同Wyckoff位置的原子位点集合
    """
    
    def __init__(self, name: str, positions: List[np.ndarray], allowed_elements: List[str]):
        """
        初始化位点群
        
        Args:
            name: 位点群名称
            positions: 位点坐标列表（分数坐标）
            allowed_elements: 该位点群允许的元素类型
        """
        self.name = name
        self.positions = positions
        self.allowed_elements = allowed_elements
        self.num_sites = len(positions)
        
        # 验证元素数量与位点数量的关系
        if sum(self.element_counts.values()) != self.num_sites:
            raise ValueError("元素总数必须等于位点数")
    
    @property 
    def element_counts(self) -> Dict[str, int]:
        """获取每种元素的数量（需要子类实现）"""
        raise NotImplementedError("子类必须实现element_counts属性")


class Particle:
    """
    粒子类 - 表示一个原子排列方案
    粒子的"位置"是原子在各位点群中的排列
    """
    
    def __init__(self, site_groups: List[SiteGroup]):
        """
        初始化粒子
        
        Args:
            site_groups: 位点群列表
        """
        self.site_groups = site_groups
        self.position = {}  # 存储每个位点群的原子排列
        self.velocity = {}  # 存储每个位点群的"速度"（交换倾向）
        self.fitness = float('inf')
        self.best_position = None
        self.best_fitness = float('inf')
        
        # 初始化随机位置
        self._initializeRandomPosition()
        
    def _initializeRandomPosition(self):
        """初始化随机原子排列"""
        for group in self.site_groups:
            # 为每个位点群创建随机的原子排列
            elements = []
            for element, count in group.element_counts.items():
                elements.extend([element] * count)
            
            # 随机排列
            random.shuffle(elements)
            self.position[group.name] = elements
            
            # 初始化速度为空交换序列
            self.velocity[group.name] = []
    
    def updateVelocity(self, global_best_position: Dict[str, List[str]], 
                      w: float = 0.5, c1: float = 1.0, c2: float = 1.0):
        """
        更新粒子速度（基于交换操作）
        
        Args:
            global_best_position: 全局最优位置
            w: 惯性权重
            c1: 个人最优影响因子
            c2: 全局最优影响因子
        """
        for group_name in self.position.keys():
            current_pos = self.position[group_name]
            personal_best = self.best_position[group_name] if self.best_position else current_pos
            global_best = global_best_position[group_name]
            
            # 计算需要进行的交换操作
            swaps = []
            
            # 朝向个人最优的交换
            if random.random() < c1:
                swaps.extend(self._calculateSwapsToTarget(current_pos, personal_best))
            
            # 朝向全局最优的交换  
            if random.random() < c2:
                swaps.extend(self._calculateSwapsToTarget(current_pos, global_best))
            
            # 应用惯性（保留部分之前的速度）
            if random.random() < w:
                swaps.extend(self.velocity[group_name])
            
            # 限制交换操作数量避免过度变化
            max_swaps = min(len(swaps), 3)
            self.velocity[group_name] = swaps[:max_swaps]
    
    def _calculateSwapsToTarget(self, current: List[str], target: List[str]) -> List[Tuple[int, int]]:
        """
        计算从current到target需要的交换操作序列
        
        Args:
            current: 当前排列
            target: 目标排列
            
        Returns:
            List[Tuple[int, int]]: 交换操作列表（索引对）
        """
        swaps = []
        temp = current.copy()
        
        for i in range(len(temp)):
            if temp[i] != target[i]:
                # 找到目标元素的位置
                for j in range(i + 1, len(temp)):
                    if temp[j] == target[i]:
                        swaps.append((i, j))
                        temp[i], temp[j] = temp[j], temp[i]
                        break
        
        return swaps
    
    def updatePosition(self):
        """根据速度更新位置（执行交换操作）"""
        for group_name in self.position.keys():
            current_pos = self.position[group_name].copy()
            
            # 执行速度中定义的交换操作
            for i, j in self.velocity[group_name]:
                if 0 <= i < len(current_pos) and 0 <= j < len(current_pos):
                    current_pos[i], current_pos[j] = current_pos[j], current_pos[i]
            
            self.position[group_name] = current_pos
    
    def addRandomMutation(self, mutation_rate: float = 0.1):
        """
        添加随机变异以增加多样性
        
        Args:
            mutation_rate: 变异概率
        """
        for group_name in self.position.keys():
            if random.random() < mutation_rate:
                pos = self.position[group_name]
                if len(pos) > 1:
                    # 随机交换两个位置
                    i, j = random.sample(range(len(pos)), 2)
                    pos[i], pos[j] = pos[j], pos[i]
    
    def updateBest(self, fitness: float):
        """
        更新个人最优记录
        
        Args:
            fitness: 当前适应度
        """
        if fitness < self.best_fitness:
            self.best_fitness = fitness
            self.best_position = {name: pos.copy() for name, pos in self.position.items()}
    
    def copy(self) -> 'Particle':
        """创建粒子副本"""
        new_particle = Particle(self.site_groups)
        new_particle.position = {name: pos.copy() for name, pos in self.position.items()}
        new_particle.velocity = {name: vel.copy() for name, vel in self.velocity.items()}
        new_particle.fitness = self.fitness
        new_particle.best_fitness = self.best_fitness
        if self.best_position:
            new_particle.best_position = {name: pos.copy() for name, pos in self.best_position.items()}
        return new_particle


class HighEntropySiteGroup(SiteGroup):
    """
    高熵合金位点群实现
    """
    
    def __init__(self, name: str, positions: List[np.ndarray], 
                 element_counts: Dict[str, int]):
        """
        初始化高熵位点群
        
        Args:
            name: 位点群名称
            positions: 位点坐标列表
            element_counts: 每种元素的数量
        """
        self._element_counts = element_counts
        allowed_elements = list(element_counts.keys())
        super().__init__(name, positions, allowed_elements)
    
    @property
    def element_counts(self) -> Dict[str, int]:
        """获取每种元素的数量"""
        return self._element_counts


def generate_structure_from_particle_stage1(particle: Particle, 
                                           lattice: Lattice) -> Structure:
    """
    从粒子编码生成pymatgen Structure
    
    Args:
        particle: 粒子对象
        lattice: 晶格参数
        
    Returns:
        Structure: 生成的晶体结构
    """
    species = []
    coords = []
    
    for group in particle.site_groups:
        group_elements = particle.position[group.name]
        group_positions = group.positions
        
        if len(group_elements) != len(group_positions):
            raise ValueError(f"位点群 {group.name} 的元素数量与位点数量不匹配")
        
        for element, position in zip(group_elements, group_positions):
            species.append(element)
            coords.append(position)
    
    return Structure(lattice, species, coords)


class ParticleSwarm:
    """粒子群类 - 管理整个粒子群"""
    
    def __init__(self, site_groups: List[SiteGroup], swarm_size: int = 20):
        """
        初始化粒子群
        
        Args:
            site_groups: 位点群列表
            swarm_size: 粒子群大小
        """
        self.site_groups = site_groups
        self.swarm_size = swarm_size
        self.particles = [Particle(site_groups) for _ in range(swarm_size)]
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.global_best_particle = None
    
    def updateGlobalBest(self):
        """更新全局最优"""
        for particle in self.particles:
            if particle.fitness < self.global_best_fitness:
                self.global_best_fitness = particle.fitness
                self.global_best_position = {name: pos.copy() 
                                           for name, pos in particle.position.items()}
                self.global_best_particle = particle.copy()
    
    def getGlobalBestStructure(self, lattice: Lattice) -> Structure:
        """获取全局最优结构"""
        if self.global_best_particle is None:
            raise ValueError("还没有找到全局最优解")
        return generate_structure_from_particle_stage1(self.global_best_particle, lattice)