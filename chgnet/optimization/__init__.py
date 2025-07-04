"""
高熵结构优化 - PSO优化模块

这个模块包含了粒子群优化算法用于高熵合金结构优化的核心组件
"""

from chgnet.optimization.particle import PSOParticle, initialize_population
from chgnet.optimization.structure_ops import generate_structure_from_particle_stage1
from chgnet.optimization.entropy import calculate_configurational_entropy

# 延迟导入CHGNet相关功能以避免torch依赖问题
def get_fitness_functions():
    """延迟导入适应度函数以避免torch依赖"""
    from chgnet.optimization.fitness import calculate_fitness_stage1, evaluate_particle_fitness, create_chgnet_calculator
    return calculate_fitness_stage1, evaluate_particle_fitness, create_chgnet_calculator

__all__ = [
    "PSOParticle",
    "initialize_population", 
    "generate_structure_from_particle_stage1",
    "calculate_configurational_entropy",
    "get_fitness_functions"
]