"""
高熵结构优化模块

包含PSO优化算法和CHGNet集成的相关功能
"""
from optimization.chgnet_wrapper import CHGNetEnergyCalculator, chgnet_calculator
from optimization.particle_encoding import Particle, ParticleSwarm
from optimization.fitness_function import calculate_fitness_stage1
from optimization.pso_optimizer import PSOOptimizer

__all__ = [
    "CHGNetEnergyCalculator",
    "chgnet_calculator", 
    "Particle",
    "ParticleSwarm",
    "calculate_fitness_stage1",
    "PSOOptimizer"
]