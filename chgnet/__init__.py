"""The pytorch implementation for CHGNet neural network potential."""

from __future__ import annotations

import os
from importlib.metadata import PackageNotFoundError, version
from typing import Literal

try:
    __version__ = version(__name__)  # read from pyproject.toml
except PackageNotFoundError:
    __version__ = "unknown"

TrainTask = Literal["ef", "efs", "efsm"]
PredTask = Literal["e", "ef", "em", "efs", "efsm"]

ROOT = os.path.dirname(os.path.dirname(__file__))

# 高熵结构优化模块
try:
    from chgnet.optimization import (
        PSOParticle,
        initialize_population,
        generate_structure_from_particle_stage1,
        calculate_configurational_entropy,
        calculate_fitness_stage1
    )
    __all__ = [
        "PSOParticle",
        "initialize_population", 
        "generate_structure_from_particle_stage1",
        "calculate_configurational_entropy",
        "calculate_fitness_stage1"
    ]
except ImportError:
    # 如果优化模块导入失败，提供友好的错误信息
    pass
