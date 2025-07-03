"""
高熵结构优化模块 - CHGNet计算器封装
提供简单的CHGNet能量计算接口
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

if TYPE_CHECKING:
    from ase import Atoms

# 延迟导入CHGNet以避免初始化时的依赖问题
_chgnet_calculator = None
_chgnet_model = None


def _get_chgnet_calculator():
    """获取CHGNet计算器实例（延迟初始化）"""
    global _chgnet_calculator
    if _chgnet_calculator is None:
        try:
            from chgnet.model.dynamics import CHGNetCalculator
            _chgnet_calculator = CHGNetCalculator()
            print("CHGNet计算器初始化成功")
        except Exception as e:
            print(f"CHGNet计算器初始化失败: {e}")
            raise
    return _chgnet_calculator


def _get_chgnet_model():
    """获取CHGNet模型实例（延迟初始化）"""
    global _chgnet_model
    if _chgnet_model is None:
        try:
            from chgnet.model.model import CHGNet
            _chgnet_model = CHGNet.load()
            print("CHGNet模型加载成功")
        except Exception as e:
            print(f"CHGNet模型加载失败: {e}")
            raise
    return _chgnet_model


def calculateEnergyFromStructure(structure: Structure) -> float:
    """
    从pymatgen Structure计算CHGNet能量
    
    Args:
        structure: pymatgen Structure对象
        
    Returns:
        float: CHGNet计算的能量 (eV)
        
    Raises:
        ValueError: 当结构无效时
        RuntimeError: 当CHGNet计算失败时
    """
    if not isinstance(structure, Structure):
        raise ValueError("输入必须是pymatgen Structure对象")
        
    try:
        model = _get_chgnet_model()
        prediction = model.predict_structure(structure, task="e")
        energy = prediction["e"].item()  # 提取标量值
        return energy
    except Exception as e:
        raise RuntimeError(f"CHGNet能量计算失败: {e}")


def calculateEnergyFromAtoms(atoms: Atoms) -> float:
    """
    从ASE Atoms计算CHGNet能量
    
    Args:
        atoms: ASE Atoms对象
        
    Returns:
        float: CHGNet计算的能量 (eV)
        
    Raises:
        ValueError: 当结构无效时
        RuntimeError: 当CHGNet计算失败时
    """
    try:
        # 转换为pymatgen Structure
        structure = AseAtomsAdaptor.get_structure(atoms)
        return calculateEnergyFromStructure(structure)
    except Exception as e:
        raise RuntimeError(f"从ASE Atoms计算能量失败: {e}")


class CHGNetEnergyCalculator:
    """
    CHGNet能量计算器封装类
    提供更高级的接口和错误处理
    """
    
    def __init__(self):
        """初始化计算器"""
        self.calculator = None
        self.model = None
        self._initialized = False
    
    def initialize(self):
        """手动初始化CHGNet（如果需要控制初始化时机）"""
        if not self._initialized:
            self.model = _get_chgnet_model()
            self.calculator = _get_chgnet_calculator()
            self._initialized = True
    
    def calculateEnergy(self, structure: Structure | Atoms) -> float:
        """
        计算结构的CHGNet能量
        
        Args:
            structure: pymatgen Structure或ASE Atoms对象
            
        Returns:
            float: CHGNet计算的能量 (eV)
        """
        if not self._initialized:
            self.initialize()
            
        if isinstance(structure, Structure):
            return calculateEnergyFromStructure(structure)
        else:
            return calculateEnergyFromAtoms(structure)
    
    def batchCalculateEnergy(self, structures: list[Structure | Atoms]) -> list[float]:
        """
        批量计算多个结构的能量
        
        Args:
            structures: 结构列表
            
        Returns:
            list[float]: 能量列表 (eV)
        """
        if not self._initialized:
            self.initialize()
            
        energies = []
        for structure in structures:
            try:
                energy = self.calculateEnergy(structure)
                energies.append(energy)
            except Exception as e:
                print(f"警告: 结构能量计算失败: {e}")
                energies.append(float('inf'))  # 使用无穷大表示失败
        
        return energies
    
    def isAvailable(self) -> bool:
        """检查CHGNet是否可用"""
        try:
            if not self._initialized:
                self.initialize()
            return True
        except Exception:
            return False


# 创建全局计算器实例
chgnet_calculator = CHGNetEnergyCalculator()