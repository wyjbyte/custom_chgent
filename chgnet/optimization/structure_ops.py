"""
结构生成模块

实现从PSO粒子生成晶体结构的核心功能
"""

from __future__ import annotations
from typing import Dict
import numpy as np
from pymatgen.core import Structure, Element
from ase import Atoms
from chgnet.optimization.particle import PSOParticle


def generate_structure_from_particle_stage1(
    base_structure: Structure,
    particle_arrangement: PSOParticle,
    site_groups_definition: Dict[str, Dict]
) -> Structure:
    """
    根据PSO粒子排列信息生成新的晶体结构
    
    Args:
        base_structure (Structure): 基础结构，定义晶胞和总原子数
        particle_arrangement (PSOParticle): PSO粒子，描述原子在位点群内的排列
        site_groups_definition (Dict[str, Dict]): 位点群信息
        
    Returns:
        Structure: 根据粒子排列修改后的新结构
    """
    # 创建新结构的副本
    new_structure = base_structure.copy()
    
    # 根据粒子排列修改结构中的原子种类
    for group_name, group_info in site_groups_definition.items():
        site_indices = group_info["site_indices"]
        particle_arrangement_for_group = particle_arrangement.site_groups_arrangement[group_name]
        
        # 更新每个位点的原子种类
        for i, site_idx in enumerate(site_indices):
            atomic_number = particle_arrangement_for_group[i]
            element = Element.from_Z(atomic_number)
            new_structure.replace(site_idx, element)
    
    return new_structure


def structure_to_ase_atoms(structure: Structure) -> Atoms:
    """
    将pymatgen Structure转换为ASE Atoms对象
    
    Args:
        structure (Structure): pymatgen结构
        
    Returns:
        Atoms: ASE原子对象
    """
    symbols = [str(site.specie) for site in structure]
    positions = structure.cart_coords
    cell = structure.lattice.matrix
    
    atoms = Atoms(
        symbols=symbols,
        positions=positions,
        cell=cell,
        pbc=True
    )
    
    return atoms


def ase_atoms_to_structure(atoms: Atoms) -> Structure:
    """
    将ASE Atoms对象转换为pymatgen Structure
    
    Args:
        atoms (Atoms): ASE原子对象
        
    Returns:
        Structure: pymatgen结构
    """
    from pymatgen.core import Lattice
    
    lattice = Lattice(atoms.get_cell())
    species = atoms.get_chemical_symbols()
    coords = atoms.get_scaled_positions()
    
    structure = Structure(lattice, species, coords)
    return structure