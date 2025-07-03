from __future__ import annotations

import contextlib
import csv
import inspect
import io
import math
import pickle
import sys
import warnings
from typing import TYPE_CHECKING, Literal
from pathlib import Path
import os

import numpy as np
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    
from ase import Atoms, units
from ase.calculators.calculator import (
    BaseCalculator,
    Calculator,
    all_changes,
    all_properties,
)
from ase.md.npt import NPT
from ase.md.nptberendsen import Inhomogeneous_NPTBerendsen, NPTBerendsen, NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet
from ase.optimize.bfgs import BFGS
from ase.optimize.bfgslinesearch import BFGSLineSearch
from ase.optimize.fire import FIRE
from ase.optimize.lbfgs import LBFGS, LBFGSLineSearch
from ase.optimize.mdmin import MDMin
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG

try:
    from pymatgen.analysis.eos import BirchMurnaghan
    from pymatgen.core.structure import Molecule, Structure
    from pymatgen.io.ase import AseAtomsAdaptor
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False

from chgnet.model.model import CHGNet
from chgnet.utils import determine_device

if TYPE_CHECKING:
    from ase.io import Trajectory
    from ase.optimize.optimize import Optimizer
    from typing_extensions import Self

    from chgnet import PredTask

# We would like to thank M3GNet develop team for this module
# source: https://github.com/materialsvirtuallab/m3gnet

OPTIMIZERS = {
    "FIRE": FIRE,
    "BFGS": BFGS,
    "LBFGS": LBFGS,
    "LBFGSLineSearch": LBFGSLineSearch,
    "MDMin": MDMin,
    "SciPyFminCG": SciPyFminCG,
    "SciPyFminBFGS": SciPyFminBFGS,
    "BFGSLineSearch": BFGSLineSearch,
}


class CHGNetCalculator(Calculator):
    """CHGNet Calculator for ASE applications."""

    implemented_properties = ("energy", "forces", "stress", "magmoms", "energies")

    def __init__(
        self,
        model: CHGNet | None = None,
        *,
        use_device: str | None = None,
        check_cuda_mem: bool = False,
        stress_weight: float = units.GPa,  # GPa to eV/A^3
        on_isolated_atoms: Literal["ignore", "warn", "error"] = "warn",
        return_site_energies: bool = False,
        **kwargs,
    ) -> None:
        """Provide a CHGNet instance to calculate various atomic properties using ASE.

        Args:
            model (CHGNet): instance of a chgnet model. If set to None,
                the pretrained CHGNet is loaded.
                Default = None
            use_device (str, optional): The device to be used for predictions,
                either "cpu", "cuda", or "mps". If not specified, the default device is
                automatically selected based on the available options.
                Default = None
            check_cuda_mem (bool): Whether to use cuda with most available memory
                Default = False
            stress_weight (float): the conversion factor to convert GPa to eV/A^3.
                Default = 1/160.21
            on_isolated_atoms ('ignore' | 'warn' | 'error'): how to handle Structures
                with isolated atoms.
                Default = 'warn'
            return_site_energies (bool): whether to return the energy of each atom
            **kwargs: Passed to the Calculator parent class.
        """
        super().__init__(**kwargs)

        # Determine the device to use
        device = determine_device(use_device=use_device, check_cuda_mem=check_cuda_mem)
        self.device = device

        # Move the model to the specified device
        if model is None:
            self.model = CHGNet.load(verbose=False, use_device=self.device)
        else:
            self.model = model.to(self.device)
        self.model.graph_converter.set_isolated_atom_response(on_isolated_atoms)
        self.stress_weight = stress_weight
        self.return_site_energies = return_site_energies
        print(f"CHGNet will run on {self.device}")

    @classmethod
    def from_file(cls, path: str, use_device: str | None = None, **kwargs) -> Self:
        """Load a user's CHGNet model and initialize the Calculator."""
        return cls(
            model=CHGNet.from_file(path),
            use_device=use_device,
            **kwargs,
        )

    @property
    def version(self) -> str | None:
        """The version of CHGNet."""
        return self.model.version

    @property
    def n_params(self) -> int:
        """The number of parameters in CHGNet."""
        return self.model.n_params

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list | None = None,
        system_changes: list | None = None,
        task: PredTask = "efsm",
    ) -> None:
        """Calculate various properties of the atoms using CHGNet.

        Args:
            atoms (Atoms | None): The atoms object to calculate properties for.
            properties (list | None): The properties to calculate.
                Default is all properties.
            system_changes (list | None): The changes made to the system.
                Default is all changes.
            task (PredTask): The task to perform. One of "e", "ef", "em", "efs", "efsm".
                Default = "efsm"
        """
        properties = properties or all_properties
        system_changes = system_changes or all_changes
        super().calculate(
            atoms=atoms,
            properties=properties,
            system_changes=system_changes,
        )

        # Run CHGNet
        structure = AseAtomsAdaptor.get_structure(atoms)
        graph = self.model.graph_converter(structure)
        model_prediction = self.model.predict_graph(
            graph.to(self.device),
            task=task,
            return_crystal_feas=True,
            return_site_energies=self.return_site_energies,
        )

        # Convert Result
        extensive_factor = len(structure) if self.model.is_intensive else 1
        key_map = dict(
            e=("energy", extensive_factor),
            f=("forces", 1),
            m=("magmoms", 1),
            s=("stress", self.stress_weight),
        )
        self.results |= {
            long_key: model_prediction[key] * factor
            for key, (long_key, factor) in key_map.items()
            if key in model_prediction
        }
        self.results["free_energy"] = self.results["energy"]
        self.results["crystal_fea"] = model_prediction["crystal_fea"]
        if self.return_site_energies:
            self.results["energies"] = model_prediction["site_energies"]


class StructOptimizer:
    """Wrapper class for structural relaxation."""

    def __init__(
        self,
        model: CHGNet | CHGNetCalculator | None = None,
        optimizer_class: Optimizer | str | None = "FIRE",
        use_device: str | None = None,
        stress_weight: float = units.GPa,
        on_isolated_atoms: Literal["ignore", "warn", "error"] = "warn",
    ) -> None:
        """Provide a trained CHGNet model and an optimizer to relax crystal structures.

        Args:
            model (CHGNet): instance of a CHGNet model or CHGNetCalculator.
                If set to None, the pretrained CHGNet is loaded.
                Default = None
            optimizer_class (Optimizer,str): choose optimizer from ASE.
                Default = "FIRE"
            use_device (str, optional): The device to be used for predictions,
                either "cpu", "cuda", or "mps". If not specified, the default device is
                automatically selected based on the available options.
                Default = None
            stress_weight (float): the conversion factor to convert GPa to eV/A^3.
                Default = 1/160.21
            on_isolated_atoms ('ignore' | 'warn' | 'error'): how to handle Structures
                with isolated atoms.
                Default = 'warn'
        """
        if isinstance(optimizer_class, str):
            if optimizer_class in OPTIMIZERS:
                optimizer_class = OPTIMIZERS[optimizer_class]
            else:
                raise ValueError(
                    f"Optimizer instance not found. Select from {list(OPTIMIZERS)}"
                )

        self.optimizer_class: Optimizer = optimizer_class

        if isinstance(model, CHGNetCalculator):
            self.calculator = model
        else:
            self.calculator = CHGNetCalculator(
                model=model,
                stress_weight=stress_weight,
                use_device=use_device,
                on_isolated_atoms=on_isolated_atoms,
            )

    @property
    def version(self) -> str:
        """The version of CHGNet."""
        return self.calculator.model.version

    @property
    def n_params(self) -> int:
        """The number of parameters in CHGNet."""
        return self.calculator.model.n_params


    def relax(
        self,
        atoms: Structure | Atoms,
        *,
        fmax: float | None = 0.1,
        steps: int | None = 500,
        relax_cell: bool | None = True,
        ase_filter: str | None = "FrechetCellFilter",
        save_path: str | None = None,
        loginterval: int | None = 1,
        crystal_feas_save_path: str | None = None,
        verbose: bool = True,
        assign_magmoms: bool = True,
        **kwargs,
    ) -> dict[str, Structure | TrajectoryObserver]:
        """Relax the Structure/Atoms until maximum force is smaller than fmax.

        Args:
            atoms (Structure | Atoms): A Structure or Atoms object to relax.
            fmax (float | None): The maximum force tolerance for relaxation.
                Default = 0.1
            steps (int | None): The maximum number of steps for relaxation.
                Default = 500
            relax_cell (bool | None): Whether to relax the cell as well.
                Default = True
            ase_filter (str | ase.filters.Filter): The filter to apply to the atoms
                object for relaxation. Default = FrechetCellFilter
                Default used to be ExpCellFilter which was removed due to bug reported
                in https://gitlab.com/ase/ase/-/issues/1321 and fixed in
                https://gitlab.com/ase/ase/-/merge_requests/3024.
            save_path (str | None): The path to save the trajectory.
                Default = None
            loginterval (int | None): Interval for logging trajectory and crystal
                features. Default = 1
            crystal_feas_save_path (str | None): Path to save crystal feature vectors
                which are logged at a loginterval rage
                Default = None
            verbose (bool): Whether to print the output of the ASE optimizer.
                Default = True
            assign_magmoms (bool): Whether to assign magnetic moments to the final
                structure. Default = True
            **kwargs: Additional parameters for the optimizer.

        Returns:
            dict[str, Structure | TrajectoryObserver]:
                A dictionary with 'final_structure' and 'trajectory'.
        """
        from ase import filters
        from ase.filters import Filter

        valid_filter_names = [
            name
            for name, cls in inspect.getmembers(filters, inspect.isclass)
            if issubclass(cls, Filter)
        ]

        if isinstance(ase_filter, str):
            if ase_filter in valid_filter_names:
                ase_filter = getattr(filters, ase_filter)
            else:
                raise ValueError(
                    f"Invalid {ase_filter=}, must be one of {valid_filter_names}. "
                )

        if isinstance(atoms, Structure):
            atoms = AseAtomsAdaptor().get_atoms(atoms)

        atoms.calc = self.calculator  # assign model used to predict forces

        stream = sys.stdout if verbose else io.StringIO()
        with contextlib.redirect_stdout(stream):
            obs = TrajectoryObserver(atoms)

            if crystal_feas_save_path:
                cry_obs = CrystalFeasObserver(atoms)

            if relax_cell:
                atoms = ase_filter(atoms)
            optimizer: Optimizer = self.optimizer_class(atoms, **kwargs)
            optimizer.attach(obs, interval=loginterval)

            if crystal_feas_save_path:
                optimizer.attach(cry_obs, interval=loginterval)

            optimizer.run(fmax=fmax, steps=steps)
            obs()

        if save_path is not None:
            obs.save(save_path)

        if crystal_feas_save_path:
            cry_obs.save(crystal_feas_save_path)

        if isinstance(atoms, Filter):
            atoms = atoms.atoms
        struct = AseAtomsAdaptor.get_structure(atoms)

        if assign_magmoms:
            for key in struct.site_properties:
                struct.remove_site_property(property_name=key)
            struct.add_site_property(
                "magmom", [float(magmom) for magmom in atoms.get_magnetic_moments()]
            )
        return {"final_structure": struct, "trajectory": obs}



class TrajectoryObserver:
    """Trajectory observer is a hook in the relaxation process that saves the
    intermediate structures.
    """

    def __init__(self, atoms: Atoms) -> None:
        """Create a TrajectoryObserver from an Atoms object.

        Args:
            atoms (Atoms): the structure to observe.
        """
        self.atoms = atoms
        self.energies: list[float] = []
        self.forces: list[np.ndarray] = []
        self.stresses: list[np.ndarray] = []
        self.magmoms: list[np.ndarray] = []
        self.atom_positions: list[np.ndarray] = []
        self.cells: list[np.ndarray] = []

    def __call__(self) -> None:
        """The logic for saving the properties of an Atoms during the relaxation."""
        self.energies.append(self.compute_energy())
        self.forces.append(self.atoms.get_forces())
        self.stresses.append(self.atoms.get_stress())
        self.magmoms.append(self.atoms.get_magnetic_moments())
        self.atom_positions.append(self.atoms.get_positions())
        self.cells.append(self.atoms.get_cell()[:])

    def __len__(self) -> int:
        """The number of steps in the trajectory."""
        return len(self.energies)

    def compute_energy(self) -> float:
        """Calculate the potential energy.

        Returns:
            energy (float): the potential energy.
        """
        return self.atoms.get_potential_energy()

    def save(self, filename: str) -> None:
        """Save the trajectory to file.

        Args:
            filename (str): filename to save the trajectory
        """
        out_pkl = {
            "energy": self.energies,
            "forces": self.forces,
            "stresses": self.stresses,
            "magmoms": self.magmoms,
            "atom_positions": self.atom_positions,
            "cell": self.cells,
            "atomic_number": self.atoms.get_atomic_numbers(),
        }
        with open(filename, "wb") as file:
            pickle.dump(out_pkl, file)


class CrystalFeasObserver:
    """CrystalFeasObserver is a hook in the relaxation and MD process that saves the
    intermediate crystal feature structures.
    """

    def __init__(self, atoms: Atoms) -> None:
        """Create a CrystalFeasObserver from an Atoms object."""
        self.atoms = atoms
        self.crystal_feature_vectors: list[np.ndarray] = []

    def __call__(self) -> None:
        """Record Atoms crystal feature vectors after an MD/relaxation step."""
        self.crystal_feature_vectors.append(self.atoms._calc.results["crystal_fea"])  # noqa: SLF001

    def __len__(self) -> int:
        """Number of recorded steps."""
        return len(self.crystal_feature_vectors)

    def save(self, filename: str) -> None:
        """Save the crystal feature vectors to filename in pickle format."""
        out_pkl = {"crystal_feas": self.crystal_feature_vectors}
        with open(filename, "wb") as file:
            pickle.dump(out_pkl, file)


class MolecularDynamics:
    """Molecular dynamics class."""

    def __init__(
        self,
        atoms: Atoms | Structure,
        *,
        model: CHGNet | CHGNetCalculator | None = None,
        ensemble: str = "nvt",
        thermostat: str = "Berendsen_inhomogeneous",
        temperature: int = 300,
        starting_temperature: int | None = None,
        timestep: float = 2.0,
        pressure: float = 1.01325e-4,
        taut: float | None = None,
        taup: float | None = None,
        bulk_modulus: float | None = None,
        trajectory: str | Trajectory | None = None,
        logfile: str | None = None,
        loginterval: int = 1,
        crystal_feas_logfile: str | None = None,
        append_trajectory: bool = False,
        on_isolated_atoms: Literal["ignore", "warn", "error"] = "warn",
        return_site_energies: bool = False,
        use_device: str | None = None,
    ) -> None:
        """Initialize the MD class.

        Args:
            atoms (Atoms): atoms to run the MD
            model (CHGNet): instance of a CHGNet model or CHGNetCalculator.
                If set to None, the pretrained CHGNet is loaded.
                Default = None
            ensemble (str): choose from 'nve', 'nvt', 'npt'
                Default = "nvt"
            thermostat (str): Thermostat to use
                choose from "Nose-Hoover", "Berendsen", "Berendsen_inhomogeneous"
                Default = "Berendsen_inhomogeneous"
            temperature (float): temperature for MD simulation, in K
                Default = 300
            starting_temperature (float): starting temperature of MD simulation, in K
                if set as None, the MD starts with the momentum carried by ase.Atoms
                if input is a pymatgen.core.Structure, the MD starts at 0K
                Default = None
            timestep (float): time step in fs
                Default = 2
            pressure (float): pressure in GPa
                Can be 3x3 or 6 np.array if thermostat is "Nose-Hoover"
                Default = 1.01325e-4 GPa = 1 atm
            taut (float): time constant for temperature coupling in fs.
                The temperature will be raised to target temperature in approximate
                10 * taut time.
                Default = 100 * timestep
            taup (float): time constant for pressure coupling in fs
                Default = 1000 * timestep
            bulk_modulus (float): bulk modulus of the material in GPa.
                Used in NPT ensemble for the barostat pressure coupling.
                The DFT bulk modulus can be found for most materials at
                https://next-gen.materialsproject.org/

                In NPT ensemble, the effective damping time for pressure is multiplied
                by compressibility. In LAMMPS, Bulk modulus is defaulted to 10
                see: https://docs.lammps.org/fix_press_berendsen.html
                and: https://gitlab.com/ase/ase/-/blob/master/ase/md/nptberendsen.py

                If bulk modulus is not provided here, it will be calculated by CHGNet
                through Birch Murnaghan equation of state (EOS).
                Note the EOS fitting can fail because of non-parabolic potential
                energy surface, which is common with soft system like liquid and gas.
                In such case, user should provide an input bulk modulus for better
                barostat coupling, otherwise a guessed bulk modulus = 2 GPa will be used
                (water's bulk modulus)

                Default = None
            trajectory (str or Trajectory): Attach trajectory object
                Default = None
            logfile (str): open this file for recording MD outputs
                Default = None
            loginterval (int): write to log file every interval steps
                Default = 1
            crystal_feas_logfile (str): open this file for recording crystal features
                during MD. Default = None
            append_trajectory (bool): Whether to append to prev trajectory.
                If false, previous trajectory gets overwritten
                Default = False
            on_isolated_atoms ('ignore' | 'warn' | 'error'): how to handle Structures
                with isolated atoms.
                Default = 'warn'
            return_site_energies (bool): whether to return the energy of each atom
            use_device (str): the device for the MD run
                Default = None
        """
        self.ensemble = ensemble
        self.thermostat = thermostat
        if isinstance(atoms, Structure | Molecule):
            atoms = AseAtomsAdaptor().get_atoms(atoms)
            # atoms = atoms.to_ase_atoms()

        if starting_temperature is not None:
            MaxwellBoltzmannDistribution(
                atoms, temperature_K=starting_temperature, force_temp=True
            )
            Stationary(atoms)

        self.atoms = atoms
        if isinstance(model, Calculator | BaseCalculator):
            self.atoms.calc = model
        else:
            self.atoms.calc = CHGNetCalculator(
                model=model,
                use_device=use_device,
                on_isolated_atoms=on_isolated_atoms,
                return_site_energies=return_site_energies,
            )

        if taut is None:
            taut = 100 * timestep
        if taup is None:
            taup = 1000 * timestep

        if ensemble.lower() == "nve":
            """
            VelocityVerlet (constant N, V, E) molecular dynamics.

            Note: it's recommended to use smaller timestep for NVE compared to other
            ensembles, since the VelocityVerlet algorithm assumes a strict conservative
            force field.
            """
            self.dyn = VelocityVerlet(
                atoms=self.atoms,
                timestep=timestep * units.fs,
                trajectory=trajectory,
                logfile=logfile,
                loginterval=loginterval,
                append_trajectory=append_trajectory,
            )
            print("NVE-MD created")

        elif ensemble.lower() == "nvt":
            """
            Constant volume/temperature molecular dynamics.
            """
            if thermostat.lower() == "nose-hoover":
                """
                Nose-hoover (constant N, V, T) molecular dynamics.
                ASE implementation currently only supports upper triangular lattice
                """
                self.upper_triangular_cell()
                self.dyn = NPT(
                    atoms=self.atoms,
                    timestep=timestep * units.fs,
                    temperature_K=temperature,
                    externalstress=pressure
                    * units.GPa,  # ase NPT does not like externalstress=None
                    ttime=taut * units.fs,
                    pfactor=None,
                    trajectory=trajectory,
                    logfile=logfile,
                    loginterval=loginterval,
                    append_trajectory=append_trajectory,
                )
                print("NVT-Nose-Hoover MD created")
            elif thermostat.lower().startswith("berendsen"):
                """
                Berendsen (constant N, V, T) molecular dynamics.
                """
                self.dyn = NVTBerendsen(
                    atoms=self.atoms,
                    timestep=timestep * units.fs,
                    temperature_K=temperature,
                    taut=taut * units.fs,
                    trajectory=trajectory,
                    logfile=logfile,
                    loginterval=loginterval,
                    append_trajectory=append_trajectory,
                )
                print("NVT-Berendsen-MD created")
            else:
                raise ValueError(
                    "Thermostat not supported, choose in 'Nose-Hoover', 'Berendsen', "
                    "'Berendsen_inhomogeneous'"
                )

        elif ensemble.lower() == "npt":
            """
            Constant pressure/temperature molecular dynamics.
            """
            # Bulk modulus is needed for pressure damping time
            if bulk_modulus is not None:
                bulk_modulus_au = bulk_modulus / 160.2176  # GPa to eV/A^3
                compressibility_au = 1 / bulk_modulus_au
            else:
                try:
                    # Fit bulk modulus by equation of state
                    eos = EquationOfState(model=self.atoms.calc)
                    eos.fit(atoms=atoms, steps=500, fmax=0.1, verbose=False)
                    bulk_modulus = eos.get_bulk_modulus(unit="GPa")
                    bulk_modulus_au = eos.get_bulk_modulus(unit="eV/A^3")
                    compressibility_au = eos.get_compressibility(unit="A^3/eV")
                    print(
                        f"Completed bulk modulus calculation: "
                        f"k = {bulk_modulus:.3}GPa, {bulk_modulus_au:.3}eV/A^3"
                    )
                except Exception:
                    bulk_modulus_au = 2 / 160.2176
                    compressibility_au = 1 / bulk_modulus_au
                    warnings.warn(
                        "Warning!!! Equation of State fitting failed, setting bulk "
                        "modulus to 2 GPa. NPT simulation can proceed with incorrect "
                        "pressure relaxation time."
                        "User input for bulk modulus is recommended.",
                        stacklevel=2,
                    )
            self.bulk_modulus = bulk_modulus

            if thermostat.lower() == "nose-hoover":
                """
                Combined Nose-Hoover and Parrinello-Rahman dynamics, creating an
                NPT (or N,stress,T) ensemble.
                see: https://gitlab.com/ase/ase/-/blob/master/ase/md/npt.py
                ASE implementation currently only supports upper triangular lattice
                """
                self.upper_triangular_cell()
                ptime = taup * units.fs
                self.dyn = NPT(
                    atoms=self.atoms,
                    timestep=timestep * units.fs,
                    temperature_K=temperature,
                    externalstress=pressure * units.GPa,
                    ttime=taut * units.fs,
                    pfactor=bulk_modulus * units.GPa * ptime * ptime,
                    trajectory=trajectory,
                    logfile=logfile,
                    loginterval=loginterval,
                    append_trajectory=append_trajectory,
                )
                print("NPT-Nose-Hoover MD created")

            elif thermostat.lower() == "berendsen_inhomogeneous":
                """
                Inhomogeneous_NPTBerendsen thermo/barostat
                This is a more flexible scheme that fixes three angles of the unit
                cell but allows three lattice parameter to change independently.
                see: https://gitlab.com/ase/ase/-/blob/master/ase/md/nptberendsen.py
                """

                self.dyn = Inhomogeneous_NPTBerendsen(
                    atoms=self.atoms,
                    timestep=timestep * units.fs,
                    temperature_K=temperature,
                    pressure_au=pressure * units.GPa,
                    taut=taut * units.fs,
                    taup=taup * units.fs,
                    compressibility_au=compressibility_au,
                    trajectory=trajectory,
                    logfile=logfile,
                    loginterval=loginterval,
                )
                print("NPT-Berendsen-inhomogeneous-MD created")

            elif thermostat.lower() == "npt_berendsen":
                """
                This is a similar scheme to the Inhomogeneous_NPTBerendsen.
                This is a less flexible scheme that fixes the shape of the
                cell - three angles are fixed and the ratios between the three
                lattice constants.
                see: https://gitlab.com/ase/ase/-/blob/master/ase/md/nptberendsen.py
                """

                self.dyn = NPTBerendsen(
                    atoms=self.atoms,
                    timestep=timestep * units.fs,
                    temperature_K=temperature,
                    pressure_au=pressure * units.GPa,
                    taut=taut * units.fs,
                    taup=taup * units.fs,
                    compressibility_au=compressibility_au,
                    trajectory=trajectory,
                    logfile=logfile,
                    loginterval=loginterval,
                    append_trajectory=append_trajectory,
                )
                print("NPT-Berendsen-MD created")
            else:
                raise ValueError(
                    "Thermostat not supported, choose in 'Nose-Hoover', 'Berendsen', "
                    "'Berendsen_inhomogeneous'"
                )

        self.trajectory = trajectory
        self.logfile = logfile
        self.loginterval = loginterval
        self.timestep = timestep
        self.crystal_feas_logfile = crystal_feas_logfile

    def run(self, steps: int) -> None:
        """Thin wrapper of ase MD run.

        Args:
            steps (int): number of MD steps
        """
        if self.crystal_feas_logfile:
            obs = CrystalFeasObserver(self.atoms)
            self.dyn.attach(obs, interval=self.loginterval)

        self.dyn.run(steps)

        if self.crystal_feas_logfile:
            obs.save(self.crystal_feas_logfile)

    def set_atoms(self, atoms: Atoms) -> None:
        """Set new atoms to run MD.

        Args:
            atoms (Atoms): new atoms for running MD
        """
        calculator = self.atoms.calc
        self.atoms = atoms
        self.dyn.atoms = atoms
        self.dyn.atoms.calc = calculator

    def upper_triangular_cell(self, *, verbose: bool | None = False) -> None:
        """Transform to upper-triangular cell.
        ASE Nose-Hoover implementation only supports upper-triangular cell
        while ASE's canonical description is lower-triangular cell.

        Args:
            verbose (bool): Whether to notify user about upper-triangular cell
                transformation. Default = False
        """
        if not NPT._isuppertriangular(self.atoms.get_cell()):  # noqa: SLF001
            a, b, c, alpha, beta, gamma = self.atoms.cell.cellpar()
            angles = np.radians((alpha, beta, gamma))
            sin_a, sin_b, _sin_g = np.sin(angles)
            cos_a, cos_b, cos_g = np.cos(angles)
            cos_p = (cos_g - cos_a * cos_b) / (sin_a * sin_b)
            cos_p = np.clip(cos_p, -1, 1)
            sin_p = (1 - cos_p**2) ** 0.5

            new_basis = [
                (a * sin_b * sin_p, a * sin_b * cos_p, a * cos_b),
                (0, b * sin_a, b * cos_a),
                (0, 0, c),
            ]

            self.atoms.set_cell(new_basis, scale_atoms=True)
            if verbose:
                print("Transformed to upper triangular unit cell.", flush=True)


class EquationOfState:
    """Class to calculate equation of state."""

    def __init__(
        self,
        model: CHGNet | CHGNetCalculator | None = None,
        optimizer_class: Optimizer | str | None = "FIRE",
        use_device: str | None = None,
        stress_weight: float = units.GPa,
        on_isolated_atoms: Literal["ignore", "warn", "error"] = "error",
    ) -> None:
        """Initialize a structure optimizer object for calculation of bulk modulus.

        Args:
            model (CHGNet): instance of a CHGNet model or CHGNetCalculator.
                If set to None, the pretrained CHGNet is loaded.
                Default = None
            optimizer_class (Optimizer,str): choose optimizer from ASE.
                Default = "FIRE"
            use_device (str, optional): The device to be used for predictions,
                either "cpu", "cuda", or "mps". If not specified, the default device is
                automatically selected based on the available options.
                Default = None
            stress_weight (float): the conversion factor to convert GPa to eV/A^3.
                Default = 1/160.21
            on_isolated_atoms ('ignore' | 'warn' | 'error'): how to handle Structures
                with isolated atoms.
                Default = 'error'
        """
        self.relaxer = StructOptimizer(
            model=model,
            optimizer_class=optimizer_class,
            use_device=use_device,
            stress_weight=stress_weight,
            on_isolated_atoms=on_isolated_atoms,
        )
        self.fitted = False

    def fit(
        self,
        atoms: Structure | Atoms,
        *,
        n_points: int = 11,
        fmax: float | None = 0.1,
        steps: int | None = 500,
        verbose: bool | None = False,
        **kwargs,
    ) -> None:
        """Relax the Structure/Atoms and fit the Birch-Murnaghan equation of state.

        Args:
            atoms (Structure | Atoms): A Structure or Atoms object to relax.
            n_points (int): Number of structures used in fitting the equation of states
            fmax (float | None): The maximum force tolerance for relaxation.
                Default = 0.1
            steps (int | None): The maximum number of steps for relaxation.
                Default = 500
            verbose (bool): Whether to print the output of the ASE optimizer.
                Default = False
            **kwargs: Additional parameters for the optimizer.
        """
        if isinstance(atoms, Atoms):
            atoms = AseAtomsAdaptor.get_structure(atoms)
        primitive_cell = atoms.get_primitive_structure()
        local_minima = self.relaxer.relax(
            primitive_cell,
            relax_cell=True,
            fmax=fmax,
            steps=steps,
            verbose=verbose,
            **kwargs,
        )

        volumes, energies = [], []
        for idx in np.linspace(-0.1, 0.1, n_points):
            structure_strained = local_minima["final_structure"].copy()
            structure_strained.apply_strain([idx, idx, idx])
            result = self.relaxer.relax(
                structure_strained,
                relax_cell=False,
                fmax=fmax,
                steps=steps,
                verbose=verbose,
                **kwargs,
            )
            volumes.append(result["final_structure"].volume)
            energies.append(result["trajectory"].energies[-1])
        self.bm = BirchMurnaghan(volumes=volumes, energies=energies)
        self.bm.fit()
        self.fitted = True

    def get_bulk_modulus(self, unit: Literal["eV/A^3", "GPa"] = "eV/A^3") -> float:
        """Get the bulk modulus of from the fitted Birch-Murnaghan equation of state.

        Args:
            unit (str): The unit of bulk modulus. Can be "eV/A^3" or "GPa"
                Default = "eV/A^3"

        Returns:
            float: Bulk Modulus

        Raises:
            ValueError: If the equation of state is not fitted.
        """
        if self.fitted is False:
            raise ValueError(
                "Equation of state needs to be fitted first through self.fit()"
            )
        if unit == "eV/A^3":
            return self.bm.b0
        if unit == "GPa":
            return self.bm.b0_GPa
        raise ValueError("unit has to be eV/A^3 or GPa")

    def get_compressibility(self, unit: str = "A^3/eV") -> float:
        """Get the bulk modulus of from the fitted Birch-Murnaghan equation of state.

        Args:
            unit (str): The unit of bulk modulus. Can be "A^3/eV",
            "GPa^-1" "Pa^-1" or "m^2/N"
                Default = "A^3/eV"

        Returns:
            Bulk Modulus (float)
        """
        if self.fitted is False:
            raise ValueError(
                "Equation of state needs to be fitted first through self.fit()"
            )
        if unit == "A^3/eV":
            return 1 / self.bm.b0
        if unit == "GPa^-1":
            return 1 / self.bm.b0_GPa
        if unit in {"Pa^-1", "m^2/N"}:
            return 1 / (self.bm.b0_GPa * 1e9)
        raise NotImplementedError("unit has to be one of A^3/eV, GPa^-1 Pa^-1 or m^2/N")


class HighEntropyOptimizer:
    """高熵结构优化器类
    
    基于CHGNet框架实现两阶段高熵结构优化，包括：
    1. 原子半径表加载与解析
    2. 位点群定义与原子统计
    3. 构型熵计算模块
    """
    
    def __init__(self, model=None, radius_table_path: str | None = None):
        """初始化高熵优化器
        
        Args:
            model: CHGNet模型实例，如果为None则加载默认模型
            radius_table_path: 原子半径表路径，如果为None则使用默认路径
        """
        # 初始化CHGNet模型 (如果可用)
        if model is None and "CHGNet" in globals():
            try:
                self.model = CHGNet.load()
            except Exception:
                print("警告: 无法加载CHGNet模型，某些功能可能不可用")
                self.model = None
        else:
            self.model = model
            
        # 设置原子半径表路径
        if radius_table_path is None:
            # 获取CHGNet包的路径
            chgnet_path = Path(__file__).parent.parent
            self.radius_table_path = chgnet_path / "data" / "Atomic_radius_table.csv"
        else:
            self.radius_table_path = Path(radius_table_path)
            
        # 加载原子半径表
        self.radius_table = self._load_radius_table()
        
        # 玻尔兹曼常数 (eV/K)
        self.k_B = 8.617333262145e-5
        
    def _load_radius_table(self) -> dict:
        """加载和解析原子半径表 (使用标准库CSV模块)
        
        Returns:
            dict: 原子半径表数据字典
        """
        if not self.radius_table_path.exists():
            raise FileNotFoundError(f"原子半径表文件未找到: {self.radius_table_path}")
            
        radius_data = {}
        
        with open(self.radius_table_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)  # 跳过表头
            
            for row in reader:
                if len(row) >= 4:
                    element = row[0]
                    radius_a = row[1] if row[1] else None
                    radius_b = row[2] if row[2] else None  
                    radius_o = row[3] if row[3] else None
                    
                    radius_data[element] = {
                        'A': float(radius_a) if radius_a else None,
                        'B': float(radius_b) if radius_b else None,
                        'O': float(radius_o) if radius_o else None
                    }
                    
        return radius_data
        
    def get_ionic_radius(self, element: str, site_type: str, default_radius: float = 100.0) -> float:
        """根据元素和位点类型查询离子半径
        
        Args:
            element: 元素符号
            site_type: 位点类型 ('A', 'B', 'O')
            default_radius: 默认半径值 (pm)
            
        Returns:
            float: 离子半径值 (pm)
        """
        # 查找元素
        if element not in self.radius_table:
            print(f"警告: 元素 {element} 未在半径表中找到，使用默认半径 {default_radius} pm")
            return default_radius
            
        # 根据位点类型获取半径
        radius_value = self.radius_table[element][site_type]
        
        if radius_value is None:
            print(f"警告: 元素 {element} 在 {site_type} 位点的半径未定义，使用默认半径 {default_radius} pm")
            return default_radius
            
        return float(radius_value)
        
    def identify_site_groups(self, structure_info: dict) -> dict:
        """基于原子半径表识别A/B/O位点群
        
        Args:
            structure_info: 结构信息字典，包含 'elements' 键 (元素列表)
            
        Returns:
            dict: 位点群定义，包含每个位点群的原子索引和元素组成
        """
        site_groups = {"A": [], "B": [], "O": []}
        
        elements = structure_info.get('elements', [])
        
        # 遍历结构中的每个原子
        for i, element in enumerate(elements):
            # 根据原子半径表确定位点类型
            # 优先级: O > A > B (基于简单的遍历方法)
            if element in self.radius_table:
                data = self.radius_table[element]
                if data["O"] is not None:
                    site_groups["O"].append(i)
                elif data["A"] is not None:
                    site_groups["A"].append(i)
                elif data["B"] is not None:
                    site_groups["B"].append(i)
                else:
                    # 如果都没有定义，使用默认分配
                    self._assign_default_site(element, i, site_groups)
            else:
                # 元素不在表中，使用默认分配
                self._assign_default_site(element, i, site_groups)
                        
        return site_groups
        
    def _assign_default_site(self, element: str, index: int, site_groups: dict):
        """为未在半径表中的元素分配默认位点"""
        # 根据常见规律分配位点
        if element in ["O", "F", "Cl", "Br", "I", "N", "S", "Se", "Te"]:
            site_groups["O"].append(index)
        else:
            # 对于金属元素，根据元素符号简单分配
            # 重金属倾向于A位点，轻金属倾向于B位点
            if element in ["Cs", "Rb", "K", "Ba", "Sr", "Ca", "La", "Ce", "Nd", "Sm", "Pr"]:
                site_groups["A"].append(index)
            else:
                site_groups["B"].append(index)
        
    def count_elements_in_site_group(self, elements: list, site_indices: list) -> dict:
        """统计位点群中各元素的原子数量
        
        Args:
            elements: 元素列表
            site_indices: 位点群的原子索引列表
            
        Returns:
            dict: 元素-数量映射
        """
        element_counts = {}
        
        for idx in site_indices:
            if idx < len(elements):
                element = elements[idx]
                element_counts[element] = element_counts.get(element, 0) + 1
            
        return element_counts
        
    def calculate_configurational_entropy(self, structure_info: dict, site_groups_definition: dict | None = None) -> float:
        """计算构型熵
        
        Args:
            structure_info: 结构信息字典，包含 'elements' 键
            site_groups_definition: 位点群定义，如果为None则自动识别
            
        Returns:
            float: 总构型熵值 (eV/K)
        """
        if site_groups_definition is None:
            site_groups_definition = self.identify_site_groups(structure_info)
            
        elements = structure_info.get('elements', [])
        total_entropy = 0.0
        
        # 对每个位点群独立计算构型熵
        for site_type, site_indices in site_groups_definition.items():
            if not site_indices:  # 空位点群跳过
                continue
                
            # 统计该位点群中各元素的数量
            element_counts = self.count_elements_in_site_group(elements, site_indices)
            
            # 计算总原子数
            total_atoms_in_group = sum(element_counts.values())
            
            if total_atoms_in_group <= 1:  # 单原子或空群组没有构型熵
                continue
                
            # 计算各元素的摩尔分数
            site_entropy = 0.0
            for element, count in element_counts.items():
                mole_fraction = count / total_atoms_in_group
                
                # 处理数值稳定性，避免ln(0)
                if mole_fraction > 1e-10:
                    site_entropy -= mole_fraction * math.log(mole_fraction)
                    
            # 加到总熵中（乘以该位点群的原子数）
            total_entropy += site_entropy * total_atoms_in_group
            
        # 返回以eV/K为单位的熵值
        return total_entropy * self.k_B
        
    # Pymatgen结构支持 (如果可用)
    if PYMATGEN_AVAILABLE:
        def process_pymatgen_structure(self, structure) -> dict:
            """处理pymatgen结构对象"""
            elements = []
            for site in structure:
                elements.append(site.species_string)
            return {'elements': elements}
            
        def calculate_configurational_entropy_from_structure(self, structure, site_groups_definition: dict | None = None) -> float:
            """从pymatgen结构计算构型熵"""
            structure_info = self.process_pymatgen_structure(structure)
            return self.calculate_configurational_entropy(structure_info, site_groups_definition)
