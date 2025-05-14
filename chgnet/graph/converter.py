from __future__ import annotations

import gc
import sys
import warnings
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from chgnet.graph.crystalgraph import CrystalGraph
from chgnet.graph.graph import Graph, Node

if TYPE_CHECKING:
    from typing import Literal

    from pymatgen.core import Structure
    from typing_extensions import Self

try:
    from chgnet.graph.cygraph import make_graph
except (ImportError, AttributeError):
    make_graph = None

TORCH_DTYPE = torch.float32


class CrystalGraphConverter(nn.Module):
    """Convert a pymatgen.core.Structure to a CrystalGraph
    The CrystalGraph dataclass stores essential field to make sure that
    gradients like force and stress can be calculated through back-propagation later.
    """

    make_graph = None

    def __init__(
        self,
        *,
        atom_graph_cutoff: float = 6,
        bond_graph_cutoff: float = 3,
        algorithm: Literal["legacy", "fast"] = "fast",
        on_isolated_atoms: Literal["ignore", "warn", "error"] = "error",
        verbose: bool = False,
    ) -> None:
        """Initialize the Crystal Graph Converter.

        Args:
            atom_graph_cutoff (float): cutoff radius to search for neighboring atom in
                atom_graph. Default = 5.
            bond_graph_cutoff (float): bond length threshold to include bond in
                bond_graph. Default = 3.
            algorithm ('legacy' | 'fast'): algorithm to use for converting graphs.
                'legacy': python implementation of graph creation
                'fast': C implementation of graph creation, this is faster,
                    but will need the cygraph.c file correctly compiled from pip install
                Default = 'fast'
            on_isolated_atoms ('ignore' | 'warn' | 'error'): how to handle Structures
                with isolated atoms.
                Default = 'error'
            verbose (bool): whether to print the CrystalGraphConverter
                initialization message. Default = False.
        """
        super().__init__()
        self.atom_graph_cutoff = atom_graph_cutoff
        self.bond_graph_cutoff = (
            atom_graph_cutoff if bond_graph_cutoff is None else bond_graph_cutoff
        )
        self.on_isolated_atoms = on_isolated_atoms

        # Set graph conversion algorithm
        self.create_graph = self._create_graph_legacy
        self.algorithm = "legacy"
        if algorithm == "fast":
            if make_graph is not None:
                self.create_graph = self._create_graph_fast
                self.algorithm = "fast"
            else:
                warnings.warn(
                    "`fast` algorithm is not available, using `legacy`",
                    UserWarning,
                    stacklevel=1,
                )
        elif algorithm != "legacy":
            warnings.warn(
                f"Unknown {algorithm=}, using `legacy`",
                UserWarning,
                stacklevel=1,
            )

        if verbose:
            print(self)

    def __repr__(self) -> str:
        """String representation of the CrystalGraphConverter."""
        atom_graph_cutoff = self.atom_graph_cutoff
        bond_graph_cutoff = self.bond_graph_cutoff
        algorithm = self.algorithm
        cls_name = type(self).__name__
        return f"{cls_name}({algorithm=}, {atom_graph_cutoff=}, {bond_graph_cutoff=})"

    def forward(
        self,
        structure: Structure,
        graph_id=None,
        mp_id=None,
    ) -> CrystalGraph:
        """Convert a structure, return a CrystalGraph.

        Args:
            structure (pymatgen.core.Structure): structure to convert
            graph_id (str): an id to keep track of this crystal graph
                Default = None
            mp_id (str): Materials Project id of this structure
                Default = None

        Return:
            CrystalGraph that is ready to use by CHGNet
        """
        n_atoms = len(structure)
        atomic_number = torch.tensor(
            [site.specie.Z for site in structure],
            dtype=torch.int32,
            requires_grad=False,
        )
        atom_frac_coord = torch.tensor(
            structure.frac_coords, dtype=TORCH_DTYPE, requires_grad=True
        )
        lattice = torch.tensor(
            structure.lattice.matrix, dtype=TORCH_DTYPE, requires_grad=True
        )
        center_index, neighbor_index, image, distance = structure.get_neighbor_list(
            r=self.atom_graph_cutoff, sites=structure.sites, numerical_tol=1e-8
        )

        # Make Graph
        graph = self.create_graph(
            n_atoms, center_index, neighbor_index, image, distance
        )

        # Atom Graph
        atom_graph, directed2undirected = graph.adjacency_list()
        atom_graph = torch.tensor(atom_graph, dtype=torch.int32)
        directed2undirected = torch.tensor(directed2undirected, dtype=torch.int32)

        # Bond Graph
        try:
            bond_graph, undirected2directed = graph.line_graph_adjacency_list(
                cutoff=self.bond_graph_cutoff
            )
        except Exception as exc:
            # Report structures that failed creating bond graph
            # This happen occasionally with pymatgen version issue
            structure.to(filename="bond_graph_error.cif")
            raise RuntimeError(
                f"Failed creating bond graph for {graph_id}, check bond_graph_error.cif"
            ) from exc
        bond_graph = torch.tensor(bond_graph, dtype=torch.int32)
        undirected2directed = torch.tensor(undirected2directed, dtype=torch.int32)

        # Check if graph has isolated atom
        n_isolated_atoms = len({*range(n_atoms)} - {*center_index})
        if n_isolated_atoms:
            atom_graph_cutoff = self.atom_graph_cutoff
            msg = (
                f"Structure {graph_id=} has {n_isolated_atoms} isolated atom(s) with "
                f"{atom_graph_cutoff=}. "
                f"CHGNet calculation will likely go wrong"
            )
            if self.on_isolated_atoms == "error":
                # Discard this structure if it has isolated atom in the graph
                raise ValueError(msg)
            elif self.on_isolated_atoms == "warn":  # noqa: RET506
                print(msg, file=sys.stderr)

        return CrystalGraph(
            atomic_number=atomic_number,
            atom_frac_coord=atom_frac_coord,
            atom_graph=atom_graph,
            neighbor_image=torch.tensor(image, dtype=TORCH_DTYPE),
            directed2undirected=directed2undirected,
            undirected2directed=undirected2directed,
            bond_graph=bond_graph,
            lattice=lattice,
            graph_id=graph_id,
            mp_id=mp_id,
            composition=structure.composition.formula,
            atom_graph_cutoff=self.atom_graph_cutoff,
            bond_graph_cutoff=self.bond_graph_cutoff,
        )

    @staticmethod
    def _create_graph_legacy(
        n_atoms: int,
        center_index: np.ndarray,
        neighbor_index: np.ndarray,
        image: np.ndarray,
        distance: np.ndarray,
    ) -> Graph:
        """Given structure information, create a Graph structure to be used to
        create Crystal_Graph using pure python implementation.

        Args:
            n_atoms (int): the number of atoms in the structure
            center_index (np.ndarray): np array of indices of center atoms.
                [num_undirected_bonds]
            neighbor_index (np.ndarray): np array of indices of neighbor atoms.
                [num_undirected_bonds]
            image (np.ndarray): np array of images for each edge.
                [num_undirected_bonds, 3]
            distance (np.ndarray): np array of distances.
                [num_undirected_bonds]

        Return:
            Graph data structure used to create Crystal_Graph object
        """
        graph = Graph([Node(index=idx) for idx in range(n_atoms)])
        for ii, jj, img, dist in zip(
            center_index, neighbor_index, image, distance, strict=True
        ):
            graph.add_edge(center_index=ii, neighbor_index=jj, image=img, distance=dist)

        return graph

    @staticmethod
    def _create_graph_fast(
        n_atoms: int,
        center_index: np.ndarray,
        neighbor_index: np.ndarray,
        image: np.ndarray,
        distance: np.ndarray,
    ) -> Graph:
        """Given structure information, create a Graph structure to be used to
        create Crystal_Graph using C implementation.

        NOTE: this is the fast version of _create_graph_legacy optimized
            in c (~3x speedup).

        Args:
            n_atoms (int): the number of atoms in the structure
            center_index (np.ndarray): np array of indices of center atoms.
                [num_undirected_bonds]
            neighbor_index (np.ndarray): np array of indices of neighbor atoms.
                [num_undirected_bonds]
            image (np.ndarray): np array of images for each edge.
                [num_undirected_bonds, 3]
            distance (np.ndarray): np array of distances.
                [num_undirected_bonds]

        Return:
            Graph data structure used to create Crystal_Graph object
        """
        center_index = np.ascontiguousarray(center_index)
        neighbor_index = np.ascontiguousarray(neighbor_index)
        image = np.ascontiguousarray(image, dtype=np.int64)
        distance = np.ascontiguousarray(distance)
        gc_saved = gc.get_threshold()
        gc.set_threshold(0)
        nodes, dir_edges_list, undir_edges_list, undirected_edges = make_graph(
            center_index, len(center_index), neighbor_index, image, distance, n_atoms
        )
        graph = Graph(nodes=nodes)
        graph.directed_edges_list = dir_edges_list
        graph.undirected_edges_list = undir_edges_list
        graph.undirected_edges = undirected_edges
        gc.set_threshold(gc_saved[0])

        return graph

    def set_isolated_atom_response(
        self, on_isolated_atoms: Literal["ignore", "warn", "error"]
    ) -> None:
        """Set the graph converter's response to isolated atom graph
        Args:
            on_isolated_atoms ('ignore' | 'warn' | 'error'): how to handle Structures
                with isolated atoms. Default = 'error'.
        """
        self.on_isolated_atoms = on_isolated_atoms

    def as_dict(self) -> dict[str, str | float]:
        """Save the args of the graph converter."""
        return {
            "atom_graph_cutoff": self.atom_graph_cutoff,
            "bond_graph_cutoff": self.bond_graph_cutoff,
            "algorithm": self.algorithm,
        }

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """Create converter from dictionary."""
        return cls(**dct)
