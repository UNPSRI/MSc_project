# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright Stefano Angioletti-Uberti for added functions (as marked)

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#from abc import ABC, abstractmethod
from typing import List, Optional

import copy
import numpy as np
from scipy.spatial.distance import pdist, squareform
from biotite.structure import annotate_sse, AtomArray, rmsd, sasa, superimpose

from .utilities import get_atomarray_in_residue_range
from .constants import MULTIMER_RESIDUE_INDEX_SKIP_LENGTH
from .my_data_classes import EnergyTerm, FoldingResult


class EnergyPTM(EnergyTerm):
    def __init__(self) -> None:
        super().__init__()

    def compute(self, node, folding_result: FoldingResult) -> float:
        del node
        return 1.0 - folding_result.ptm


class EnergyPLDDT(EnergyTerm):
    def __init__(self) -> None:
        super().__init__()

    def compute(self, node, folding_result: FoldingResult) -> float:
        del node
        return 1.0 - folding_result.plddt


class SymmetryRing(EnergyTerm):
    def __init__(self, all_to_all_protomer_symmetry: bool = False) -> None:
        super().__init__()
        self.all_to_all_protomer_symmetry: bool = all_to_all_protomer_symmetry

    def compute(self, node, folding_result: FoldingResult) -> float:
        protomer_nodes = node.get_children()
        protomer_residue_ranges = [
            protomer_node.get_residue_index_range() for protomer_node in protomer_nodes
        ]

        centers_of_mass = []
        for start, end in protomer_residue_ranges:
            backbone_coordinates = get_backbone_atoms(
                folding_result.atoms[
                    np.logical_and(
                        folding_result.atoms.res_id >= start,
                        folding_result.atoms.res_id < end,
                    )
                ]
            ).coord
            centers_of_mass.append(get_center_of_mass(backbone_coordinates))
        centers_of_mass = np.vstack(centers_of_mass)

        return (
            float(np.std(pairwise_distances(centers_of_mass)))
            if self.all_to_all_protomer_symmetry
            else float(np.std(adjacent_distances(centers_of_mass)))
        )


def get_backbone_atoms(atoms: AtomArray) -> AtomArray:
    return atoms[
        (atoms.atom_name == "CA") | (atoms.atom_name == "N") | (atoms.atom_name == "C")
    ]


def _is_Nx3(array: np.ndarray) -> bool:
    return len(array.shape) == 2 and array.shape[1] == 3


def get_center_of_mass(coordinates: np.ndarray) -> np.ndarray:
    assert _is_Nx3(coordinates), "Coordinates must be Nx3."
    return coordinates.mean(axis=0).reshape(1, 3)


def pairwise_distances(coordinates: np.ndarray) -> np.ndarray:
    assert _is_Nx3(coordinates), "Coordinates must be Nx3."
    m = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
    distance_matrix = np.linalg.norm(m, axis=-1)
    return distance_matrix[np.triu_indices(distance_matrix.shape[0], k=1)]

def adjacent_distances(coordinates: np.ndarray) -> np.ndarray:
    assert _is_Nx3(coordinates), "Coordinates must be Nx3."
    m = coordinates - np.roll(coordinates, shift=1, axis=0)
    return np.linalg.norm(m, axis=-1)


class MinimizeSurfaceHydrophobics(EnergyTerm):
    def __init__(self) -> None:
        super().__init__()

    def compute(self, node, folding_result: FoldingResult) -> float:
        start, end = node.get_residue_index_range()

        return hydrophobic_score(folding_result.atoms, start, end)

#Added by Stefano Angioletti-Uberti
#I need this to minimise use of hydrophobic in specific 
#designable parts of a protein/peptide
class EnergyHydrophobics(EnergyTerm):
    def __init__(self, group ) -> None:
        super().__init__()
        self.group = group

    def compute(self, node, folding_result: FoldingResult) -> float:
        return hydrophobic_score_group(folding_result.atoms, self.group)


_HYDROPHOBICS = {"VAL", "ILE", "LEU", "PHE", "MET", "TRP"}

#Added by Stefano Angioletti-Uberti
#Very similar to hydrophobic_score but does not take start and end but rather a group 
#of residues
def hydrophobic_score_group(
    atom_array: AtomArray,
    group: list
) -> float:
    """
    Computes hydrophobic score by summing over AA in group
    Lower is less hydrophobic (better).
    """

    #Note: atom_array contains all the atoms, so because you can use different amino-acids, the
    #total number of atoms might vary.
    hydrophobic_mask = np.array([aa in _HYDROPHOBICS for aa in atom_array.res_name])
    selection_mask = np.array([ aa in group for aa in range(len(atom_array.res_name)) ])

    #At the end of this, all aminoacids that are both hydrophobic AND in the group have
    #mask = 1, the others have mask = 0.
    hydrophobic_total = ( selection_mask * hydrophobic_mask )
    score = hydrophobic_total.sum() / len( group ) #normalise to get a x Aminoacid number

    return score 



def hydrophobic_score(
    atom_array: AtomArray,
    start_residue_index: Optional[int] = None,
    end_residue_index: Optional[int] = None,
) -> float:
    """
    Computes ratio of hydrophobic atoms in a biotite AtomArray that are also surface
    exposed. Typically, lower is better.
    """

    hydrophobic_mask = np.array([aa in _HYDROPHOBICS for aa in atom_array.res_name])

    if start_residue_index is None and end_residue_index is None:
        selection_mask = np.ones_like(hydrophobic_mask)
    else:
        start_residue_index = 0 if start_residue_index is None else start_residue_index
        end_residue_index = (
            len(hydrophobic_mask) if end_residue_index is None else end_residue_index
        )
        selection_mask = np.array(
            [
                i >= start_residue_index and i < end_residue_index
                for i in range(len(hydrophobic_mask))
            ]
        )

    # TODO(scandido): Resolve the float/bool thing going on here.
    hydrophobic_surf = np.logical_and(
        selection_mask * hydrophobic_mask, sasa(atom_array)
    )
    # TODO(brianhie): Figure out how to handle divide-by-zero.
    return sum(hydrophobic_surf) / sum(selection_mask * hydrophobic_mask)


class MinimizeSurfaceExposure(EnergyTerm):
    def __init__(self) -> None:
        super().__init__()

    def compute(self, node, folding_result: FoldingResult) -> float:
        start, end = node.get_residue_index_range()

        return surface_ratio(folding_result.atoms, list(range(start, end)))


class MaximizeSurfaceExposure(EnergyTerm):
    def __init__(self) -> None:
        super().__init__()

    def compute(self, node, folding_result: FoldingResult) -> float:
        start, end = node.get_residue_index_range()

        return 1.0 - surface_ratio(folding_result.atoms, list(range(start, end)))


def surface_ratio(atom_array: AtomArray, residue_indices: List[int]) -> float:
    """Computes ratio of atoms in specified ratios which are on the protein surface."""

    residue_mask = np.array([res_id in residue_indices for res_id in atom_array.res_id])
    surface = np.logical_and(residue_mask, sasa(atom_array))
    return sum(surface) / sum(residue_mask)


#class MinimizeSurfaceExposure(EnergyTerm):
#    def __init__(self) -> None:
#        super().__init__()
#
#    def compute(self, node, folding_result: FoldingResult) -> float:
#        start, end = node.get_residue_index_range()
#
#        return surface_ratio(folding_result.atoms, list(range(start, end)))


#class MaximizeSurfaceExposure(EnergyTerm):
#    def __init__(self) -> None:
#        super().__init__()
#
#    def compute(self, node, folding_result: FoldingResult) -> float:
#        start, end = node.get_residue_index_range()
#
#        return 1.0 - surface_ratio(folding_result.atoms, list(range(start, end)))


#def surface_ratio(atom_array: AtomArray, residue_indices: List[int]) -> float:
#    """Computes ratio of atoms in specified ratios which are on the protein surface."""
#
#    residue_mask = np.array([res_id in residue_indices for res_id in atom_array.res_id])
#    surface = np.logical_and(residue_mask, sasa(atom_array))
#    return sum(surface) / sum(residue_mask)


#Added by Stefano Angioletti-Uberti
class NormalizedDistance(EnergyTerm):
    """Calculates the distance between the com of two groups of atoms (only using backbone atoms)"""
    def __init__(self, group_X, group_Y, norm_X : bool = True, norm_Y : bool = True ) -> None:
        super().__init__()
        self.group_X = group_X
        self.group_Y = group_Y
        self.norm_X = norm_X
        self.norm_Y = norm_Y

    def compute(self, node, folding_result: FoldingResult) -> float:
        assert False,"""This function is deprecated because it might fail when used with children
                      that are not on the same chain!"""

        input_array = folding_result.atoms.res_id

        print( f"All residues ID - Normalized_Distance {input_array}" )
        print( "- Check, is there a 1000 hole in indices above due to offset?" )

        specific_set = set( self.group_X )
        mask = np.isin( input_array, list(specific_set) )

        X_coord = get_backbone_atoms(
                  folding_result.atoms[ mask ]
                  ).coord


        #X_coord = get_backbone_atoms(
        #    	  folding_result.atoms[ folding_result.atoms.res_id in self.group_X ]
        #	  ).coord

        #print( f"Atoms X: {folding_result.atoms[ mask ] }" )

        input_array = folding_result.atoms.res_id
        specific_set = set( self.group_Y )
        mask = np.isin( input_array, list(specific_set) )

        Y_coord = get_backbone_atoms(
                  folding_result.atoms[ mask ]
                  ).coord

        #print( f"Atoms Y: {folding_result.atoms[ mask ] }" )
        
        com_X = get_center_of_mass( coordinates = X_coord )
        com_Y = get_center_of_mass( coordinates = Y_coord )

        #Calculate the distance normalised by atoms in group, if their norm is True
        norm = len( X_coord ) * self.norm_X
        norm += len( Y_coord ) * self.norm_Y
        dist = float( np.linalg.norm( com_X - com_Y ) / norm ) 

        return dist

#Added by Stefano Angioletti-Uberti
class AverageDistance( EnergyTerm ):
    """Calculates the average distance between two groups of atoms (only using backbone atoms)"""
    def __init__(self, group_X, group_Y, weighting_power : float = 1.0 ) -> None:
        super().__init__()
        self.group_X = group_X
        self.group_Y = group_Y
        self.weighting_power = weighting_power # take distance**weighting_power when calculating 

    def compute(self, node, folding_result: FoldingResult) -> float:
        
        input_array = folding_result.atoms.res_id
        mask_X = np.isin( input_array, self.group_X )

        X_coord = get_backbone_atoms(
                  folding_result.atoms[ mask_X ]
                  ).coord
        
        mask_Y = np.isin( input_array, self.group_Y ) 

        Y_coord = get_backbone_atoms(
                  folding_result.atoms[ mask_Y ]
                  ).coord

        NX = len( X_coord )

        coord = np.vstack( (X_coord, Y_coord) )
        pairwise = pdist( coord )
        distance_matrix = squareform(pairwise) 
        distance_matrix = distance_matrix**self.weighting_power
        #Now remove "spurious distances between AA that are in the same group (set them to zero)"
        #Basically you remove all the blocks on the diagonal and all that remains is distances 
        #between AA of different groups, it is correct - rechecked multiple time although it might
        #seem strange!
        distance_matrix[ 0: NX, 0:NX ] = 0.0   
        distance_matrix[ NX: , NX: ] = 0.0   
        mask = distance_matrix == 0.0 #So that average is calculated on non-zero elements
        masked_data = np.ma.masked_array( distance_matrix, mask = mask )        
        d_ave = np.mean( masked_data )
     
        #print( f"Average distance between groups is {d_ave}" )
      
        return d_ave 

#def get_center_of_mass(coordinates: np.ndarray) -> np.ndarray:
#    assert _is_Nx3(coordinates), "Coordinates must be Nx3."
#    return coordinates.mean(axis=0).reshape(1, 3)
        

#Added by Stefano Angioletti-Uberti
class LeafToLeafDistance( EnergyTerm ):
    """Calculates the distance between the com of two groups of atoms (only using backbone atoms)"""
    def __init__( self, leaf1 : list, leaf2 : list, norm : bool = True ) -> None:
        super().__init__()
        self.leaf1 = leaf1
        self.leaf2 = leaf2
        self.norm = norm

    def compute(self, node, folding_result: FoldingResult) -> float:

        leaf1 = copy.deepcopy( self.leaf1 )
        first_leaf = node.retrieve_child_leaf( leaf_pos = leaf1 ) 

        start, end = first_leaf.get_residue_index_range()
        leaf1_coord = get_backbone_atoms(
            folding_result.atoms[
                np.logical_and(
                    folding_result.atoms.res_id >= start,
                    folding_result.atoms.res_id < end,
                ) 
            ]     
        ).coord

        #print( f"Start and end of residues for calculation of distance {start} {end}" )
        
        leaf2 = copy.deepcopy( self.leaf2 )
        second_leaf = node.retrieve_child_leaf( leaf_pos = leaf2 ) 
        start, end = second_leaf.get_residue_index_range()
        leaf2_coord = get_backbone_atoms(
            folding_result.atoms[
                np.logical_and(
                    folding_result.atoms.res_id >= start,
                    folding_result.atoms.res_id < end,
                ) 
            ]     
        ).coord

        #print( f"Start and end of residues for calculation of distance {start} {end}" )
        
        com_1 = get_center_of_mass( coordinates = leaf1_coord )
        com_2 = get_center_of_mass( coordinates = leaf2_coord )
        dist = float( np.linalg.norm( com_1 - com_2 ) ) 

        #Calculate the distance normalised by atoms in group, if their norm is True
        norm_dist = len( com_1 ) * self.norm
        norm_dist += len( com_2 ) * self.norm
        dist /= norm_dist 

        return dist



class MaximizeGlobularity(EnergyTerm):
    def __init__(self) -> None:
        super().__init__()

    def compute(self, node, folding_result: FoldingResult) -> float:
        start, end = node.get_residue_index_range()

        backbone = get_backbone_atoms(
            folding_result.atoms[
                np.logical_and(
                    folding_result.atoms.res_id >= start,
                    folding_result.atoms.res_id < end,
                )
            ]
        ).coord

        return float(np.std(distances_to_centroid(backbone)))


def distances_to_centroid(coordinates: np.ndarray) -> np.ndarray:
    """
    Computes the distances from each of the coordinates to the
    centroid of all coordinates.
    """
    assert _is_Nx3(coordinates), "Coordinates must be Nx3."
    center_of_mass = get_center_of_mass(coordinates)
    m = coordinates - center_of_mass
    return np.linalg.norm(m, axis=-1)


#This is used to minimise the difference in the rmsd of a template vs
#a prediction
class MinimizeCRmsd(EnergyTerm):
    def __init__(self, template: AtomArray, backbone_only: bool = False) -> None:
        super().__init__()

        self.template: AtomArray = template
        self.backbone_only: bool = backbone_only
        if self.backbone_only:
            self.template = get_backbone_atoms(template)

    def compute(self, node, folding_result: FoldingResult) -> float:
        start, end = node.get_residue_index_range()

        atoms = get_atomarray_in_residue_range(folding_result.atoms, start, end)

        if self.backbone_only:
            atoms = get_backbone_atoms(atoms)

        return crmsd(self.template, atoms)

#This is used to minimise the difference in the rmsd of a template vs
#a prediction
#Simply renamed compared to original definition by Stefano Angioletti-Uberti to remember it is 
#equivalent to an energy
class EnergyCRmsd(EnergyTerm):
    def __init__(self, template: AtomArray, backbone_only: bool = False) -> None:
        super().__init__()

        self.template: AtomArray = template
        self.backbone_only: bool = backbone_only
        if self.backbone_only:
            self.template = get_backbone_atoms(template)

    def compute(self, node, folding_result: FoldingResult) -> float:
        start, end = node.get_residue_index_range()

        atoms = get_atomarray_in_residue_range(folding_result.atoms, start, end)

        if self.backbone_only:
            atoms = get_backbone_atoms(atoms)

        return crmsd(self.template, atoms)


def crmsd(atom_array_a: AtomArray, atom_array_b: AtomArray) -> float:
    # TODO(scandido): Add this back.
    # atom_array_a = canonicalize_within_residue_atom_order(atom_array_a)
    # atom_array_b = canonicalize_within_residue_atom_order(atom_array_b)
    superimposed_atom_array_b_onto_a, _ = superimpose(atom_array_a, atom_array_b)
    return float(rmsd(atom_array_a, superimposed_atom_array_b_onto_a).mean())


#This is used to minimise the difference in the distogram of a template vs
#a prediction
class MinimizeDRmsd(EnergyTerm):
    def __init__(self, template: AtomArray, backbone_only: bool = False) -> None:
        super().__init__()

        self.template: AtomArray = template
        self.backbone_only: bool = backbone_only
        if self.backbone_only:
            self.template = get_backbone_atoms(template)

    def compute(self, node, folding_result: FoldingResult) -> float:
        start, end = node.get_residue_index_range()

        atoms = get_atomarray_in_residue_range(folding_result.atoms, start, end)

        if self.backbone_only:
            atoms = get_backbone_atoms(atoms)

        return drmsd(self.template, atoms)


#Simply renamed compared to original definition by Stefano Angioletti-Uberti
#to remember it is equivalent to an energy
class EnergyDRmsdLocal(EnergyTerm):
    def __init__(self, template: AtomArray, backbone_only: bool = False) -> None:
        super().__init__()

        self.template: AtomArray = template
        self.backbone_only: bool = backbone_only
        if self.backbone_only:
            self.template = get_backbone_atoms(template)

    def compute(self, node, folding_result: FoldingResult) -> float:
        start, end = node.get_residue_index_range()

        atoms = get_atomarray_in_residue_range(folding_result.atoms, start, end)

        if self.backbone_only:
            atoms = get_backbone_atoms(atoms)

        return drmsd(self.template, atoms)


def drmsd(atom_array_a: AtomArray, atom_array_b: AtomArray) -> float:
    # TODO(scandido): Add this back.
    # atom_array_a = canonicalize_within_residue_atom_order(atom_array_a)
    # atom_array_b = canonicalize_within_residue_atom_order(atom_array_b)

    dp = pairwise_distances(atom_array_a.coord)
    dq = pairwise_distances(atom_array_b.coord)

    return float(np.sqrt(((dp - dq) ** 2).mean()))


#def pairwise_distances(coordinates: np.ndarray) -> np.ndarray:
#    assert _is_Nx3(coordinates), "Coordinates must be Nx3."
#    m = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
#    distance_matrix = np.linalg.norm(m, axis=-1)
#    return distance_matrix[np.triu_indices(distance_matrix.shape[0], k=1)]


class MatchSecondaryStructure(EnergyTerm):
    def __init__(self, secondary_structure_element: str) -> None:
        super().__init__()
        self.secondary_structure_element = secondary_structure_element

    def compute(self, node, folding_result: FoldingResult) -> float:
        start, end = node.get_residue_index_range()

        subprotein = folding_result.atoms[
            np.logical_and(
                folding_result.atoms.res_id >= start,
                folding_result.atoms.res_id < end,
            )
        ]
        sse = annotate_sse(subprotein)

        return np.mean(sse != self.secondary_structure_element)



#Added by Stefano Angioletti-Uberti
#This could be useful to actually force binding interface
class EnergyBinderPAE( EnergyTerm ):
    #Minimise Predicted Alignment Error
    #if 
    def __init__( self, groupA : list = [], groupB : list = [], mode : str = "all",
                  offset = MULTIMER_RESIDUE_INDEX_SKIP_LENGTH, debug = False ) -> None:
                  #NOTE: offset is necessary because there is a gap of "offset" between
                  #[0:protein_length] to [protein_length:protein_length+binder_length]
                  #so the residues_id of the binder are shifted by +offset compared to their
                  #positional index "i" in folding_result.pae[ i ] 
        super().__init__()
        self.groupA = list( groupA ) #Identity of AA to be considered as the epitope/binding spot
        self.groupB = list( groupB ) #Identity of AA to be considered as the binder
        self.NA = len( self.groupA )
        self.NB = len( self.groupB )
        self.mode = mode
        self.offset = offset 
        self.mask_calculated = False
        self.debug = debug

    def compute( self, node, folding_result: FoldingResult ) -> float:
        #Compute tha mask only the first time, then it will be fixed and not needed

        #If self.mask has been defined, just go to end of if and calculate value with this mask
        if self.mask_calculated:
            pass 
        else: 
          offset = self.offset

          N = len( folding_result.pae[ 0 ][ 0, : ] )
          assert N == len( folding_result.pae[ 0 ][ :, 0 ] ) #Just to check pae is indeed square as expected...
          mask = np.ones( ( N, N ) ) #All masked by default, then you unmask those pairs you really want to check
  
          if self.mode == "limited": #consider only PAE between AA (i,j) where i in groupA, j in groupB
            
            for i in self.groupA:
              for j in self.groupB:
                if self.debug:
                    print( f" Consider candidate {i} of groupA and candidate {j} of groupB " )
                try:
                  mask[ i, j - offset ] = 0
                  mask[ j - offset, i ] = 0
                except IndexError:
                  raise IndexError( "Out of range maybe, check offset value in binderPAE (language.py)" )

          elif self.mode == "all":
            groupAB = set( self.groupA + self.groupB )
            for i in groupAB:
              for j in groupAB:
                try:
                  mask[ i, j - offset ] = 0
                  mask[ j - offset, i ] = 0
                except IndexError:
                  raise IndexError( "Out of range maybe, check offset value in binderPAE (language.py)" )

          else:
            raise ValueError( "Error in binderEnergy function, mode not recognised" )

          self.mask = mask
          self.mask_calculated = True #So you do not recalculate again anymore


        energy = calculate_PAE( folding_result = folding_result, AA_mask = self.mask  ) 

        return energy 

#Added by Stefano Angioletti-Uberti
#This could be useful to actually force binding interface
class GeneralEnergyBinderPAE( EnergyTerm ):
    #Minimise Predicted Alignment Error
    #if 
    def __init__( self, group1 : list = [], group2 : list = [], chain_1_num : int = 0, chain_2_num : int = 0, mode : str = "all",
                  debug = False ) -> None:
        super().__init__()
        self.group1 = list( group1 ) #Identity of AA to be considered as the epitope/binding spot
        self.group2 = list( group2 ) #Identity of AA to be considered as the binder
        self.chain_1_num = chain_1_num
        self.chain_2_num = chain_2_num
        self.N1 = len( self.group1 )
        self.N2 = len( self.group2 )
        self.mode = mode
        #NOTE: offset is necessary because there is a gap of "offset" between
        #[0:protein_length] to [protein_length:protein_length+binder_length]
        #so the residues_id of the binder are shifted by +offset compared to their
        #positional index "i" in folding_result.pae[ i ] 
        self.offset = MULTIMER_RESIDUE_INDEX_SKIP_LENGTH
        self.mask_calculated = False
        self.debug = debug

    def compute( self, node, folding_result: FoldingResult ) -> float:
        #Compute tha mask only the first time, then it will be fixed and not needed

        #If self.mask has been defined, just go to end of if and calculate value with this mask
        if self.mask_calculated:
            pass 
        else: 
          offset = self.offset

          N = len( folding_result.pae[ 0 ][ 0, : ] )
          assert N == len( folding_result.pae[ 0 ][ :, 0 ] ) #Just to check pae is indeed square as expected...
          mask = np.ones( ( N, N ) ) #All masked by default, then you unmask those pairs you really want to check
  
          if self.mode == "limited": #consider only PAE between AA (i,j) where i in group1, j in group2
            
            for i in self.group1:
              for j in self.group2:
                if self.debug:
                    print( f" Consider candidate {i} of group1 and candidate {j} of group2 " )
                try:
                  ii_s = i - offset * self.chain_1_num
                  jj_s = j - offset * self.chain_2_num
                  mask[ ii_s, jj_s ] = 0
                  mask[ jj_s, ii_s ] = 0
                except IndexError:
                  raise IndexError( "Out of range maybe, check offset value in binderPAE (language.py)" )

          elif self.mode == "all":
            group12 = set( self.group1 + self.group2 )
            for i in group12:
              for j in group12:
                try:
                  ii_s = i - offset * self.chain_1_num
                  jj_s = j - offset * self.chain_2_num
                  mask[ ii_s, jj_s ] = 0
                  mask[ jj_s, ii_s ] = 0
                except IndexError:
                  raise IndexError( "Out of range maybe, check offset value in binderPAE (language.py)" )

          else:
            raise ValueError( "Error in binderEnergy function, mode not recognised" )

          self.mask = mask
          self.mask_calculated = True #So you do not recalculate again anymore


        energy = calculate_PAE( folding_result = folding_result, AA_mask = self.mask  ) 

        return energy 

def calculate_PAE( folding_result : FoldingResult, AA_mask : np.array  ) -> float:
    """Compute the average predicted alignment error between two groups of aminoacids.
    This is useful to force a specific binding interface.
    Basically, calculates the AVERAGE predicted alignment error, where the average is over
    a set of aminoacid pairs specified by mask"""
    
    data = folding_result.pae[0]

    #print( f"calculate_PAE fun, length of data: {len(data)}" )
    mask = AA_mask 
    masked_data = np.ma.masked_array( data, mask = mask )       

    #print( f"Masked_data {masked_data}" ) 
    result = np.mean( masked_data ) / 30.0 #30.0 is the maximum value for PAE between two pairs
    return result 


#Added by Stefano Angioletti-Uberti
#This could be useful when protein is well-defined and won't change much upon binding,
#so only binder characteristics are important
class EnergyLocalPLDDT( EnergyTerm ):
    #Minimise Predicted Alignment Error
    #if 
    def __init__( self, group : list = [], chain_index : int = 0, offset: int = MULTIMER_RESIDUE_INDEX_SKIP_LENGTH,
                  verbose = False ) -> None:
                  #NOTE: offset is necessary because there is a gap of "offset" between
                  #[0:protein_length] to [protein_length:protein_length+binder_length]
                  #so the residues_id of the binder are shifted by +offset compared to their
                  #positional index "i" in folding_result.plddt[ i ]  OR SO I ASSUME. It is true
                  #for folding_result.pae
        super().__init__()
        self.group = list( group ) #Identity of AA to be considered 
        self.NA = len( self.group )
        self.offset = offset 
        self.chain_index = chain_index
        self.mask_calculated = False
        self.verbose = verbose

    def compute( self, node, folding_result: FoldingResult ) -> float:

        data = folding_result.local_plddt
        
        #Compute tha mask only the first time, then it will be fixed and not needed
        #If self.mask has been defined, just go to end of if and calculate value with this mask
        if self.mask_calculated:
            pass 
        else: 
          N = len( data )
          self.mask = np.ones( ( N ) ) #All masked by default, then you unmask those pairs you really want to check
          
          offset = self.chain_index * self.offset #The offset for the positional embedding is added 
                                                  #at the end of each chain so this removes it
          for i in self.group:
                try:
                  self.mask[ i - offset ] = 0 #Unmask necessary elements
                except KeyError:
                  raise KeyError( "Out of range maybe, check offset value in localPLDDT (language.py)" )
          self.mask_calculated = True

        #Define masked array
        masked_data = np.ma.masked_array( data, mask = self.mask )       

        if self.verbose: 
            print( f"Mask used for data {masked_data.mask}" ) 
            print( f"PLDDT {masked_data}" ) 

        loss = 1.0 - np.mean( masked_data ) #if plddt = 1.0, then loss must be at a minimum 

        return loss 


##Added by Unnop Srikulwong
##Corrected by Stefano Angioletti-Uberti
##Minimize protein structure to the target shape of hypercube
class EnergyBrick(EnergyTerm):
    """INPUT:
    a,b,c: the length of the axis a, b and c determining the super-cube
    m,n,p: the exponents of the supercube that control the sharpness
    supercube equation" (x/a)^2m + (y/b)^2n + (z/c)^2p
    """
    def __init__(self, a : float = 1.0, b : float = 1.0, c : float = 1.0, 
                       m : int = 2, n : int = 2, p : int = 2,
                       verbose : bool = False,
                       k_cube : float = 10.0,
                       soft_rep : bool = True, #Use a soft repulsion by default
                       power_rep : float = 2.0, #Only used if soft_rep == False to determine repulsive potential
                       A_rep : float = 10.0,) -> None:

        super().__init__()
        #Note for Unnop: you want a,b and c to be specified by the user so you 
        #can make bricks with the shape/anisotropy that you want.
        self.a = a
        self.b = b
        self.c = c
        self.m = m
        self.n = n
        self.p = p
        self.verbose = verbose
        self.k_cube = k_cube
        self.A_rep = A_rep
        self.soft_rep = soft_rep
        self.power_rep = power_rep
        
    def compute(self, node, folding_result: FoldingResult) -> float:

        input_array = folding_result.atoms.res_id
        
        position = get_backbone_atoms(
                      folding_result.atoms[input_array]).coord
       
        #Rescale every position so COM is in the origin
        position = position - get_center_of_mass(position)
        
        ### PART 1 ###
	
    	#Radius of gyration 
        #Note that the gyration radius is proportional
        #to Na^2 but the proportionality factor is not 1.0, so we correct for this.
        R_g = 1.0 / np.sqrt( 6.0 ) * np.sqrt(len(position)) * 0.5 # Distance between two monomers ~ 0.5 nm

        #We now basically make sure that a, b and c are relative to the approximate 
        #gyration radius that a protein would have IF it was of a given length and 
        #had an overall spherical shape. 
        a = self.a / self.c * R_g
        b = self.b / self.c * R_g
        c = R_g

        #We make this using vectorial operations
        x = position[ :, 0 ] / a
        y = position[ :, 1 ] / b
        z = position[ :, 2 ] / c

        #For a point on the hyper_cube, rel_dist = 0.0, for a point outside it, rel_dist > 0.0
        #We basically penalise points that are OUTside of the supercube 
        rel_distance = (x**(2*self.m) + y**(2*self.n) + z**(2*self.p)) - 1.0

        #contribution will be equal to True == 1.0 if the condition is satisfied, False = 0.0 otherwise
        #We basically exploit the fact that in numpy arrays, True is literally interpreted as 1, and False as 0
        contribution =  rel_distance > 0.0
    
        #If the particle is outside of the super-cube, you add a spring to bring it back, otherwise no penalty
        loss1 = ( contribution * self.k_cube ).sum()
        
        if self.verbose:
            print( "Rel_distance" )
            print( f"{rel_distance}" )
            print( f"LOSS-cube = {loss1}" )

        ### PART 2 ###

        #Now calculate repulsive contribution, this should distribute particles homogeneously within the volume
        r_ij = pdist( position )  
        lambda_rep = np.max( [ a, b, c ] )
        #Allow to decide which potential to use
        if self.soft_rep:
          loss2 = self.A_rep * np.exp( -r_ij / lambda_rep  ).sum()   
        else:
          loss2 = self.A_rep * ( 1.0 / r_ij**self.power_rep ).sum()   
        if self.verbose:
          print( f"Energy from repulsion {loss2}" )
        
        return loss1 + loss2 

##Added by Stefano based on Unnop work
##Minimize protein structure to the target shape of hypercube
class EnergyBrickInvariant(EnergyTerm):
    """INPUT:
    a,b,c: the length of the axis a, b and c determining the super-cube
    m,n,p: the exponents of the supercube that control the sharpness
    supercube equation" (x/a)^2m + (y/b)^2n + (z/c)^2p
    """
    def __init__(self, a : float = 1.0, b : float = 1.0, c : float = 1.0, 
                       m : int = 2, n : int = 2, p : int = 2,
                       verbose : bool = False,
                       k_cube : float = 1.0,
                       soft_rep : bool = True, #Use a soft repulsion by default
                       power_rep : float = 2.0, #Only used if soft_rep == False to determine repulsive potential
                       A_rep : float = 10.0,) -> None:

        super().__init__()
        #Note for Unnop: you want a,b and c to be specified by the user so you 
        #can make bricks with the shape/anisotropy that you want.
        self.a = a
        self.b = b
        self.c = c
        self.m = m
        self.n = n
        self.p = p
        self.verbose = verbose
        self.k_cube = k_cube
        self.A_rep = A_rep
        self.soft_rep = soft_rep
        self.power_rep = power_rep
        
        #Basically, re-order the values of a, b, c so that a is always the smallest, c the largest
        #This is necessary for the scaling operation later as the z direction always corresponds to that
        #of largest eigenvalue / variance along its axis
        values = np.array( [ self.a, self.b, self.c ] )
        sorted_indices = np.argsort(values)[::-1]
        sorted_indices = np.array(sorted_indices,dtype=int)
        self.c, self.b, self.a = values[sorted_indices]

        #Sanity check
        assert self.c >= self.a, AssertionError( f"Expected c>=a but found c = {self.c} and a = {self.a}" )
        assert self.c >= self.b, AssertionError( f"Expected c>=b but found c = {self.c} and b = {self.b}" ) 
        assert self.b >= self.a, AssertionError( f"Expected b>=a but found b = {self.b} and a = {self.a}" )
        
    def compute(self, node, folding_result: FoldingResult) -> float:

        input_array = folding_result.atoms.res_id
        
        position = get_backbone_atoms(
                      folding_result.atoms[input_array]).coord
        
        #if self.verbose:
        #    print( "Every 10 positions" )
        #    print( f"{position[::10]}" )
        #    print( "Type" )
        #    print( f"{type( position )}")

        #Now align the cartesian axes with the principal axes and transform all the coordinates accordingly.
        #This is needed to make the program invariant to rotations and translations of the protein.
        #Note that pca also re-aligns positions so that COM is at origin
        position, _, _ = pca_align( position )
        #After this transformation, z is aligned with the axis of highest variance, y the second highest, x with the last!
        #You need to take this into account going forward when comparing to the super-cube
        
        ### PART 1 ###
	
    	#Radius of gyration 
        #Note that the gyration radius is proportional
        #to Na^2 but the proportionality factor is not 1.0, so we correct for this.
        R_g = 1.0 / np.sqrt( 6.0 ) * np.sqrt(len(position)) * 0.5 # Distance between two monomers ~ 0.5 nm

        #We now basically make sure that a, b and c are relative to the approximate 
        #gyration radius that a protein would have IF it was of a given length and 
        #had an overall spherical shape. 
        a = R_g
        b = self.b / self.a * R_g
        c = self.c / self.a * R_g

        #We make this using vectorial operations
        x = position[ :, 0 ] / a
        y = position[ :, 1 ] / b
        z = position[ :, 2 ] / c

        #For a point on the hyper_cube, rel_dist = 0.0, for a point outside it, rel_dist > 0.0
        #We basically penalise points that are OUTside of the supercube 
        rel_distance = (x**(2*self.m) + y**(2*self.n) + z**(2*self.p)) - 1.0

        #contribution will be equal to True == 1.0 if the condition is satisfied, False = 0.0 otherwise
        #We basically exploit the fact that in numpy arrays, True is literally interpreted as 1, and False as 0
        contribution =  rel_distance > 0.0

    
        #If the particle is outside of the super-cube, you add a spring to bring it back, otherwise no penalty
        loss1 = ( contribution * self.k_cube ).sum()
        
        if self.verbose:
            #print( "Rel_distance (every 10)" )
            #print( f"{rel_distance[::10]}" )
            print( f"LOSS-cube = {loss1}" )

        ### PART 2 ###

        #Now calculate repulsive contribution, this should distribute particles homogeneously within the volume
        r_ij = pdist( position )  
        lambda_rep = np.max( [ a, b, c ] )
        #Allow to decide which potential to use
        if self.soft_rep:
          loss2 = self.A_rep * np.exp( -r_ij / lambda_rep  ).mean()  
        else:
          loss2 = self.A_rep * ( 1.0 / r_ij**self.power_rep ).mean()   
        if self.verbose:
            #print( f"Number of pairs in distances:   {len(r_ij)}" )
            print( f"Energy from PAIR repulsion {loss2}" )
        
        return loss1 + loss2

##Added by Stefano Angiolett-Uberti 
##Force formation of asymmetric protein
class EnergyEllipsoid(EnergyTerm):
    """INPUT:
    a,b,c: the length of the axis a, b and c determining the ellipsoid 
    """
    def __init__(self, a : float = 1.0, b : float = 1.0, c : float = 1.0, 
                       verbose : bool = False,
                       k_spring : float = 1.0,
                       ) -> None:

        super().__init__()
        #Note for Unnop: you want a,b and c to be specified by the user so you 
        #can make bricks with the shape/anisotropy that you want.
        self.c = 1.0
        self.a = a / c
        self.b = b / c
        self.verbose = verbose
        self.k_spring = k_spring

        #Now just order the eigenvalues
        values = np.array( [ self.a, self.b, self.c ] )
        sorted_indices = np.argsort(values)[::-1]
        sorted_indices = np.array(sorted_indices,dtype=int)
        self.sorted_values = values[sorted_indices]
        
    def compute(self, node, folding_result: FoldingResult) -> float:

        input_array = folding_result.atoms.res_id
        
        position = get_backbone_atoms(
                      folding_result.atoms[input_array]).coord
        
        #Now calculate the asymmetry
        position, sorted_eigenvalues, sorted_eigenvectors = pca_align( position )
        sorted_eigenvalues = sorted_eigenvalues / np.min( sorted_eigenvalues )
        
        energy = self.k_spring * (( self.sorted_values - sorted_eigenvalues )**2).sum()

        if self.verbose:
            print( f"sorted_eigenvectors {sorted_eigenvectors}" )
            print( f"sorted_eigenvalues {sorted_eigenvalues}" )
            print( f"LOSS-cube = {energy}" )

        return energy 


def pca_align( points : np.ndarray ):
    # Step 1: Convert the list of points to a NumPy array if needed
    X = np.array(points)

    # Step 2: Mean centering the data
    mu = np.mean(X, axis=0)
    X_centered = X - mu

    # Step 3: Compute the covariance matrix
    C = np.cov(X_centered, rowvar=False)

    # Step 4: Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(C)

    # Step 5: Sort the eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Step 6: Reorder eigenvectors to align with z-y-x convention
    P = sorted_eigenvectors[:, [2, 1, 0]]  # Assuming sorted_eigenvalues are in ascending order

    # Step 7: Transform the original data to the new coordinate system
    X_transformed = np.dot(X_centered, P)

    return X_transformed, sorted_eigenvalues, P
