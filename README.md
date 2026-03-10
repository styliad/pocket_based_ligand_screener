# docked_ligand_functional_group_annotator
Annotate ligand functional group - residue interactions for docked ligand poses

# Installation
1. Go to the installation directory
2. In the installation dir:
+ Run:
```python
gh repo clone styliad/docked_ligand_functional_group_annotator
```
OR 
+ Download manually
3. Run
```
poetry install
```
4. Add package `dock_ligand_annotator` to your PATH

# TO DO


- [X] Complete the 'Understanding Prolif interaction fingerprint (plf.Fingerprint.ifp) data structure'

- [X] Check that that atom indices match between the interactions and the functional groups

- [ ] identify_functional_group code, doesn't treat phenyl rings as fg - Add relevant SMARTS - edit their code
    - [ ] Are there other ring systems not recognised as well?    

- [ ] Get residue atom type with internal residue numbering - It was throwing an error
                internal_residue_atoms = interaction['indices']['protein']
                print(residue_num)
                # residue_atom_types =  tuple([protein_mol.residues[residue_num].GetAtomWithIdx(atm_idx).GetSymbol() for atm_idx in internal_residue_atoms])

#### Include bond type-specific information
- [ ] For Pi-Pi interactions include also angle?

#### Extra annotations
##### Lipophiic hotspots
- [ ] Calculation for lipophilic hotspots - Checks against interaction with residue
- [ ] Add to check whenever a non-FG atom occupies the hotspot

#### Tests
- [ ] Include files other than Glide docking

# INPUT FILES
1. Protein (.pdb)
2. Docked ligands (.sdf)
3. Config file (.yaml)

## For config file
### Small number of protein-ligand pairs
Manually edit the config file and add related paths

### Large number of protein-ligand pairs
Run create_config_file(protein_ligand_dict), where protein_ligand_dict is a dictionary where keys are protein file paths and values are ligand file paths.

## END RESULT - wide-format table
### Type of result
(internal_docked_ligand_id, interaction_type, (ligand_atom_indices), (ligand_atom_types),residue_type, residue_number, residue_atom_number, residue_atom_types, residue_atom_backbone_or_sidechain, distance (ligand_fg_atoms, ligand_fg_type))

### Example
[2, 'HBDonor', (3, 30), ('N', 'H'), 'SER', 188, (2470,), ('O',), ('bb',), 3.493, (('NC=O', 'cNC(C)=O'), 'No_fg')]

## Ensuring that atom indices are the same during the calculation of interactions and functional groups
Docked ligands are loaded only once. It is based on the prolif.molecule.Molecule class. This object is used for both the calculations of interactions and functional groups.
In molecular viewer software like Maestro, where the numbering begins from 1 instead of the pythonic 0, one should considering adding +1 to make atom indices equivalent.