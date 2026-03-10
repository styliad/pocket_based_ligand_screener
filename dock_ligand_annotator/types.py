from typing import List, Tuple, Union

# Each row: [docked_lgd_idx, interaction_type, ligand_atom_idx, ...]
InteractionRow = List[Union[int, str, Tuple[int, ...], Tuple[str, ...], float]]
InteractionList = List[InteractionRow]

Functional_group = Tuple[str, ...]
