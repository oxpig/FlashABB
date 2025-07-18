import os
import torch
from torch import nn
from torch.nn.functional import one_hot
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from .openfold.np import residue_constants
from .structure_transformer import StructureModule


def atom14_to_atom37(position, aatype):
    from .openfold.utils.feats import (
        atom14_to_atom37 as openfold_atom14_to_atom37,
    )
    from .openfold.data.data_transforms import make_atom14_masks
    position = position.cpu()
    aatype = aatype.cpu()
    batch = make_atom14_masks({"aatype": aatype.squeeze().to(position.device)})
    return openfold_atom14_to_atom37(position.cpu(), batch)


def featurize(seqs, device=torch.device('cuda')):
    features = {}
    clean_seqs = [seq.replace('|', '') for seq in seqs]
    chains = [seq.split('|') for seq in seqs]
    features['aatype'] = [
        torch.tensor([residue_constants.restype_order_with_x[aa] for aa in seq])
        for seq in clean_seqs
    ]
    features['is_heavy'] = [
        torch.tensor([1]*len(heavy) + [0]*len(light))
        for heavy, light in chains
    ]
    features['single'] = [
        torch.cat((
            one_hot(aatype, 21),
            one_hot(is_heavy, 2)
        ), dim=-1).float()
        for aatype, is_heavy in zip(features['aatype'], features['is_heavy'])
    ]
    features['res_idx'] = [
        torch.cat((torch.arange(len(heavy)), torch.arange(len(light)) + 500), dim=-1)
        for heavy, light in chains
    ]
    features['mask'] = [torch.ones_like(aatype) for aatype in features['aatype']]
    for key in features:
        features[key] = pad_sequence(features[key], batch_first=True).to(device)
    return features


class FlashABB(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.model = StructureModule(**params)


class FlashABBResult:
    def __init__(self, seqs, output, mask):
        self.seqs = seqs
        self.output = output
        self.mask = mask

    # @classmethod
    @property
    def coords(self):
        return self.output['positions'][-1,...]

    @property
    def bb_coords(self):
        return self.output['positions'][-1,...,:4,:]

    def to_pdbs(self, names, pdb_dir='.', idxs=None):
        from .openfold.np.protein import Protein, to_pdb
        for i, name in enumerate(names):
            if idxs is not None and i not in idxs:
                continue
            features = featurize(self.seqs)
            residue_idx = features['res_idx'][i].unsqueeze(0)
            aatype = features['aatype'][i].unsqueeze(0)
            coords = self.coords[i]
            coords = atom14_to_atom37(coords, aatype)
            coords = coords.detach().cpu().numpy()
            residue_idx = residue_idx[0,...].detach().cpu().numpy()
            aatype = aatype[0,...].long().detach().cpu().numpy()
            atom_mask = self.mask[i].unsqueeze(dim=-1).expand(coords.shape[:-1])
            atom_mask = atom_mask.detach().cpu().numpy()
            b_factors = np.zeros_like(atom_mask)
            prot = Protein(
                aatype=aatype,
                atom_positions=coords,
                atom_mask=atom_mask,
                residue_index=residue_idx + 1,
                b_factors=b_factors,
                chain_index=(residue_idx >= 500).astype(int),
            )
            pdb_lines = to_pdb(prot)
            os.makedirs(pdb_dir, exist_ok=True)
            with open(f'{pdb_dir}/{names[i]}.pdb', 'w') as f:
                f.write(pdb_lines)
