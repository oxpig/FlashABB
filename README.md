# FlashABB: modelling antibody structures at the speed of language

![Inference speed comparison](figures/speedup_multiplier.png)

Installation:

```bash
pip install flash-abb
```

Or from source:

```bash
git clone git@github.com:oxpig/FlashABB.git
cd FlashABB
pip install .
```

## Structure prediction

The following is also in `example.py` and can be used to create the structures in `sample_preds`.

Predictions apply the Dropout-LayerNorm Correction (DLC) by default, which improves accuracy at no extra cost ([preprint](https://arxiv.org/abs/2609.32062)); disable it with `pretrained(dlc_correction=False)`.

```python
from flash_abb import pretrained
import torch

flabb = pretrained(device='cuda')

# Sequences in heavy|light format
seqs = [
    'EVQLLESGGEVKKPGASVKVSCRASGYTFRNYGLTWVRQAPGQGLEWMGWISAYNGNTNYAQKFQGRVTLTTDTSTSTAYMELRSLRSDDTAVYFCARDVPGHGAAFMDVWGTGTTVTVSS|DIQLTQSPLSLPVTLGQPASISCRSSQSLEASDTNIYLSWFQQRPGQSPRRLIYKISNRDSGVPDRFSGSGSGTHFTLRISRVEADDVAVYYCMQGTHWPPAFGQGTKVDIK',
    'EVQLLESGGEVKKPGASVKVSCRASGYTFRNYGLTWVRQAPGQGLEWMGWISAYNGNTNYAQKFQGRVTLTTDTSTSTAYMELRSLRSDDTAVYFCARDVPGHGAAFMDVWGTGTTVTVS|DIQLTQSPLSLPVTLGQPASISCRSSQSLEASDTNIYLSWFQQRPGQSPRRLIYKISNRDSGVPDRFSGSGSGTHFTLRISRVEADDVAVYYCMQGTHWPPAFGQGTKVDIK',
]

with torch.no_grad():
    result = flabb(seqs)

print(result.coords.shape)          # (2, n_residues, 14, 3)
print(result.bb_coords.shape)       # (2, n_residues, 4, 3)

result.to_pdbs(['ab1', 'ab2'], pdb_dir='sample_preds')
```

## Developability scoring (FlashTAP)

FlashTAP predicts four [TAP](https://doi.org/10.1038/s42003-023-05744-8) developability scores: PSH, PPC, PNC, and SFvCSP.

```python
from flash_abb import pretrained_tap

tap = pretrained_tap(device='cuda')

seqs = [
    'EVQLLESGGEVKKPGASVKVSCRASGYTFRNYGLTWVRQAPGQGLEWMGWISAYNGNTNYAQKFQGRVTLTTDTSTSTAYMELRSLRSDDTAVYFCARDVPGHGAAFMDVWGTGTTVTVSS|DIQLTQSPLSLPVTLGQPASISCRSSQSLEASDTNIYLSWFQQRPGQSPRRLIYKISNRDSGVPDRFSGSGSGTHFTLRISRVEADDVAVYYCMQGTHWPPAFGQGTKVDIK',
]

result = tap(seqs)
print(result.scores)        # [{'PSH': ..., 'PPC': ..., 'PNC': ..., 'SFvCSP': ...}]
print(result.tensor)        # (1, 4) raw score tensor
print(result.flag_probs)    # [{'PSH': 0.12, 'PPC': 0.03, 'PNC': 0.05, 'SFvCSP': 0.41}]
print(result.any_flag_prob) # [0.47]
```

## Structure-aware embeddings (FlashABB-SSS)

FlashABB-SSS (seq2struct2seq) produces per-residue embeddings that combine sequence and predicted 3D structure. These can be used as features for downstream tasks.

```python
from flash_abb import pretrained_sss

sss = pretrained_sss(device='cuda')

seqs = [
    'EVQLLESGGEVKKPGASVKVSCRASGYTFRNYGLTWVRQAPGQGLEWMGWISAYNGNTNYAQKFQGRVTLTTDTSTSTAYMELRSLRSDDTAVYFCARDVPGHGAAFMDVWGTGTTVTVSS|DIQLTQSPLSLPVTLGQPASISCRSSQSLEASDTNIYLSWFQQRPGQSPRRLIYKISNRDSGVPDRFSGSGSGTHFTLRISRVEADDVAVYYCMQGTHWPPAFGQGTKVDIK',
]

result = sss(seqs)
print(result.embeddings.shape)  # (1, n_residues, 128)
print(result.mask.shape)        # (1, n_residues)
```

## Citation

```bibtex
@article{ellmen_modelling_2026,
    title = {Modelling antibody structures at the speed of language},
    author = {Ellmen, Isaac and Errington, David and Raybould, Matthew I. J. and Deane, Charlotte M.},
    year = {2026},
    url = {https://www.biorxiv.org/content/10.64898/2026.06.03.729879v1},
}

@article{ellmen_correcting_2026,
    title = {Correcting the Dropout-LayerNorm Expectation Gap Improves Protein Structure Models},
    author = {Ellmen, Isaac and Errington, David and Raybould, Matthew I. J. and Deane, Charlotte M.},
    year = {2026},
    url = {https://arxiv.org/abs/2609.32062},
}
```
