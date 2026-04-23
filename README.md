# FlashABB: modelling antibody structures at the speed of language

![Inference speed comparison](figures/speedup_multiplier.png)

Installation:

PyPi coming soon

```bash
git clone git@github.com:Ellmen/flash-abb.git
cd flash-abb
pip install .
```

Usage:

The following is also in `example.py` and can be used to create the structures in `sample_preds`

```python
from flash_abb import pretrained
import torch

flabb = pretrained(device='cuda')

seq1 = [
    'EVQLLESGGEVKKPGASVKVSCRASGYTFRNYGLTWVRQAPGQGLEWMGWISAYNGNTNYAQKFQGRVTLTTDTSTSTAYMELRSLRSDDTAVYFCARDVPGHGAAFMDVWGTGTTVTVSS', # Heavy chain
    'DIQLTQSPLSLPVTLGQPASISCRSSQSLEASDTNIYLSWFQQRPGQSPRRLIYKISNRDSGVPDRFSGSGSGTHFTLRISRVEADDVAVYYCMQGTHWPPAFGQGTKVDIK' # Light chain
]
seq2 = [
    'EVQLLESGGEVKKPGASVKVSCRASGYTFRNYGLTWVRQAPGQGLEWMGWISAYNGNTNYAQKFQGRVTLTTDTSTSTAYMELRSLRSDDTAVYFCARDVPGHGAAFMDVWGTGTTVTVS', # Heavy chain
    'DIQLTQSPLSLPVTLGQPASISCRSSQSLEASDTNIYLSWFQQRPGQSPRRLIYKISNRDSGVPDRFSGSGSGTHFTLRISRVEADDVAVYYCMQGTHWPPAFGQGTKVDIK' # Light chain
]
seqs = [seq1, seq2]
seqs = [f'{seq[0]}|{seq[1]}' for seq in seqs]
names = ['test1', 'test2']

with torch.no_grad():
    result = flabb(seqs)

# Coords returns a tensor of the predicted coordinates
print(result.coords.shape)

# Save predictions as PDB files
result.to_pdbs(names, pdb_dir='sample_preds')
```
