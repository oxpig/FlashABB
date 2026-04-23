import torch

_VOCAB = {
    "<": 0,   # start
    "-": 21,  # pad
    ">": 22,  # end
    "*": 23,  # mask
    "X": 24,  # unknown
    "|": 25,  # heavy/light separator
    "M": 1, "R": 2, "H": 3, "K": 4, "D": 5, "E": 6, "S": 7, "T": 8,
    "N": 9, "Q": 10, "C": 11, "G": 12, "P": 13, "A": 14, "V": 15,
    "I": 16, "F": 17, "Y": 18, "W": 19, "L": 20,
}


class ABtokenizer:
    """Antibody sequence tokenizer (vocabulary from AbLang2)."""

    def __init__(self):
        self.aa_to_token = _VOCAB
        self.token_to_aa = {v: k for k, v in self.aa_to_token.items()}
        self.pad_token = self.aa_to_token['-']
        self.start_token = self.aa_to_token['<']
        self.end_token = self.aa_to_token['>']
        self.sep_token = self.aa_to_token['|']
        self.mask_token = self.aa_to_token['*']
        self.unknown_token = self.aa_to_token['X']

    def __call__(self, sequence_list, mode='encode', pad=False, w_extra_tkns=True, device='cpu'):
        if w_extra_tkns:
            sequence_list = [sequence_list] if isinstance(sequence_list[0], str) else sequence_list
        else:
            sequence_list = [sequence_list] if isinstance(sequence_list, str) else sequence_list

        if mode == 'encode':
            data = [self.encode(seq, w_extra_tkns=w_extra_tkns, device=device) for seq in sequence_list]
            if pad:
                return torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=self.pad_token)
            return data
        elif mode == 'decode':
            return [self.decode(seq) for seq in sequence_list]
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'encode' or 'decode'.")

    def encode(self, sequence, w_extra_tkns=True, device='cpu'):
        if w_extra_tkns:
            heavy, light = sequence
            sequence = f"<{heavy}>|<{light}>".replace("<>", "")
        return torch.tensor([self.aa_to_token[r] for r in sequence], dtype=torch.long, device=device)

    def decode(self, tokenized_seq):
        if torch.is_tensor(tokenized_seq):
            tokenized_seq = tokenized_seq.cpu().numpy()
        return ''.join(self.token_to_aa[t] for t in tokenized_seq)
