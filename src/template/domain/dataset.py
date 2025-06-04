from typing import Sequence, Type

import torch
from torch.utils.data import Dataset

from template.domain.tokenizer import Tokenizer, CharTokenizer


class VilleDataset(Dataset):
    """A simple dataset for character-level tokenization."""

    sentences: Sequence[str]
    tokenizer: Tokenizer

    def __init__(self, sentences: Sequence[str], type_tokenizer: Type[Tokenizer] = CharTokenizer):
        self.sentences = sentences
        self.tokenizer = type_tokenizer.from_sentences(sentences)

        self.num_sequences = len(sentences)
        self.max_len = max((len(ville) for ville in sentences)) + 2  # <SOS> and <EOS>

        self.sos_index = self.tokenizer.token_to_index[self.tokenizer.sos_token]
        self.eos_index = self.tokenizer.token_to_index[self.tokenizer.eos_token]
        self.pad_index = self.tokenizer.token_to_index[self.tokenizer.pad_token]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index: int) -> (torch.Tensor, torch.Tensor):
        """Get a single item from the dataset."""
        sentence = self.sentences[index]
        tokens = [self.sos_index] + self.tokenizer.encode(sentence) + [self.eos_index]

        # Padding pour que toutes les séquences aient la même longueur
        padding_length = self.max_len - len(tokens)
        tokens += [self.pad_index] * padding_length

        input_tensor = torch.tensor(tokens[:-1], dtype=torch.long)  # [L-1]
        target_tensor = torch.tensor(tokens[1:], dtype=torch.long)  # [L-1]

        return input_tensor, target_tensor
