from typing import Optional, List

from torch.utils.data import Dataset, random_split


def dataset_split(dataset: Dataset, val_pct: float, test_pct: Optional[float] = None) -> List[Dataset]:
    """ Split a torch Dataset into train/val/test sets """
    if test_pct is None:
        val_length = int(len(dataset) * val_pct)
        train_length = len(dataset) - val_length
        return random_split(dataset, [train_length, val_length])
    else:
        val_length = int(len(dataset) * val_pct)
        test_length = int(len(dataset) * test_pct)
        train_length = len(dataset) - (val_length + test_length)
        return random_split(dataset, [train_length, val_length, test_length])
