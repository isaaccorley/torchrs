import inspect
import pytest
from torch.utils.data import Dataset

from torchrs import datasets


skip = ["RSVQAxBEN", "S2MTCP"]


@pytest.mark.parametrize("dataset", reversed(datasets.__all__))
def test_datamodules(dataset: Dataset):

    if dataset in skip:
        return

    dataclass = getattr(datasets, dataset)

    if "split" in inspect.getfullargspec(dataclass).args:
        for split in dataclass.splits:
            ds = dataclass(split=split)
            length = len(ds)
            sample = ds[0]
            sample = ds[-1]
    else:
        ds = dataclass()
        length = len(ds)
        sample = ds[0]
        sample = ds[-1]
