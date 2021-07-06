import os
from pathlib import Path
from typing import Optional, Tuple

from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
import musdb

from src.datamodules.datasets.MusdbWrapper import MusdbWrapperDataset, MusdbTrainDataset, MusdbValidationDataset


class Musdb18hqDataModule(LightningDataModule):
    """
    LightningDataModule for Musdb18-HQ dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
            self,
            data_dir: str,
            augmentation: bool,
            external_datasets,
            target_name: str,
            n_fft: int,
            hop_length: int,
            dim_c: int,
            dim_f: int,
            dim_t: int,
            sampling_rate: int,
            batch_size: int,
            num_workers: int,
            pin_memory: bool,
            **kwargs,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.augmentation = augmentation
        self.external_datasets = external_datasets
        self.target_name = target_name

        # audio-related
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.dim_c = dim_c
        self.dim_f = dim_f
        self.dim_t = dim_t
        self.sampling_rate = sampling_rate

        # derived
        self.n_bins = n_fft // 2 + 1
        self.sampling_size = hop_length * (dim_t - 1)
        self.trim = n_fft // 2  # trim each generated sub-signal (noise due to conv zero-padding)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        if not os.path.exists(self.data_dir + "/valid"):
            from shutil import move
            root = Path(self.data_dir)
            train_root = root.joinpath('train')
            valid_root = root.joinpath('valid')
            os.mkdir(valid_root)

            for track in kwargs['validation_set']:
                move(train_root.joinpath(track), valid_root.joinpath(track))

            print()
        else:
            valid_files = os.listdir(Path(self.data_dir).joinpath('valid'))
            assert set(valid_files) == set(kwargs['validation_set'])

    @property
    def num_classes(self) -> int:
        return 10

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        self.data_train = MusdbTrainDataset(self.data_dir,
                                            self.augmentation,
                                            self.external_datasets,
                                            self.target_name,
                                            self.sampling_size)

        self.data_val = MusdbValidationDataset(self.batch_size,
                                               self.data_dir,
                                               self.target_name,
                                               self.sampling_size,
                                               self.n_fft)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
