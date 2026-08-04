from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from distance_not_damage.config import DataConfig

PARITY_TASK_INDICATOR = 1.0
FASHION_TASK_INDICATOR = -1.0
CLASS_COUNT = 10


class VisionDataset(Protocol):
    targets: Tensor | list[int]

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> tuple[Tensor, int]: ...


class TaggedDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Attach a task indicator and optionally replace labels deterministically."""

    def __init__(
        self,
        dataset: Dataset[tuple[Tensor, int]],
        *,
        task_indicator: float,
        replacement_targets: Tensor | None = None,
    ) -> None:
        self.dataset = dataset
        self.task_indicator = task_indicator
        self.replacement_targets = replacement_targets
        if replacement_targets is not None and len(replacement_targets) != len(dataset):
            raise ValueError("replacement_targets must match dataset length")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        image, target = self.dataset[index]
        if self.replacement_targets is not None:
            target = int(self.replacement_targets[index])
        return (
            image,
            torch.tensor(target, dtype=torch.long),
            torch.tensor(self.task_indicator, dtype=torch.float32),
        )


@dataclass(frozen=True)
class DataBundle:
    pretrain_loader: DataLoader[tuple[Tensor, Tensor, Tensor]]
    parity_train_loader: DataLoader[tuple[Tensor, Tensor, Tensor]]
    parity_eval_loader: DataLoader[tuple[Tensor, Tensor, Tensor]]
    fashion_probe_train_loader: DataLoader[tuple[Tensor, Tensor, Tensor]]
    fashion_eval_loader: DataLoader[tuple[Tensor, Tensor, Tensor]]


def prepare_data(config: DataConfig, seed: int) -> DataBundle:
    """Download once, then construct all deterministic loaders.

    The sweep runner is sequential, so dataset download and result writing each
    have a single owner. Worker processes only read already-downloaded files.
    """

    mnist_train, mnist_test, fashion_train, fashion_test = _load_datasets(config.root)
    mnist_subset = _stratified_subset(
        mnist_train,
        total_examples=config.pretrain_examples_per_task,
        seed=seed,
    )
    fashion_subset = _stratified_subset(
        fashion_train,
        total_examples=config.pretrain_examples_per_task,
        seed=seed + 1,
    )

    parity_pretrain_targets = _sample_uniform_valid_parity_targets(mnist_subset, seed=seed + 2)
    parity_pretrain = TaggedDataset(
        mnist_subset,
        task_indicator=PARITY_TASK_INDICATOR,
        replacement_targets=parity_pretrain_targets,
    )
    fashion_pretrain = TaggedDataset(
        fashion_subset,
        task_indicator=FASHION_TASK_INDICATOR,
    )
    parity_fine_tune: Dataset[tuple[Tensor, int]] = mnist_train
    if config.fine_tune_examples is not None:
        parity_fine_tune = _stratified_subset(
            mnist_train,
            total_examples=config.fine_tune_examples,
            seed=seed + 3,
        )

    loader_generator = torch.Generator().manual_seed(seed)
    loader_arguments = {
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": config.num_workers > 0,
    }
    return DataBundle(
        pretrain_loader=DataLoader(
            ConcatDataset((parity_pretrain, fashion_pretrain)),
            batch_size=config.train_batch_size,
            shuffle=True,
            generator=loader_generator,
            **loader_arguments,
        ),
        parity_train_loader=DataLoader(
            TaggedDataset(parity_fine_tune, task_indicator=PARITY_TASK_INDICATOR),
            batch_size=config.train_batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed + 3),
            **loader_arguments,
        ),
        parity_eval_loader=DataLoader(
            TaggedDataset(mnist_test, task_indicator=PARITY_TASK_INDICATOR),
            batch_size=config.eval_batch_size,
            shuffle=False,
            **loader_arguments,
        ),
        fashion_probe_train_loader=DataLoader(
            TaggedDataset(fashion_subset, task_indicator=FASHION_TASK_INDICATOR),
            batch_size=config.eval_batch_size,
            shuffle=False,
            **loader_arguments,
        ),
        fashion_eval_loader=DataLoader(
            TaggedDataset(fashion_test, task_indicator=FASHION_TASK_INDICATOR),
            batch_size=config.eval_batch_size,
            shuffle=False,
            **loader_arguments,
        ),
    )


def _load_datasets(
    root: Path,
) -> tuple[datasets.MNIST, datasets.MNIST, datasets.FashionMNIST, datasets.FashionMNIST]:
    transform = transforms.ToTensor()
    root.mkdir(parents=True, exist_ok=True)
    mnist_train = datasets.MNIST(root=root, train=True, transform=transform, download=True)
    mnist_test = datasets.MNIST(root=root, train=False, transform=transform, download=True)
    fashion_train = datasets.FashionMNIST(root=root, train=True, transform=transform, download=True)
    fashion_test = datasets.FashionMNIST(root=root, train=False, transform=transform, download=True)
    return mnist_train, mnist_test, fashion_train, fashion_test


def _stratified_subset(dataset: VisionDataset, *, total_examples: int, seed: int) -> Subset:
    if total_examples % CLASS_COUNT != 0:
        raise ValueError(f"total_examples must be divisible by {CLASS_COUNT}")

    examples_per_class = total_examples // CLASS_COUNT
    targets = torch.as_tensor(dataset.targets)
    generator = torch.Generator().manual_seed(seed)
    selected: list[int] = []
    for class_index in range(CLASS_COUNT):
        candidates = torch.where(targets == class_index)[0]
        if len(candidates) < examples_per_class:
            raise ValueError(f"Class {class_index} has too few examples")
        order = torch.randperm(len(candidates), generator=generator)
        selected.extend(candidates[order[:examples_per_class]].tolist())
    return Subset(dataset, selected)


def _sample_uniform_valid_parity_targets(dataset: Subset, *, seed: int) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    targets: list[int] = []
    for _, digit in dataset:
        offset = int(torch.randint(0, CLASS_COUNT // 2, (), generator=generator))
        targets.append(2 * offset + (int(digit) % 2))
    return torch.tensor(targets, dtype=torch.long)
