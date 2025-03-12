from mydatasets.utils.continual_dataset import ContinualDataset
from argparse import Namespace
from mydatasets.seq_cifar100 import SequentialCIFAR100
from .seq_imagenet_r import SequentialImageNetR
from .seq_cifar10 import SequentialCIFAR10
from .seq_tinyimagenet import SequentialTinyImagenet
from .seq_miniimagenet import SequentialMiniImagenet



NAMES = {
    SequentialCIFAR100.NAME: SequentialCIFAR100,
    SequentialCIFAR10.NAME: SequentialCIFAR10,
    SequentialImageNetR.NAME: SequentialImageNetR,
    SequentialTinyImagenet.NAME:SequentialTinyImagenet,
    SequentialMiniImagenet.NAME:SequentialMiniImagenet
}


def get_dataset(args: Namespace) -> ContinualDataset:
    """
    Creates and returns a continual dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    assert args.dataset in NAMES.keys()
    return NAMES[args.dataset](args)


def get_gcl_dataset(args: Namespace):
    """
    Creates and returns a GCL dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    return NAMES[args.dataset](args)
