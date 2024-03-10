import argparse
from util.util import enumerate_with_estimate
from .datasets import LunaDataset
from util.logconf import logging
from torch.utils.data import DataLoader

import sys

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class LunaCachePrepper:
    def __init__(self, sys_argv=None):
        self.cli_args = self.parse_arguments(sys_argv)

    @staticmethod
    def parse_arguments(sys_argv):
        """
        Parses the command line arguments.

        Args:
            sys_argv: Command line arguments passed to the application.

        Returns:
            Parsed arguments as an argparse.Namespace object.
        """
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size',
                            help='Batch size to use for training',
                            default=1024,
                            type=int)
        parser.add_argument('--num-workers',
                            help='Number of worker processes for background data loading',
                            default=8,
                            type=int)
        return parser.parse_args(sys_argv)

    def main(self):
        self.prep_dl = self.init_dataloader()
        self.stuff_cache()

    def init_dataloader(self):
        """
        Initializes the DataLoader for the LunaDataset.

        Returns:
            A DataLoader object.
        """
        return DataLoader(
            LunaDataset(sortby_str='series_uid'),
            batch_size=self.cli_args.batch_size,
            num_workers=self.cli_args.num_workers,
        )

    def stuff_cache(self):
        """
        Iterates through the DataLoader to stuff the cache.
        """
        batch_iteration = enumerate_with_estimate(
            self.prep_dl,
            "Stuffing cache",
            start_index=self.prep_dl.num_workers,
        )
        for batch_index, batch_tup in batch_iteration:
            pass  # The actual processing is done in the LunaDataset class


if __name__ == '__main__':
    LunaCachePrepper().main()
