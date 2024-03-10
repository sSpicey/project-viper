import argparse
import sys
from util.util import enumerate_with_estimate
from .datasets import PrepcacheLunaDataset
from util.logconf import logging
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class LunaCachePrepper:
    def __init__(self, sys_argv=None):
        """
        Initializes the LunaPrepCacheApp with command line arguments.

        Args:
            sys_argv: Command line arguments passed to the application.
        """
        self.cli_args = self.parse_arguments(sys_argv)

    def parse_arguments(self, sys_argv):
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
                            help='Batch size',
                            default=1024,
                            type=int)
        parser.add_argument('--num-workers',
                            help='Number of workers',
                            default=8,
                            type=int)

        return parser.parse_args(sys_argv)

    def main(self):
        """
        The main entry point for the application.
        """
        log.info("Running {}, {}...".format(type(self).__name__, self.cli_args))

        self.prep_dl = DataLoader(PrepcacheLunaDataset(), batch_size=self.cli_args.batch_size, num_workers=self.cli_args.num_workers)

        batch_iteration = enumerate_with_estimate(self.prep_dl, "Stuffing cache with data from dset...", start_index=self.prep_dl.num_workers)
        for _, _ in batch_iteration:
            pass  # The actual processing is done in the PrepcacheLunaDataset class


if __name__ == '__main__':
    LunaCachePrepper().main()
