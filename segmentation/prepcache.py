import argparse
import sys
from util.util import enumerate_with_estimate
from .dsets import PrepcacheLunaDataset
from util.logconf import logging
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class LunaPrepCacheApp:
    @classmethod
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size',
            help='Batch size ',
            default=1024,
            type=int,
        )
        parser.add_argument('--num-workers',
            help='Number of workers',
            default=8,
            type=int,
        )

        self.cli_args = parser.parse_args(sys_argv)

    def main(self):
        log.info("Running {}, {}...".format(type(self).__name__, self.cli_args))

        self.prep_dl = DataLoader(
            PrepcacheLunaDataset(
            ),
            batch_size=self.cli_args.batch_size,
            num_workers=self.cli_args.num_workers,
        )

        batch_iteration = enumerate_with_estimate(
            self.prep_dl,
            "Stuffing cache with data from dset...",
            start_index=self.prep_dl.num_workers,
        )
        for _, _ in batch_iteration:
            pass


if __name__ == '__main__':
    LunaPrepCacheApp().main()
