import argparse
import sys
from util.util import enumerate_with_estimate
from .dsets import LunaDataset
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
            help='Batch size to use for training',
            default=1024,
            type=int,
        )
        parser.add_argument('--num-workers',
            help='Number of worker processes for background data loading',
            default=8,
            type=int,
        )
        self.cli_args = parser.parse_args(sys_argv)

    def main(self):
        self.prep_dl = DataLoader(
            LunaDataset(
                sortby_str='series_uid',
            ),
            batch_size=self.cli_args.batch_size,
            num_workers=self.cli_args.num_workers,
        )

        batch_iteration = enumerate_with_estimate(
            self.prep_dl,
            "Stuffing cache",
            start_index=self.prep_dl.num_workers,
        )
        for batch_index, batch_tup in batch_iteration:
            pass


if __name__ == '__main__':
    LunaPrepCacheApp().main()
