import argparse
import sys
from torch.utils.data import Dataset, DataLoader

from util.util import enumerate_with_estimate, prhist
from .dsets import get_candidates_info, get_Ct
from util.logconf import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class LunaScreenCtDataset(Dataset):
    def __init__(self):
        self.series_list = sorted(set(candidate_info_tup.series_uid for candidate_info_tup in get_candidates_info()))

    def __len__(self):
        return len(self.series_list)

    def __getitem__(self, index):
        series_uid = self.series_list[index]
        ct = get_Ct(series_uid)
        mid_index = ct.hu_a.shape[0] // 2

        _, _, dense_mask, denoise_mask, _, _, _ = ct.build2dLungMask(mid_index)

        return series_uid, float(dense_mask.sum() / denoise_mask.sum())


class LunaScreenCtApp:
    @classmethod
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size',
            help='Batch size to use for training',
            default=4,
            type=int,
        )
        parser.add_argument('--num-workers',
            help='Number of worker processes for background data loading',
            default=8,
            type=int,
        )

        self.cli_args = parser.parse_args(sys_argv)

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        self.prep = DataLoader(
            LunaScreenCtDataset(),
            batch_size=self.cli_args.batch_size,
            num_workers=self.cli_args.num_workers,
        )

        series2ratio_dict = {}

        batch_iteration = enumerate_with_estimate(
            self.prep,
            "Screening CTs",
            start_index=self.prep.num_workers,
        )
        for batch_index, batch_tup in batch_iteration:
            series_list, ratio_list = batch_tup
            for series_uid, ratio_float in zip(series_list, ratio_list):
                series2ratio_dict[series_uid] = ratio_float

        prhist(list(series2ratio_dict.values()))




if __name__ == '__main__':
    LunaScreenCtApp().main()
