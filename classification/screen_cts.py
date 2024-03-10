import argparse
import sys
from util.util import enumerate_with_estimate, print_histogram
from .datasets import get_candidates_info, get_Ct
from util.logconf import logging
from torch.utils.data import Dataset, DataLoader



class LunaScreenCtDataset(Dataset):
    def __init__(self):
        """
        Initializes the LunaScreenCtDataset by extracting the series list from the candidates.
        """
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
    def __init__(self, sys_argv=None):
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
                            help='Batch size to use for training',
                            default=4,
                            type=int)
        parser.add_argument('--num-workers',
                            help='Number of worker processes for background data loading',
                            default=8,
                            type=int)
        return parser.parse_args(sys_argv)

    def main(self):
        self.prep = DataLoader(LunaScreenCtDataset(), batch_size=self.cli_args.batch_size, num_workers=self.cli_args.num_workers)

        series2ratio_dict = {}

        batch_iteration = enumerate_with_estimate(self.prep, "Screening CTs", start_index=self.prep.num_workers)
        for batch_index, batch_tup in batch_iteration:
            series_list, ratio_list = batch_tup
            for series_uid, ratio_float in zip(series_list, ratio_list):
                series2ratio_dict[series_uid] = ratio_float

        print_histogram(list(series2ratio_dict.values()))


if __name__ == '__main__':
    LunaScreenCtApp().main()
