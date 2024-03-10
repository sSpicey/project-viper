import os
import sys
import glob
import torch.optim
import torch.nn as nn
import argparse
import hashlib

import numpy as np
from segmentation.datasets import Luna2dSegmentationDataset, get_Ct, get_candidates_info, get_candidate_info, CandidateInfo
from classification.model import LunaModel
from util.logconf import logging
from torch.utils.data import DataLoader
from classification.datasets import LunaDataset

import torch
import scipy.ndimage.measurements as measure

from util.util import irc_to_xyz
from segmentation.model import WrapperUNet

from util.util import enumerate_with_estimate

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class FalsePositiveRateCheck:
    """
    A class to check the false positive rate of lung nodule detection using segmentation and classification models.
    """
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            log.debug(sys.argv)
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        self._add_parser_args(parser)
        self.cli_args = parser.parse_args(sys_argv)
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        if not self.cli_args.segmentation_path:
            self.cli_args.segmentation_path = self.initialize_model_path('seg')

        if not self.cli_args.classification_path:
            self.cli_args.classification_path = self.initialize_model_path('cls')

        self.segmentation_model, self.classification_model = self.initialize_models()

    @staticmethod
    def _add_parser_args(parser) -> None:
        parser.add_argument('--batch-size',
                            help='Training batch size',
                            default=8,
                            type=int,
                            )
        parser.add_argument('--num-workers',
                            help='Number of worker processes',
                            default=8,
                            type=int,
                            )
        parser.add_argument('--series-uid',
                            help='Choose series-uid to infer on.',
                            default=None,
                            type=str,
                            )
        parser.add_argument('--include-train',
                            help="Include data from training set.",
                            action='store_true',
                            default=False,
                            )
        parser.add_argument('--segmentation-path',
                            help="Path to the segmentation model",
                            nargs='?',
                            default=None,
                            )
        parser.add_argument('--classification-path',
                            help="Path to the classification model",
                            nargs='?',
                            default=None,
                            )

    @staticmethod
    def initialize_model_path(model_type):
        """
        Initializes the path to the model based on the provided type string.

        Args:
            model_type (str): The type of the model ('seg' for segmentation or 'cls' for classification).

        Returns:
            str: The path to the latest model of the specified type.
        """
        path = os.path.join(
            'data',
            'part2',
            'models',
            model_type + '_{}_{}.{}.state'.format('*', '*', '*'),
        )
        files = glob.glob(path)
        files.sort()

        try:
            return files[-1]
        except IndexError as e:
            raise e

    def initialize_models(self):
        """
        Initializes the segmentation and classification models.

        Returns:
            tuple: A tuple containing the segmentation model and classification model.
        """
        with open(self.cli_args.segmentation_path, 'rb') as f:
            log.debug(self.cli_args.segmentation_path)
            log.debug(hashlib.sha1(f.read()).hexdigest())

        segmentation = torch.load(self.cli_args.segmentation_path)

        segmentation_model = WrapperUNet(
            in_channels=7,
            depth=3,
            wf=4,
            n_classes=1,
            batch_norm=True,
            up_mode='upconv',
            padding=True,
        )
        segmentation_model.load_state_dict(segmentation['model_state'])
        segmentation_model.eval()

        with open(self.cli_args.classification_path, 'rb') as f:
            log.debug(self.cli_args.classification_path)
            log.debug(hashlib.sha1(f.read()).hexdigest())

        classification = torch.load(self.cli_args.classification_path)
        classification_model = LunaModel()
        classification_model.load_state_dict(classification['model_state'])
        classification_model.eval()

        if self.use_cuda:
            if torch.cuda.device_count() > 1:
                classification_model = nn.DataParallel(classification_model)
                segmentation_model = nn.DataParallel(segmentation_model)

            classification_model = classification_model.to(self.device)
            segmentation_model = segmentation_model.to(self.device)

        self.convolutions = nn.ModuleList([
            self.convolute(radius).to(self.device) for radius in range(1, 8)
        ])

        return segmentation_model, classification_model

    def initialize_segmentation(self, series_uid):
        """
        Initializes the segmentation dataset and dataloader for a given series UID.

        Args:
            series_uid (str): The series UID for which to initialize the segmentation.

        Returns:
            DataLoader: The dataloader for the segmentation dataset.
        """
        segmentation_dataset = Luna2dSegmentationDataset(
            series_uid=series_uid,
            fullCt_bool=True,
            contextSlices_count=3,
        )
        segmentation_dataloader = DataLoader(
            segmentation_dataset,
            batch_size=self.cli_args.batch_size * (torch.cuda.device_count() if self.use_cuda else 1),
            num_workers=1,
            pin_memory=self.use_cuda,
        )

        return segmentation_dataloader

    def initialize_classification(self, candidate_info):
        """
        Initializes the classification dataset and dataloader for a given list of candidate information.

        Args:
            candidate_info (list): A list of CandidateInfo objects.

        Returns:
            DataLoader: The dataloader for the classification dataset.
        """
        classification_dataset = LunaDataset(
            sortby_str='series_uid',
            candidates_info=candidate_info,
        )
        classification_dataloader = DataLoader(
            classification_dataset,
            num_workers=1,
            pin_memory=self.use_cuda,
            batch_size=self.cli_args.batch_size * (torch.cuda.device_count() if self.use_cuda else 1),
        )

        return classification_dataloader

    def main(self):
        log.info("Running {}, {}".format(type(self).__name__, self.cli_args))

        values_dataset = LunaDataset(
            val_stride=10,
            is_value_set=True,
        )
        values = set(
            candidate.series_uid
            for candidate in values_dataset.candidates_info
        )

        if self.cli_args.series_uid:
            series = set(self.cli_args.series_uid.split(','))
        else:
            series = set(
                candidate.series_uid
                for candidate in get_candidates_info()
            )

        training_list = sorted(series - values) if self.cli_args.include_train else []
        validation_list = sorted(series & values)

        total_tp = total_tn = total_fp = total_fn = 0
        total_missed_positives = 0
        missed_positives_distances = []
        missed_positives_cits = []
        candidates_info = get_candidate_info()
        series_iteration = enumerate_with_estimate(
            validation_list + training_list,
            "Series",
        )
        for series_uid in series_iteration:
            ct, _, _, clean_g = self._segment_exam(series_uid)

            seg_candidates_info, _, _ = self.segmentation_output(
                series_uid,
                ct,
                clean_g,
            )
            if not seg_candidates_info:
                continue

            classification_dataloader = self.initialize_classification(seg_candidates_info)
            results = []
            for index, batch_tuple in enumerate(classification_dataloader):
                input_t, _, _, series_list, center_t = batch_tuple

                input_g = input_t.to(self.device)
                with torch.no_grad():
                    _, probability_g = self.classification_model(input_g)
                probability_t = probability_g.to('cpu')

                for i, _series_uid in enumerate(series_list):
                    assert series_uid == _series_uid, repr([index, i, series_uid, _series_uid, seg_candidates_info])
                    results.append((center_t[i], probability_t[i, 0].item()))

            tp = tn = fp = fn = 0
            missed_positives = 0
            ct = get_Ct(series_uid)
            candidate_info = candidates_info[series_uid]
            candidate_info = [cit for cit in candidate_info if cit.is_nodule]

            found_cit_list = [None] * len(results)

            for candidate in candidate_info:
                minimal_distance = (999, None)

                for result_index, (result_center_Irc_t, nodule_probability_t) in enumerate(results):
                    result_center_xyz = irc_to_xyz(result_center_Irc_t, ct.origin_xyz, ct.vx_size_xyz, ct.direction_a)
                    delta_xyz_t = torch.tensor(result_center_xyz) - torch.tensor(candidate.center_xyz)
                    distance_t = (delta_xyz_t ** 2).sum().sqrt()

                    minimal_distance = min(minimal_distance, (distance_t, result_index))

                distance_cutoff = max(10, candidate.diameter_mm / 2)
                if minimal_distance[0] < distance_cutoff:
                    _, result_index = minimal_distance
                    nodule_probability_t = results[result_index][1]

                    assert candidate.is_nodule

                    if nodule_probability_t > 0.5:
                        tp += 1
                    else:
                        fn += 1

                    found_cit_list[result_index] = candidate

                else:
                    missed_positives += 1
                    missed_positives_distances.append(float(minimal_distance[0]))
                    missed_positives_cits.append(candidate)

            total_tp += tp
            total_tn += tn
            total_fp += fp
            total_fn += fn
            total_missed_positives += missed_positives

        with open(self.cli_args.segmentation_path, 'rb') as f:
            log.info(self.cli_args.segmentation_path)
            log.info(hashlib.sha1(f.read()).hexdigest())
        with open(self.cli_args.classification_path, 'rb') as f:
            log.info(self.cli_args.classification_path)
            log.info(hashlib.sha1(f.read()).hexdigest())

        for cit, dist in zip(missed_positives_cits, missed_positives_distances):
            log.info("    Missed by {}: {}".format(dist, cit))

    def _segment_exam(self, series_uid):
        """
        Segments an exam for a given series UID.

        Args:
            series_uid (str): The series UID of the exam to segment.

        Returns:
            tuple: A tuple containing the CT object, the output tensor, the mask tensor, and the cleaned mask tensor.
        """
        with torch.no_grad():
            ct = get_Ct(series_uid)

            output = torch.zeros(ct.hu_a.shape, dtype=torch.float32, device=self.device)

            segmentation_dataloader = self.initialize_segmentation(series_uid)
            for batch in segmentation_dataloader:
                input_t, _, _, slice = batch

                input_g = input_t.to(self.device)
                prediction_g = self.segmentation_model(input_g)

                for i, slice in enumerate(slice):
                    output[slice] = prediction_g[i, 0]

            mask_g = output > 0.5
            clean_g = self.erode(mask_g.unsqueeze(0).unsqueeze(0), 1)[0][0]

        return ct, output, mask_g, clean_g

    def convolute(self, radius):
        """
        Creates a convolutional layer for erosion with a given radius.

        Args:
            radius (int): The radius of the convolutional layer.

        Returns:
            nn.Conv3d: The convolutional layer.
        """
        diameter = 1 + radius * 2

        a = torch.linspace(-1, 1, steps=diameter) ** 2
        b = (a[None] + a[:, None]) ** 0.5

        circle_weights = (b <= 1.0).to(torch.float32)

        conv = nn.Conv3d(1, 1, kernel_size=(1, diameter, diameter), padding=(0, radius, radius), bias=False)
        conv.weight.data.fill_(1)
        conv.weight.data *= circle_weights / circle_weights.sum()

        return conv

    def erode(self, input_mask, radius, threshold=1):
        """
        Erodes the input mask using a convolutional layer with the given radius and threshold.

        Args:
            input_mask (torch.Tensor): The input mask to erode.
            radius (int): The radius of the erosion.
            threshold (int, optional): The threshold for the erosion. Defaults to 1.

        Returns:
            torch.Tensor: The eroded mask.
        """
        conv = self.convolutions[radius - 1]
        input_float = input_mask.to(torch.float32)
        result = conv(input_float)

        return result >= threshold

    @staticmethod
    def segmentation_output(series_uid, ct, clean_g):
        """
        Processes the segmentation output for a given series UID.

        Args:
            series_uid (str): The series UID of the exam.
            ct (CT object): The CT object of the exam.
            clean_g (torch.Tensor): The cleaned mask tensor.

        Returns:
            tuple: A tuple containing the list of candidate information, the centerIrc list, and the candidate label array.
        """
        clean_a = clean_g.cpu().numpy()
        candidate_label_a, candidate_count = measure.label(clean_a)
        irc_centers = measure.center_of_mass(
            ct.hu_a.clip(-1000, 1000) + 1001,
            labels=candidate_label_a,
            index=list(range(1, candidate_count + 1)),
        )

        candidate_info = []
        for i, center in enumerate(irc_centers):
            assert np.isfinite(center).all(), repr([series_uid, i, candidate_count, (ct.hu_a[candidate_label_a == i + 1]).sum(), center])
            center_xyz = irc_to_xyz(
                center,
                ct.origin_xyz,
                ct.vx_size_xyz,
                ct.direction_a,
            )
            diameter = 0.0

            candidate = CandidateInfo(None, None, None, diameter, series_uid, center_xyz)
            candidate_info.append(candidate)

        return candidate_info, irc_centers, candidate_label_a


if __name__ == '__main__':
    FalsePositiveRateCheck().main()
