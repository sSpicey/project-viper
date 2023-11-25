import argparse
import glob
import os
import sys

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader

from .dsets import LunaDataset, get_Ct, get_candidate_info, get_candidates_info, CandidateInfo
from segmentation.model import UNetWrapper
from util.util import enumerate_with_estimate
from segmentation.dsets import Luna2dSegmentationDataset

import numpy as np
import scipy.ndimage.measurements as measurements
import scipy.ndimage.morphology as morphology

import classification.model
from util.logconf import logging
from util.util import Irc2xyz

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
logging.getLogger("segmentation.dsets").setLevel(logging.WARNING)
logging.getLogger("classification.dsets").setLevel(logging.WARNING)

def match_and_score(detections, truth, threshold=0.5, threshold_mal=0.5):
    true_nodules = [c for c in truth if c.is_nodule]
    truth_diams = np.array([c.diameter_mm for c in true_nodules])
    truth_xyz = np.array([c.center_xyz for c in true_nodules])

    detected_xyz = np.array([n[2] for n in detections])
    detected_classes = np.array([1 if d[0] < threshold
                                 else (2 if d[1] < threshold
                                       else 3) for d in detections])

    confusion = np.zeros((3, 4), dtype=np.int)
    if len(detected_xyz) == 0:
        for tn in true_nodules:
            confusion[2 if tn.is_malignant else 1, 0] += 1
    elif len(truth_xyz) == 0:
        for dc in detected_classes:
            confusion[0, dc] += 1
    else:
        normalized_dists = np.linalg.norm(truth_xyz[:, None] - detected_xyz[None], ord=2, axis=-1) / truth_diams[:, None]
        matches = (normalized_dists < 0.7)
        unmatched_detections = np.ones(len(detections), dtype=np.bool)
        matched_true_nodules = np.zeros(len(true_nodules), dtype=np.int)
        for i_tn, i_detection in zip(*matches.nonzero()):
            matched_true_nodules[i_tn] = max(matched_true_nodules[i_tn], detected_classes[i_detection])
            unmatched_detections[i_detection] = False

        for ud, dc in zip(unmatched_detections, detected_classes):
            if ud:
                confusion[0, dc] += 1
        for tn, dc in zip(true_nodules, matched_true_nodules):
            confusion[2 if tn.is_malignant else 1, dc] += 1
    return confusion

class NoduleAnalysisApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            log.debug(sys.argv)
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size',
            help='Batch size to use for training',
            default=4,
            type=int,
        )
        parser.add_argument('--num-workers',
            help='Number of worker processes for background data loading',
            default=4,
            type=int,
        )

        parser.add_argument('--run-validation',
            help='Run over validation rather than a single CT.',
            action='store_true',
            default=False,
        )
        parser.add_argument('--include-train',
            help="Include data that was in the training set. (default: validation data only)",
            action='store_true',
            default=False,
        )

        parser.add_argument('--segmentation-path',
            help="Path to the saved segmentation model",
            nargs='?',
            default='data/part2/models/seg_2020-01-26_19.45.12_w4d3c1-bal_1_nodupe-label_pos-d1_fn8-adam.best.state',
        )

        parser.add_argument('--cls-model',
            help="What to model class name to use for the classifier.",
            action='store',
            default='LunaModel',
        )
        parser.add_argument('--classification-path',
            help="Path to the saved classification model",
            nargs='?',
            default='data/part2/models/cls_2020-02-06_14.16.55_final-nodule-nonnodule.best.state',
        )

        parser.add_argument('--malignancy-model',
            help="What to model class name to use for the malignancy classifier.",
            action='store',
            default='LunaModel',
        )
        parser.add_argument('--malignancy-path',
            help="Path to the saved malignancy classification model",
            nargs='?',
            default=None,
        )

        parser.add_argument('--tb-prefix',
            default='classification',
            help="Data prefix to use for Tensorboard run. Defaults to chapter.",
        )

        parser.add_argument('series_uid',
            nargs='?',
            default=None,
            help="Series UID to use.",
        )

        self.cli_args = parser.parse_args(sys_argv)

        if not (bool(self.cli_args.series_uid) ^ self.cli_args.run_validation):
            raise Exception("One and only one of series_uid and --run-validation should be given")


        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        if not self.cli_args.segmentation_path:
            self.cli_args.segmentation_path = self.init_modelPath('seg')

        if not self.cli_args.classification_path:
            self.cli_args.classification_path = self.init_modelPath('cls')

        self.seg_model, self.cls_model, self.malignancy_model = self.init_models()

    def init_modelPath(self, type_str):
        local_path = os.path.join(
            'data-unversioned',
            'part2',
            'models',
            'segmentation',
            type_str + '_{}_{}.{}.state'.format('*', '*', 'best'),
        )

        file_list = glob.glob(local_path)
        if not file_list:
            pretrained_path = os.path.join(
                'data',
                'part2',
                'models',
                type_str + '_{}_{}.{}.state'.format('*', '*', '*'),
            )
            file_list = glob.glob(pretrained_path)
        else:
            pretrained_path = None

        file_list.sort()

        try:
            return file_list[-1]
        except IndexError:
            log.debug([local_path, pretrained_path, file_list])
            raise

    def init_models(self):
        log.debug(self.cli_args.segmentation_path)
        seg_dict = torch.load(self.cli_args.segmentation_path, map_location=torch.device('cpu'))

        seg_model = UNetWrapper(
            in_channels=7,
            n_classes=1,
            depth=3,
            wf=4,
            padding=True,
            batch_norm=True,
            up_mode='upconv',
        )

        seg_model.load_state_dict(seg_dict['model_state'])
        seg_model.eval()

        log.debug(self.cli_args.classification_path)
        cls_dict = torch.load(self.cli_args.classification_path, map_location=torch.device('cpu'))

        model_cls = getattr(classification.model, self.cli_args.cls_model)
        cls_model = model_cls()
        cls_model.load_state_dict(cls_dict['model_state'])
        cls_model.eval()

        if self.use_cuda:
            if torch.cuda.device_count() > 1:
                seg_model = nn.DataParallel(seg_model)
                cls_model = nn.DataParallel(cls_model)

            seg_model.to(self.device)
            cls_model.to(self.device)

        if self.cli_args.malignancy_path:
            model_cls = getattr(classification.model, self.cli_args.malignancy_model)
            malignancy_model = model_cls()
            malignancy_dict = torch.load(self.cli_args.malignancy_path, map_location=torch.device('cpu'))
            malignancy_model.load_state_dict(malignancy_dict['model_state'])
            malignancy_model.eval()
            if self.use_cuda:
                malignancy_model.to(self.device)
        else:
            malignancy_model = None
        return seg_model, cls_model, malignancy_model


    def initSegmentationDl(self, series_uid):
        seg_ds = Luna2dSegmentationDataset(
                contextSlices_count=3,
                series_uid=series_uid,
                fullCt_bool=True,
            )
        seg_dl = DataLoader(
            seg_ds,
            batch_size=self.cli_args.batch_size * (torch.cuda.device_count() if self.use_cuda else 1),
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return seg_dl

    def initClassificationDl(self, candidates_info):
        cls_ds = LunaDataset(
                sortby_str='series_uid',
                candidates_info=candidates_info,
            )
        cls_dl = DataLoader(
            cls_ds,
            batch_size=self.cli_args.batch_size * (torch.cuda.device_count() if self.use_cuda else 1),
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return cls_dl


    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        val_ds = LunaDataset(
            val_stride=10,
            is_value_set=True,
        )
        val_set = set(
            candidate_info_tup.series_uid
            for candidate_info_tup in val_ds.candidates_info
        )
        positive_set = set(
            candidate_info_tup.series_uid
            for candidate_info_tup in get_candidates_info()
            if candidate_info_tup.is_nodule
        )

        if self.cli_args.series_uid:
            series_set = set(self.cli_args.series_uid.split(','))
        else:
            series_set = set(
                candidate_info_tup.series_uid
                for candidate_info_tup in get_candidates_info()
            )

        if self.cli_args.include_train:
            train_list = sorted(series_set - val_set)
        else:
            train_list = []
        val_list = sorted(series_set & val_set)


        candidateInfo_dict = get_candidate_info()
        series_iter = enumerate_with_estimate(
            val_list + train_list,
            "Series",
        )
        all_confusion = np.zeros((3, 4), dtype=np.int)
        for _, series_uid in series_iter:
            ct = get_Ct(series_uid)
            mask_a = self.segmentCt(ct, series_uid)

            candidates_info = self.group_segmentation_output(
                series_uid, ct, mask_a)
            classifications_list = self.classifyCandidates(
                ct, candidates_info)

            if not self.cli_args.run_validation:
                print(f"found nodule candidates in {series_uid}:")
                for prob, prob_mal, center_xyz, center_Irc in classifications_list:
                    if prob > 0.5:
                        s = f"nodule prob {prob:.3f}, "
                        if self.malignancy_model:
                            s += f"malignancy prob {prob_mal:.3f}, "
                        s += f"center xyz {center_xyz}"
                        print(s)

            if series_uid in candidateInfo_dict:
                one_confusion = match_and_score(
                    classifications_list, candidateInfo_dict[series_uid]
                )
                all_confusion += one_confusion
                


    def classifyCandidates(self, ct, candidates_info):
        cls_dl = self.initClassificationDl(candidates_info)
        classifications_list = []
        for batch_index, batch_tup in enumerate(cls_dl):
            input_t, _, _, series_list, center_list = batch_tup

            input_g = input_t.to(self.device)
            with torch.no_grad():
                _, probability_nodule_g = self.cls_model(input_g)
                if self.malignancy_model is not None:
                    _, probability_mal_g = self.malignancy_model(input_g)
                else:
                    probability_mal_g = torch.zeros_like(probability_nodule_g)

            zip_iter = zip(center_list,
                probability_nodule_g[:,1].tolist(),
                probability_mal_g[:,1].tolist())
            for center_Irc, prob_nodule, prob_mal in zip_iter:
                center_xyz = Irc2xyz(center_Irc,
                    direction_a=ct.direction_a,
                    origin_xyz=ct.origin_xyz,
                    vx_size_xyz=ct.vx_size_xyz,
                )
                cls_tup = (prob_nodule, prob_mal, center_xyz, center_Irc)
                classifications_list.append(cls_tup)
        return classifications_list

    def segmentCt(self, ct, series_uid):
        with torch.no_grad():
            output_a = np.zeros_like(ct.hu_a, dtype=np.float32)
            seg_dl = self.initSegmentationDl(series_uid)  #  <3>
            for input_t, _, _, slice_index_list in seg_dl:

                input_g = input_t.to(self.device)
                prediction_g = self.seg_model(input_g)

                for i, slice_index in enumerate(slice_index_list):
                    output_a[slice_index] = prediction_g[i].cpu().numpy()

            mask_a = output_a > 0.5
            mask_a = morphology.binary_erosion(mask_a, iterations=1)

        return mask_a

    def group_segmentation_output(self, series_uid,  ct, clean_a):
        candidate_label_a, candidate_count = measurements.label(clean_a)
        centerIrc_list = measurements.center_of_mass(
            ct.hu_a.clip(-1000, 1000) + 1001,
            labels=candidate_label_a,
            index=np.arange(1, candidate_count+1),
        )

        candidates_info = []
        for i, center_Irc in enumerate(centerIrc_list):
            center_xyz = Irc2xyz(
                center_Irc,
                ct.origin_xyz,
                ct.vx_size_xyz,
                ct.direction_a,
            )
            assert np.all(np.isfinite(center_Irc)), repr(['Irc', center_Irc, i, candidate_count])
            assert np.all(np.isfinite(center_xyz)), repr(['xyz', center_xyz])
            candidate_info_tup = \
                CandidateInfo(False, False, False, 0.0, series_uid, center_xyz)
            candidates_info.append(candidate_info_tup)

        return candidates_info

if __name__ == '__main__':
    NoduleAnalysisApp().main()
