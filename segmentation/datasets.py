import csv
import functools
import glob
import os
import random
import torch
import torch.cuda
from torch.utils.data import Dataset
from collections import namedtuple
import SimpleITK as sitk
import numpy as np
from util.disk import get_cache
from util.util import Xyz, xyz_to_irc
from util.logconf import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

raw_cache = get_cache('segmentation')

Mask = namedtuple('Mask', 'raw_dense_mask, dense_mask, body_mask, air_mask, raw_candidate_mask, candidate_mask, lung_mask, neg_mask, pos_mask')

CandidateInfo = namedtuple('CandidateInfo', 'is_nodule, has_annotation, is_malignant, diameter_mm, series_uid, center_xyz')

@functools.lru_cache(1)
def get_candidates_info(required_on_disk=True):
    """
    Retrieves the candidate info from the annotations and candidates CSV files.

    Args:
        required_on_disk: Whether the candidate's series UID should be present on disk.

    Returns:
        A list of CandidateInfo namedtuples.
    """
    mhd_list = glob.glob('data-unversioned/part2/luna/subset*/*.mhd')
    present_on_disk = {os.path.split(p)[-1][:-4] for p in mhd_list}

    candidates_info = []
    with open('data/part2/luna/annotations_with_malignancy.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            annotation_center_xyz = tuple([float(x) for x in row[1:4]])
            annotation_diameter = float(row[4])
            is_malignant = {'False': False, 'True': True}[row[5]]

            candidates_info.append(
                CandidateInfo(
                    True,
                    True,
                    is_malignant,
                    annotation_diameter,
                    series_uid,
                    annotation_center_xyz,
                )
            )

    with open('data/part2/luna/candidates.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]

            if series_uid not in present_on_disk and required_on_disk:
                continue

            is_nodule = bool(int(row[4]))
            candidate_center_xyz = tuple([float(x) for x in row[1:4]])

            if not is_nodule:
                candidates_info.append(
                    CandidateInfo(
                        False,
                        False,
                        False,
                        0.0,
                        series_uid,
                        candidate_center_xyz,
                    )
                )

    candidates_info.sort(reverse=True)
    return candidates_info

@functools.lru_cache(1)
def get_candidate_info(required_on_disk=True):
    """
    Retrieves a dictionary of candidate info, keyed by series UID.

    Args:
        required_on_disk: Whether the candidate's series UID should be present on disk.

    Returns:
        A dictionary with series UID as keys and lists of CandidateInfo namedtuples as values.
    """
    candidates_info = get_candidates_info(required_on_disk)
    candidateInfo_dict = {}

    for candidate_info_tup in candidates_info:
        candidateInfo_dict.setdefault(candidate_info_tup.series_uid, []).append(candidate_info_tup)

    return candidateInfo_dict

class Ct:
    def __init__(self, series_uid):
        """
        Initializes a CT scan object.

        Args:
            series_uid: The series UID of the CT scan.
        """
        mhd_path = glob.glob('data-unversioned/part2/luna/subset*/{}.mhd'.format(series_uid))[0]

        ct_mhd = sitk.ReadImage(mhd_path)
        self.hu_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)

        self.series_uid = series_uid

        self.origin_xyz = Xyz(*ct_mhd.GetOrigin())
        self.vx_size_xyz = Xyz(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

        candidates_info = get_candidate_info()[self.series_uid]

        self.positives_info = [candidate_tup for candidate_tup in candidates_info if candidate_tup.is_nodule]
        self.positive_mask = self.build_annotation_mask(self.positives_info)
        self.positive_indexes = (self.positive_mask.sum(axis=(1, 2)).nonzero()[0].tolist())

    def build_annotation_mask(self, positives_info, threshold_hu=-700):
        """
        Builds a mask for the annotations.

        Args:
            positives_info: A list of positive CandidateInfo namedtuples.
            threshold_hu: The Hounsfield unit threshold.

        Returns:
            A 3D boolean array representing the mask.
        """
        bounding_box = np.zeros_like(self.hu_a, dtype=np.bool)

        for candidate_info_tup in positives_info:
            center_Irc = xyz_to_irc(
                candidate_info_tup.center_xyz,
                self.origin_xyz,
                self.vx_size_xyz,
                self.direction_a,
            )
            center = int(center_Irc.index)
            center_row = int(center_Irc.row)
            center_column = int(center_Irc.col)

            index_radius = 2
            try:
                while self.hu_a[center + index_radius, center_row, center_column] > threshold_hu and \
                        self.hu_a[center - index_radius, center_row, center_column] > threshold_hu:
                    index_radius += 1
            except IndexError:
                index_radius -= 1

            row_radius = 2
            try:
                while self.hu_a[center, center_row + row_radius, center_column] > threshold_hu and \
                        self.hu_a[center, center_row - row_radius, center_column] > threshold_hu:
                    row_radius += 1
            except IndexError:
                row_radius -= 1

            col_radius = 2
            try:
                while self.hu_a[center, center_row, center_column + col_radius] > threshold_hu and \
                        self.hu_a[center, center_row, center_column - col_radius] > threshold_hu:
                    col_radius += 1
            except IndexError:
                col_radius -= 1

            bounding_box[
                center - index_radius: center + index_radius + 1,
                center_row - row_radius: center_row + row_radius + 1,
                center_column - col_radius: center_column + col_radius + 1] = True

        mask_a = bounding_box & (self.hu_a > threshold_hu)

        return mask_a

    def get_raw_candidate(self, center_xyz, width_Irc):
        """
        Gets a raw candidate chunk from the CT scan.

        Args:
            center_xyz: The center coordinates of the candidate.
            width_Irc: The width of the chunk in Irc coordinates.

        Returns:
            A tuple containing the CT chunk, positive mask chunk, and center Irc coordinates.
        """
        center_Irc = xyz_to_irc(center_xyz, self.origin_xyz, self.vx_size_xyz, self.direction_a)

        slices = []
        for axis, center_val in enumerate(center_Irc):
            start_index = int(round(center_val - width_Irc[axis] / 2))
            end_index = int(start_index + width_Irc[axis])

            assert center_val >= 0 and center_val < self.hu_a.shape[axis], repr([self.series_uid, center_xyz, self.origin_xyz, self.vx_size_xyz, center_Irc, axis])

            if start_index < 0:
                start_index = 0
                end_index = int(width_Irc[axis])

            if end_index > self.hu_a.shape[axis]:
                end_index = self.hu_a.shape[axis]
                start_index = int(self.hu_a.shape[axis] - width_Irc[axis])

            slices.append(slice(start_index, end_index))

        ct_chunk = self.hu_a[tuple(slices)]
        pos_chunk = self.positive_mask[tuple(slices)]

        return ct_chunk, pos_chunk, center_Irc

@functools.lru_cache(1, typed=True)
def get_Ct(series_uid):
    """
    Retrieves a CT scan object, cached for efficiency.

    Args:
        series_uid: The series UID of the CT scan.

    Returns:
        A Ct object.
    """
    return Ct(series_uid)

@raw_cache.memoize(typed=True)
def get_Ct_raw_candidate(series_uid, center_xyz, width_Irc):
    """
    Retrieves a raw candidate chunk from a CT scan, cached for efficiency.

    Args:
        series_uid: The series UID of the CT scan.
        center_xyz: The center coordinates of the candidate.
        width_Irc: The width of the chunk in Irc coordinates.

    Returns:
        A tuple containing the CT chunk, positive mask chunk, and center Irc coordinates.
    """
    ct = get_Ct(series_uid)
    ct_chunk, pos_chunk, center_Irc = ct.get_raw_candidate(center_xyz, width_Irc)
    ct_chunk.clip(-1000, 1000, ct_chunk)
    return ct_chunk, pos_chunk, center_Irc

@raw_cache.memoize(typed=True)
def get_Ct_sample_size(series_uid):
    """
    Retrieves the sample size and positive indexes for a CT scan, cached for efficiency.

    Args:
        series_uid: The series UID of the CT scan.

    Returns:
        A tuple containing the sample size and a list of positive indexes.
    """
    ct = Ct(series_uid)
    return int(ct.hu_a.shape[0]), ct.positive_indexes

class Luna2dSegmentationDataset(Dataset):
    def __init__(self, val_stride=0, is_value_set=None, series_uid=None, contextSlices_count=3, fullCt_bool=False):
        """
        Initializes the Luna2dSegmentationDataset.

        Args:
            val_stride: The stride for validation splitting.
            is_value_set: Whether this dataset is a validation set.
            series_uid: The series UID to use for a single-series dataset.
            contextSlices_count: The number of context slices to include.
            fullCt_bool: Whether to use the full CT scan.
        """
        self.contextSlices_count = contextSlices_count
        self.fullCt_bool = fullCt_bool

        if series_uid:
            self.series_list = [series_uid]
        else:
            self.series_list = sorted(get_candidate_info().keys())

        if is_value_set:
            assert val_stride > 0, val_stride
            self.series_list = self.series_list[::val_stride]
            assert self.series_list
        elif val_stride > 0:
            del self.series_list[::val_stride]
            assert self.series_list

        self.sample_list = []
        for series_uid in self.series_list:
            index_count, positive_indexes = get_Ct_sample_size(series_uid)

            if self.fullCt_bool:
                self.sample_list += [(series_uid, slice_index) for slice_index in range(index_count)]
            else:
                self.sample_list += [(series_uid, slice_index) for slice_index in positive_indexes]

        self.candidates_info = get_candidates_info()

        series_set = set(self.series_list)
        self.candidates_info = [cit for cit in self.candidates_info if cit.series_uid in series_set]

        self.positives = [nt for nt in self.candidates_info if nt.is_nodule]

        log.info("{!r}: {} {} series, {} slices, {} nodules".format(self, len(self.series_list), {None: 'general', True: 'validation', False: 'training'}[is_value_set], len(self.sample_list), len(self.positives)))

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            The number of samples in the dataset.
        """
        return len(self.sample_list)

    def __getitem__(self, index):
        """
        Retrieves a single item from the dataset.

        Args:
            index: The index of the item to retrieve.

        Returns:
            A tuple containing the CT slice tensor, positive mask tensor, series UID, and slice index.
        """
        series_uid, slice_index = self.sample_list[index % len(self.sample_list)]
        return self.getitem_fullSlice(series_uid, slice_index)

    def getitem_fullSlice(self, series_uid, slice_index):
        """
        Retrieves a full CT slice and its corresponding mask.

        Args:
            series_uid: The series UID of the CT scan.
            slice_index: The index of the slice.

        Returns:
            A tuple containing the CT slice tensor, positive mask tensor, series UID, and slice index.
        """
        ct = get_Ct(series_uid)
        ct_t = torch.zeros((self.contextSlices_count * 2 + 1, 512, 512))

        start_index = slice_index - self.contextSlices_count
        end_index = slice_index + self.contextSlices_count + 1
        for i, context_index in enumerate(range(start_index, end_index)):
            context_index = max(context_index, 0)
            context_index = min(context_index, ct.hu_a.shape[0] - 1)
            ct_t[i] = torch.from_numpy(ct.hu_a[context_index].astype(np.float32))

        ct_t.clamp_(-1000, 1000)

        pos_t = torch.from_numpy(ct.positive_mask[slice_index]).unsqueeze(0)

        return ct_t, pos_t, ct.series_uid, slice_index


class TrainingLuna2dSegmentationDataset(Luna2dSegmentationDataset):
    def __init__(self, *args, **kwargs):
        """
        Initializes the TrainingLuna2dSegmentationDataset.

        Args:
            *args: Positional arguments passed to the parent class.
            **kwargs: Keyword arguments passed to the parent class.
        """
        super().__init__(*args, **kwargs)
        self.ratio_int = 2

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            The number of samples in the dataset.
        """
        return 300000

    def shuffle_samples(self):
        """
        Shuffles the samples in the dataset.
        """
        random.shuffle(self.candidates_info)
        random.shuffle(self.positives)

    def __getitem__(self, index):
        """
        Retrieves a single item from the dataset.

        Args:
            index: The index of the item to retrieve.

        Returns:
            A tuple containing the CT crop tensor, positive mask tensor, series UID, and slice index.
        """
        candidate_info_tup = self.positives[index % len(self.positives)]
        return self.getitem_training_crop(candidate_info_tup)

    def getitem_training_crop(self, candidate_info_tup):
        """
        Retrieves a training crop centered around a candidate.

        Args:
            candidate_info_tup: The CandidateInfo namedtuple for the candidate.

        Returns:
            A tuple containing the CT crop tensor, positive mask tensor, series UID, and slice index.
        """
        ct_a, pos_a, center_Irc = get_Ct_raw_candidate(candidate_info_tup.series_uid, candidate_info_tup.center_xyz, (7, 96, 96))
        pos_a = pos_a[3:4]

        row_offset = random.randrange(0, 32)
        col_offset = random.randrange(0, 32)
        ct_t = torch.from_numpy(ct_a[:, row_offset:row_offset + 64, col_offset:col_offset + 64]).to(torch.float32)
        pos_t = torch.from_numpy(pos_a[:, row_offset:row_offset + 64, col_offset:col_offset + 64]).to(torch.long)

        slice_index = center_Irc.index

        return ct_t, pos_t, candidate_info_tup.series_uid, slice_index

class PrepcacheLunaDataset(Dataset):
    def __init__(self, *args, **kwargs):
        """
        Initializes the PrepcacheLunaDataset.

        Args:
            *args: Positional arguments passed to the parent class.
            **kwargs: Keyword arguments passed to the parent class.
        """
        super().__init__(*args, **kwargs)
        self.candidates_info = get_candidates_info()
        self.positives = [nt for nt in self.candidates_info if nt.is_nodule]
        self.seen_set = set()
        self.candidates_info.sort(key=lambda x: x.series_uid)

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            The number of candidates in the dataset.
        """
        return len(self.candidates_info)

    def __getitem__(self, index):
        """
        Retrieves a single item from the dataset and preloads it into the cache.

        Args:
            index: The index of the item to retrieve.

        Returns:
            A tuple containing dummy values.
        """
        candidate_info_tup = self.candidates_info[index]
        get_Ct_raw_candidate(candidate_info_tup.series_uid, candidate_info_tup.center_xyz, (7, 96, 96))

        series_uid = candidate_info_tup.series_uid
        if series_uid not in self.seen_set:
            self.seen_set.add(series_uid)
            get_Ct_sample_size(series_uid)

        return 0, 1


class TvTrainingLuna2dSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, is_value_set=False, val_stride=10, contextSlices_count=3):
        """
        Initializes the TvTrainingLuna2dSegmentationDataset.

        Args:
            is_value_set: Whether this dataset is a validation set.
            val_stride: The stride for validation splitting.
            contextSlices_count: The number of context slices to include.
        """
        assert contextSlices_count == 3
        data = torch.load('./imgs_and_masks.pt')
        suids = list(set(data['suids']))
        trn_mask_suids = torch.arange(len(suids)) % val_stride < (val_stride - 1)
        trn_suids = {s for i, s in zip(trn_mask_suids, suids) if i}
        trn_mask = torch.tensor([(s in trn_suids) for s in data["suids"]])
        if not is_value_set:
            self.imgs = data["imgs"][trn_mask]
            self.masks = data["masks"][trn_mask]
            self.suids = [s for s, i in zip(data["suids"], trn_mask) if i]
        else:
            self.imgs = data["imgs"][~trn_mask]
            self.masks = data["masks"][~trn_mask]
            self.suids = [s for s, i in zip(data["suids"], trn_mask) if not i]
        self.imgs.clamp_(-1000, 1000)
        self.imgs /= 1000

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            The number of samples in the dataset.
        """
        return len(self.imgs)

    def __getitem__(self, i):
        """
        Retrieves a single item from the dataset.

        Args:
            i: The index of the item to retrieve.

        Returns:
            A tuple containing the CT slice tensor, a dummy label, the positive mask tensor, the series UID, and a dummy slice index.
        """
        oh, ow = torch.randint(0, 32, (2,))
        sl = self.masks.size(1) // 2
        return self.imgs[i, :, oh: oh + 64, ow: ow + 64], 1, self.masks[i, sl: sl + 1, oh: oh + 64, ow: ow + 64].to(torch.float32), self.suids[i], 9999
