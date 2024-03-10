import copy
import csv
import random
import functools
import glob
import math
import torch
import torch.cuda
import torch.nn.functional as functional
import SimpleITK as sitk
import numpy as np
import os
from torch.utils.data import Dataset
from collections import namedtuple
from util.util import Xyz, xyz_to_irc
from util.disk import get_cache

Mask = namedtuple(
    'Mask',
    'raw_dense_mask, dense_mask, body_mask, air_mask, raw_candidate_mask, candidate_mask, lung_mask, neg_mask, pos_mask',
)
Candidate = namedtuple(
    'CandidateInfo',
    'is_nodule, has_annotation, is_malignant, diameter_mm, series_uid, center_xyz',
)

raw_cache = get_cache('classification')


@functools.lru_cache(1)
def get_candidates_info(required_on_disk=True):
    """
    Get information about the candidates from the dataset.

    Args:
        required_on_disk (bool, optional): Whether to require the data to be present on disk. Defaults to True.

    Returns:
        list: A list of Candidate namedtuples.
    """
    mhd_list = glob.glob('data-unversioned/part2/luna/subset*/*.mhd')
    present_on_disk = {os.path.split(p)[-1][:-4] for p in mhd_list}

    candidates = []
    with open('data/part2/luna/annotations_with_malignancy.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            annotation_center = tuple([float(x) for x in row[1:4]])
            annotation_diameter = float(row[4])
            is_malignant = {'False': False, 'True': True}[row[5]]

            candidates.append(Candidate(True, True, is_malignant, annotation_diameter, series_uid, annotation_center))

    with open('data/part2/luna/candidates.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]

            # Skip candidates that are required on disk but not present
            if series_uid not in present_on_disk and required_on_disk:
                continue

            is_nodule = bool(int(row[4]))
            candidate_center_xyz = tuple([float(x) for x in row[1:4]])

            if not is_nodule:
                candidates.append(Candidate(
                    False,
                    False,
                    False,
                    0.0,
                    series_uid,
                    candidate_center_xyz,
                ))

    # Sort the candidates by their likelihood (descending order)
    candidates.sort(reverse=True)
    return candidates

@functools.lru_cache(1)
def get_candidate_info(required_on_disk=True):
    """
    Get detailed information about each candidate.

    Args:
        required_on_disk (bool, optional): Whether to require the data to be present on disk. Defaults to True.

    Returns:
        dict: A dictionary mapping series_uids to lists of Candidate namedtuples.
    """
    candidates_info = get_candidates_info(required_on_disk)
    candidate_dict = {}

    for candidate in candidates_info:
        candidate_dict.setdefault(candidate.series_uid, []).append(candidate)

    return candidate_dict


class Ct:
    """
    A class representing a CT scan.
    """
    def __init__(self, series_uid):
        """
        Initializes the Ct object.

        Args:
            series_uid (str): The series UID of the CT scan.
        """
        mhd_path = glob.glob(
            'data-unversioned/part2/luna/subset*/{}.mhd'.format(series_uid)
        )[0]

        ct_mhd = sitk.ReadImage(mhd_path)
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)
        # Clip the Hounsfield units (HU) to be within the range -1000 to 1000
        ct_a.clip(-1000, 1000, ct_a)

        self.series_uid = series_uid
        self.hu_a = ct_a

        self.origin_xyz = Xyz(*ct_mhd.GetOrigin())
        self.vx_size_xyz = Xyz(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

    def get_raw_candidate(self, center_xyz, width):
        """
        Get a raw candidate from the CT scan.

        Args:
            center_xyz (tuple): The center of the candidate in XYZ
            coordinates.
            width (tuple): The width of the candidate in IRC coordinates.

        Returns:
            tuple: A tuple containing the CT chunk and the center in IRC coordinates.
        """
        irc_center = xyz_to_irc(center_xyz, self.origin_xyz, self.vx_size_xyz, self.direction_a)

        slices = []
        for axis, center_val in enumerate(irc_center):
            start_index = int(round(center_val - width[axis] / 2))
            end_index = int(start_index + width[axis])

            assert center_val >= 0 and center_val < self.hu_a.shape[axis], repr([self.series_uid, center_xyz, self.origin_xyz, self.vx_size_xyz, irc_center, axis])

            if start_index < 0:
                start_index = 0
                end_index = int(width[axis])

            if end_index > self.hu_a.shape[axis]:
                end_index = self.hu_a.shape[axis]
                start_index = int(self.hu_a.shape[axis] - width[axis])

            slices.append(slice(start_index, end_index))

        ct_chunk = self.hu_a[tuple(slices)]

        return ct_chunk, irc_center

@functools.lru_cache(1, typed=True)
def get_Ct(series_uid):
    """
    Get a Ct object for a given series UID.

    Args:
        series_uid (str): The series UID of the CT scan.

    Returns:
        Ct: The Ct object.
    """
    return Ct(series_uid)



def get_Ct_augmented_candidate(
        augmentation_dict,
        series_uid, center_xyz, width,
        use_cache=True):
    """
    Get an augmented candidate from a CT scan.

    Args:
        augmentation_dict (dict): A dictionary specifying the augmentations to apply.
        series_uid (str): The series UID of the CT scan.
        center_xyz (tuple): The center of the candidate in XYZ coordinates.
        width (tuple): The width of the candidate in IRC coordinates.
        use_cache (bool, optional): Whether to use cached data. Defaults to True.

    Returns:
        tuple: A tuple containing the augmented CT chunk and the center in IRC coordinates.
    """
    if use_cache:
        ct_chunk, center = get_Ct_raw_candidate(series_uid, center_xyz, width)
    else:
        ct = get_Ct(series_uid)
        ct_chunk, center = ct.get_raw_candidate(center_xyz, width)

    ct_tensor = torch.tensor(ct_chunk).unsqueeze(0).unsqueeze(0).to(torch.float32)
    transform_tensor = torch.eye(4)

    for i in range(3):
        if 'flip' in augmentation_dict and random.random() > 0.5:
            transform_tensor[i, i] *= -1
        if 'offset' in augmentation_dict:
            offset = augmentation_dict['offset']
            transform_tensor[i, 3] = offset * (random.random() * 2 - 1)
        if 'scale' in augmentation_dict:
            scale = augmentation_dict['scale']
            transform_tensor[i, i] *= 1.0 + scale * (random.random() * 2 - 1)

        # Apply rotation augmentation if specified in the augmentation dictionary
        if 'rotate' in augmentation_dict:
            # Generate a random angle for rotation between 0 and 2*pi radians
            angle = random.random() * math.pi * 2

            # Construct the rotation matrix for the Z-axis based on the random angle
            # This is a 2D rotation matrix expanded to 3D, where the Z-axis is unaffected
            rotation_tensor = torch.tensor([
                [math.cos(angle), -math.sin(angle), 0, 0],  # Rotation for the X-axis
                [math.sin(angle), math.cos(angle), 0, 0],  # Rotation for the Y-axis
                [0, 0, 1, 0],  # No rotation for the Z-axis
                [0, 0, 0, 1],  # Homogeneous coordinate
            ])

            # Combine the existing transformation tensor with the new rotation matrix
            # This applies the rotation to any previous transformations (e.g., flip, offset, scale)
            transform_tensor @= rotation_tensor

    affine_tensor = functional.affine_grid(
        transform_tensor[:3].unsqueeze(0).to(torch.float32),
        ct_tensor.size(),
        align_corners=False,
    )

    augmented_chunk = functional.grid_sample(
        ct_tensor,
        affine_tensor,
        padding_mode='border',
        align_corners=False,
    ).to('cpu')

    if 'noise' in augmentation_dict:
        noise_t = torch.randn_like(augmented_chunk)
        noise_t *= augmentation_dict['noise']

        augmented_chunk += noise_t

    return augmented_chunk[0], center

@raw_cache.memoize(typed=True)
def get_Ct_raw_candidate(series_uid, center_xyz, width):
    """
    Get a raw candidate from a CT scan.

    Args:
        series_uid (str): The series UID of the CT scan.
        center_xyz (tuple): The center of the candidate in XYZ coordinates.
        width (tuple): The width of the candidate in IRC coordinates.

    Returns:
        tuple: A tuple containing the CT chunk and the center in IRC coordinates.
    """
    ct = get_Ct(series_uid)
    ct_chunk, irc_center = ct.get_raw_candidate(center_xyz, width)
    return ct_chunk, irc_center

class LunaDataset(Dataset):
    def __init__(self,
                 val_stride=0,
                 is_value_set=None,
                 series_uid=None,
                 sortby_str='random',
                 ratio_int=0,
                 augmentation_dict=None,
                 candidates_info=None,
            ):
        self.ratio_int = ratio_int
        self.augmentation_dict = augmentation_dict

        if candidates_info:
            self.candidates_info = copy.copy(candidates_info)
            self.use_cache = False
        else:
            self.candidates_info = copy.copy(get_candidates_info())
            self.use_cache = True

        if series_uid:
            self.series_list = [series_uid]
        else:
            self.series_list = sorted(set(candidate.series_uid for candidate in self.candidates_info))

        if is_value_set:
            assert val_stride > 0, val_stride
            self.series_list = self.series_list[::val_stride]
            assert self.series_list
        elif val_stride > 0:
            del self.series_list[::val_stride]
            assert self.series_list

        series_set = set(self.series_list)
        self.candidates_info = [x for x in self.candidates_info if x.series_uid in series_set]

        if sortby_str == 'random':
            random.shuffle(self.candidates_info)
        elif sortby_str == 'series_uid':
            self.candidates_info.sort(key=lambda x: (x.series_uid, x.center_xyz))
        elif sortby_str == 'label_and_size':
            pass
        else:
            raise Exception("Unknown sort: " + repr(sortby_str))

        self.negatives = \
            [nt for nt in self.candidates_info if not nt.is_nodule]
        self.positives = \
            [nt for nt in self.candidates_info if nt.is_nodule]
        self.ben_list = \
            [nt for nt in self.positives if not nt.is_malignant]
        self.mal_list = \
            [nt for nt in self.positives if nt.is_malignant]

        log.info("{!r}: {} {} samples, {} neg, {} pos, {} ratio".format(
            self,
            len(self.candidates_info),
            "validation" if is_value_set else "training",
            len(self.negatives),
            len(self.positives),
            '{}:1'.format(self.ratio_int) if self.ratio_int else 'unbalanced'
        ))

    def shuffle_samples(self): 
        if self.ratio_int:
            random.shuffle(self.candidates_info)
            random.shuffle(self.negatives)
            random.shuffle(self.positives)
            random.shuffle(self.ben_list)
            random.shuffle(self.mal_list)

    def __len__(self):
        if self.ratio_int:
            return 50000
        else:
            return len(self.candidates_info)

    def __getitem__(self, index):
        if self.ratio_int:
            pos_index = index // (self.ratio_int + 1)

            if index % (self.ratio_int + 1):
                neg_index = index - 1 - pos_index
                neg_index %= len(self.negatives)
                candidate = self.negatives[neg_index]
            else:
                pos_index %= len(self.positives)
                candidate = self.positives[pos_index]
        else:
            candidate = self.candidates_info[index]

        return self.sample_from_candidate(
            candidate, candidate.is_nodule
        )

    def sample_from_candidate(self, candidate, label_bool):
        width_Irc = (32, 48, 48)

        if self.augmentation_dict:
            candidate_t, center_Irc = get_Ct_augmented_candidate(
                self.augmentation_dict,
                candidate.series_uid,
                candidate.center_xyz,
                width_Irc,
                self.use_cache,
            )
        elif self.use_cache:
            candidate_a, center_Irc = get_Ct_raw_candidate(
                candidate.series_uid,
                candidate.center_xyz,
                width_Irc,
            )
            candidate_t = torch.from_numpy(candidate_a).to(torch.float32)
            candidate_t = candidate_t.unsqueeze(0)
        else:
            ct = get_Ct(candidate.series_uid)
            candidate_a, center_Irc = ct.get_raw_candidate(
                candidate.center_xyz,
                width_Irc,
            )
            candidate_t = torch.from_numpy(candidate_a).to(torch.float32)
            candidate_t = candidate_t.unsqueeze(0)

        label_t = torch.tensor([False, False], dtype=torch.long)

        if not label_bool:
            label_t[0] = True
            index_t = 0
        else:
            label_t[1] = True
            index_t = 1

        return candidate_t, label_t, index_t, candidate.series_uid, torch.tensor(center_Irc)


class MalignantLunaDataset(LunaDataset):
    def __len__(self):
        if self.ratio_int:
            return 100000
        else:
            return len(self.ben_list + self.mal_list)

    def __getitem__(self, index):
        if self.ratio_int:
            if index % 2 != 0:
                candidate = self.mal_list[(index // 2) % len(self.mal_list)]
            elif index % 4 == 0:
                candidate = self.ben_list[(index // 4) % len(self.ben_list)]
            else:
                candidate = self.negatives[(index // 4) % len(self.negatives)]
        else:
            if index >= len(self.ben_list):
                candidate = self.mal_list[index - len(self.ben_list)]
            else:
                candidate = self.ben_list[index]

        return self.sample_from_candidate(
            candidate, candidate.is_malignant
        )
