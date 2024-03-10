import collections
import numpy as np
from util.logconf import logging
import datetime
import time

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

# Define namedtuple for coordinates in Index-Row-Column (Irc) and X-Y-Z (Xyz) formats.
Irc = collections.namedtuple('Irc', ['index', 'row', 'col'])
Xyz = collections.namedtuple('Xyz', ['x', 'y', 'z'])


def print_histogram(ary, prefix_str=None, **kwargs):
    """
    Prints a histogram of an array.

    Args:
        ary: The array to print the histogram for.
        prefix_str: A prefix string to add to each line of the histogram (optional).
        **kwargs: Additional keyword arguments for numpy.histogram().
    """
    if prefix_str is None:
        prefix_str = ''
    else:
        prefix_str += ' '

    count_ary, bins_ary = np.histogram(ary, **kwargs)
    for i in range(count_ary.shape[0]):
        print("{}{:-8.2f}".format(prefix_str, bins_ary[i]), "{:-10}".format(count_ary[i]))
    print("{}{:-8.2f}".format(prefix_str, bins_ary[-1]))


def xyz_to_irc(coord_xyz, origin_xyz, vx_size_xyz, direction_a):
    """
    Converts coordinates from X-Y-Z format to Index-Row-Column format.

    Args:
        coord_xyz: The X-Y-Z coordinates.
        origin_xyz: The origin in X-Y-Z format.
        vx_size_xyz: The voxel size in X-Y-Z format.
        direction_a: The direction matrix.

    Returns:
        The coordinates in Index-Row-Column format.
    """
    origin_a = np.array(origin_xyz)
    vxSize_a = np.array(vx_size_xyz)
    coord_a = np.array(coord_xyz)
    cri_a = ((coord_a - origin_a) @ np.linalg.inv(direction_a)) / vxSize_a
    cri_a = np.round(cri_a)
    return Irc(int(cri_a[2]), int(cri_a[1]), int(cri_a[0]))


def irc_to_xyz(coord_Irc, origin_xyz, vx_size_xyz, direction_a):
    """
    Converts coordinates from Index-Row-Column format to X-Y-Z format.

    Args:
        coord_Irc: The Index-Row-Column coordinates.
        origin_xyz: The origin in X-Y-Z format.
        vx_size_xyz: The voxel size in X-Y-Z format.
        direction_a: The direction matrix.

    Returns:
        The coordinates in X-Y-Z format.
    """
    cri_a = np.array(coord_Irc)[::-1]
    origin_a = np.array(origin_xyz)
    vxSize_a = np.array(vx_size_xyz)
    coords_xyz = (direction_a @ (cri_a * vxSize_a)) + origin_a
    return Xyz(*coords_xyz)


def enumerate_with_estimate(iter, desc_str, start_index=0, print_index=4, backoff=None, iter_len=None):
    """
    Enumerates an iterable with estimates for completion time.

    Args:
        iter: The iterable to enumerate.
        desc_str: A description string for logging.
        start_index: The starting index for enumeration.
        print_index: The index at which to start printing estimates.
        backoff: The backoff factor for printing estimates.
        iter_len: The length of the iterable (optional).

    Yields:
        The current index and item of the iterable.
    """
    if iter_len is None:
        iter_len = len(iter)

    if backoff is None:
        backoff = 2
        while backoff ** 7 < iter_len:
            backoff *= 2

    assert backoff >= 2
    while print_index < start_index * backoff:
        print_index *= backoff

    log.warning("{} ----/{}, starting".format(desc_str, iter_len))
    start_ts = time.time()
    for (current_index, item) in enumerate(iter):
        yield (current_index, item)
        if current_index == print_index:
            duration_sec = ((time.time() - start_ts) / (current_index - start_index + 1) * (iter_len - start_index))
            done_dt = datetime.datetime.fromtimestamp(start_ts + duration_sec)
            done_td = datetime.timedelta(seconds=duration_sec)
            log.info("{} {:-4}/{}, will be done at approx. {}, {} ..."
                     "".format(desc_str, current_index, iter_len, str(done_dt).rsplit('.', 1)[0], str(done_td).rsplit('.', 1)[0]))
            print_index *= backoff

        if current_index + 1 == start_index:
            start_ts = time.time()
