import collections
import datetime
import time
import numpy as np
from util.logconf import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

Irc = collections.namedtuple('Irc', ['index', 'row', 'col'])
Xyz = collections.namedtuple('Xyz', ['x', 'y', 'z'])

def xyz2Irc(coord_xyz, origin_xyz, vx_size_xyz, direction_a):
    origin_a = np.array(origin_xyz)
    vxSize_a = np.array(vx_size_xyz)
    coord_a = np.array(coord_xyz)
    cri_a = ((coord_a - origin_a) @ np.linalg.inv(direction_a)) / vxSize_a
    cri_a = np.round(cri_a)
    return Irc(int(cri_a[2]), int(cri_a[1]), int(cri_a[0]))

def Irc2xyz(coord_Irc, origin_xyz, vx_size_xyz, direction_a):
    cri_a = np.array(coord_Irc)[::-1]
    origin_a = np.array(origin_xyz)
    vxSize_a = np.array(vx_size_xyz)
    coords_xyz = (direction_a @ (cri_a * vxSize_a)) + origin_a
    return Xyz(*coords_xyz)


def importstr(module_str, from_=None):

    if from_ is None and ':' in module_str:
        module_str, from_ = module_str.rsplit(':')

    module = __import__(module_str)
    for sub_str in module_str.split('.')[1:]:
        module = getattr(module, sub_str)

    if from_:
        try:
            return getattr(module, from_)
        except:
            raise ImportError('{}.{}'.format(module_str, from_))
    return module

def prhist(ary, prefix_str=None, **kwargs):
    if prefix_str is None:
        prefix_str = ''
    else:
        prefix_str += ' '

    count_ary, bins_ary = np.histogram(ary, **kwargs)
    for i in range(count_ary.shape[0]):
        print("{}{:-8.2f}".format(prefix_str, bins_ary[i]), "{:-10}".format(count_ary[i]))
    print("{}{:-8.2f}".format(prefix_str, bins_ary[-1]))

def enumerate_with_estimate(iter, desc_str, start_index=0, print_index=4, backoff=None, iter_len=None):
    if iter_len is None:
        iter_len = len(iter)

    if backoff is None:
        backoff = 2
        while backoff ** 7 < iter_len:
            backoff *= 2

    assert backoff >= 2
    while print_index < start_index * backoff:
        print_index *= backoff

    log.warning("{} ----/{}, starting".format(
        desc_str,
        iter_len,
    ))
    start_ts = time.time()
    for (current_index, item) in enumerate(iter):
        yield (current_index, item)
        if current_index == print_index:
            duration_sec = ((time.time() - start_ts)
                            / (current_index - start_index + 1)
                            * (iter_len-start_index)
                            )

            done_dt = datetime.datetime.fromtimestamp(start_ts + duration_sec)
            done_td = datetime.timedelta(seconds=duration_sec)

            log.info("{} {:-4}/{}, will be done at approx. {}, {} ...".format(
                desc_str,
                current_index,
                iter_len,
                str(done_dt).rsplit('.', 1)[0],
                str(done_td).rsplit('.', 1)[0],
            ))

            print_index *= backoff

        if current_index + 1 == start_index:
            start_ts = time.time()