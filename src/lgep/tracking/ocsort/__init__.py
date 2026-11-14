"""OC-SORT (Cao et al., CVPR 2023), vendored from the official implementation.

Source: https://github.com/noahcao/OC_SORT at 8462e7e729a93ccd3bd995c0a79a890336cb3a0b, MIT licence
(``LICENSE`` in this folder). Every change is marked ``lgep:`` in the code; ``ocsort.py`` lists them.
"""

from lgep.tracking.ocsort.ocsort import OCSort, OCTrack

__all__ = ["OCSort", "OCTrack"]
