"""BoT-SORT (Aharon et al., 2022), vendored from the official implementation.

Source: https://github.com/NirAharon/BoT-SORT at 251985436d6712aaf682aaaf5f71edb4987224bd, MIT licence
(``LICENSE`` in this folder). Every change is marked ``lgep:`` in the code; ``bot_sort.py`` and ``gmc.py``
list them.
"""

from lgep.tracking.botsort.bot_sort import BoTSORT, STrack
from lgep.tracking.botsort.gmc import GMC

__all__ = ["BoTSORT", "GMC", "STrack"]
