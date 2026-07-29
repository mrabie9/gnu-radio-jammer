"""Reusable GNU Radio blocks for a robot-command link.

Two links live here:

* ``gmsk_command_tx`` / ``gmsk_command_rx`` -- the original plain GMSK link.
  Simple and low bandwidth, but unauthenticated and with no jam resistance:
  anyone can forge a command and a weak in-band tone denies it entirely. Use it
  only where no adversary can reach the RF.
* ``aj_command_tx`` / ``aj_command_rx`` -- the hardened link. Authenticated
  frames, spread spectrum, forward error correction, interleaving, narrowband
  excision and frequency hopping, switchable one layer at a time by ``level``.

The hardened link is built as a ladder: ``level=0`` is a bare point-to-point
BPSK link needing no key, and each level above adds exactly one mechanism up to
``level=6``. Bring a link up from the bottom -- see :mod:`gpsk_comms.levels` --
rather than switching the whole stack on and guessing which layer is at fault.
"""

from .aj_command import aj_command_rx, aj_command_tx
from .gmsk_command_rx import gmsk_command_rx
from .gmsk_command_tx import gmsk_command_tx
from .levels import LEVEL_NAMES, LinkProfile, ladder, profile

__all__ = [
    "gmsk_command_tx",
    "gmsk_command_rx",
    "aj_command_tx",
    "aj_command_rx",
    "profile",
    "ladder",
    "LinkProfile",
    "LEVEL_NAMES",
]

