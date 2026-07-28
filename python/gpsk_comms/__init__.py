"""Reusable GNU Radio blocks for a robot-command link.

Two links live here:

* ``gmsk_command_tx`` / ``gmsk_command_rx`` -- the original plain GMSK link.
  Simple and low bandwidth, but unauthenticated and with no jam resistance:
  anyone can forge a command and a weak in-band tone denies it entirely. Use it
  only where no adversary can reach the RF.
* ``aj_command_tx`` / ``aj_command_rx`` -- the hardened link. Authenticated
  frames, spread spectrum, forward error correction, interleaving, narrowband
  excision and frequency hopping.
"""

from .aj_command import aj_command_rx, aj_command_tx
from .gmsk_command_rx import gmsk_command_rx
from .gmsk_command_tx import gmsk_command_tx

__all__ = [
    "gmsk_command_tx",
    "gmsk_command_rx",
    "aj_command_tx",
    "aj_command_rx",
]

