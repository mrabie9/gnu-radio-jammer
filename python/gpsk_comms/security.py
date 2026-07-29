"""Keyed primitives for the anti-jam command link.

Everything a jammer must not be able to predict is derived here from a single
master key: the frame authentication tag, the spreading code, the frame sync
word, and the frequency-hop order.

Scope of the protection this provides:

* The MAC is the only thing standing between the link and a forged ``forward``
  command. It is a genuine cryptographic guarantee.
* The hop sequence is the main source of low probability of intercept. An
  adversary who cannot follow the hops must spread their power across the whole
  allocation, which is exactly the trade this link is built to win.
* The spreading code contributes far less secrecy than the hop sequence. Gold
  codes are chosen for their bounded code-to-code cross-correlation and their
  acquisition behaviour, not for concealment -- the family offers only about
  10 bits of selection entropy (see :func:`pn_code`). Do not rely on it alone.
"""

import hashlib
import hmac
import os
import secrets
import struct

import numpy

KEY_SIZE = 32
MAC_SIZE = 4
SYNC_WORD_SIZE = 4

_HASH = hashlib.sha256
_HASH_SIZE = _HASH().digest_size

_MAC_INFO = b"gpsk_comms/v2/mac"
_PN_INFO = b"gpsk_comms/v2/pn"
_SYNC_INFO = b"gpsk_comms/v2/sync"
_HOP_INFO = b"gpsk_comms/v2/hop"

#: Sentinel key. The receiver refuses to start on this so that an unkeyed
#: deployment fails loudly at startup instead of silently accepting forgeries.
INSECURE_DEFAULT_KEY = bytes(KEY_SIZE)

# Preferred pairs of maximal-length LFSR tap sets, one pair per degree, used to
# build Gold code families. Taps are bit positions for the right-shifting
# Fibonacci register in _msequence_bits (tap ``t`` reads bit ``t - 1``), which is
# the reciprocal of the usual textbook convention -- hence every set includes
# tap 1.
#
# These were found by exhaustive search, not copied from a table: each pair was
# verified to have full 2**n - 1 period and a maximum cross-correlation at or
# below the preferred-pair bound t(n). The trailing comment is the measured
# max|cross-correlation| against that bound. qa_security.py re-checks both
# properties, so a bad entry fails the suite rather than quietly degrading
# acquisition.
_PREFERRED_PAIRS = {
    7: ([7, 1], [7, 1, 2, 4]),                    # 17 <= 17
    8: ([8, 1, 2, 3], [8, 1, 2, 7]),              # 31 <= 33
    9: ([9, 1, 2, 5], [9, 1, 3, 5]),              # 33 <= 33
    10: ([10, 1, 2, 5], [10, 1, 2, 7]),           # 63 <= 65
    11: ([11, 1, 2, 4], [11, 1, 3, 8]),           # 65 <= 65
    12: ([12, 1, 3, 11], [12, 1, 2, 3, 9, 11]),   # 127 <= 129
}


def preferred_pair_bound(degree):
    """Return the Gold-code cross-correlation bound t(n) for ``degree``."""
    exponent = (degree + 2) // 2 if degree % 2 == 0 else (degree + 1) // 2
    return 1 + (1 << exponent)


SUPPORTED_CODE_LENGTHS = tuple(sorted((1 << degree) - 1 for degree in _PREFERRED_PAIRS))


class KeyError_(ValueError):
    """Raised when key material is missing, malformed, or unsafe to use."""


class LinkKeys:
    """Per-purpose keys expanded from one master key."""

    __slots__ = ("mac", "pn", "sync", "hop")

    def __init__(self, mac, pn, sync, hop):
        self.mac = mac
        self.pn = pn
        self.sync = sync
        self.hop = hop

    def __repr__(self):
        # Never let key material reach a log line or a traceback.
        return "LinkKeys(<redacted>)"


def _hkdf(master_key, info, length=KEY_SIZE):
    """RFC 5869 HKDF-SHA256 with an empty salt."""
    prk = hmac.new(bytes(_HASH_SIZE), master_key, _HASH).digest()
    output = bytearray()
    block = b""
    counter = 1
    while len(output) < length:
        block = hmac.new(prk, block + info + bytes([counter]), _HASH).digest()
        output.extend(block)
        counter += 1
    return bytes(output[:length])


def generate_master_key():
    """Return a fresh master key suitable for writing to a key file."""
    return secrets.token_bytes(KEY_SIZE)


def load_master_key(path=None, allow_insecure=False):
    """Load a master key from a file, or from ``GPSK_COMMS_KEY`` if unset.

    The key is never a repository constant. A missing or all-zero key raises
    unless ``allow_insecure`` is set, which only the simulation harnesses do.
    """
    if path is None:
        path = os.environ.get("GPSK_COMMS_KEY_FILE")
    if path is None:
        hex_key = os.environ.get("GPSK_COMMS_KEY")
        if hex_key is None:
            if allow_insecure:
                return INSECURE_DEFAULT_KEY
            raise KeyError_(
                "no key material: set GPSK_COMMS_KEY_FILE or GPSK_COMMS_KEY "
                "(generate one with gpsk_comms.security.generate_master_key)"
            )
        raw = bytes.fromhex(hex_key.strip())
    else:
        with open(path, "rb") as handle:
            content = handle.read().strip()
        try:
            raw = bytes.fromhex(content.decode("ascii"))
        except (UnicodeDecodeError, ValueError):
            raw = content
    if len(raw) != KEY_SIZE:
        raise KeyError_(f"master key must be exactly {KEY_SIZE} bytes")
    if raw == INSECURE_DEFAULT_KEY and not allow_insecure:
        raise KeyError_("refusing to use the all-zero key")
    return raw


def derive_keys(master_key, allow_insecure=False):
    """Expand a master key into per-purpose subkeys."""
    if not isinstance(master_key, (bytes, bytearray)):
        raise KeyError_("master key must be bytes")
    master_key = bytes(master_key)
    if len(master_key) != KEY_SIZE:
        raise KeyError_(f"master key must be exactly {KEY_SIZE} bytes")
    if master_key == INSECURE_DEFAULT_KEY and not allow_insecure:
        raise KeyError_("refusing to derive keys from the all-zero key")
    return LinkKeys(
        mac=_hkdf(master_key, _MAC_INFO),
        pn=_hkdf(master_key, _PN_INFO),
        sync=_hkdf(master_key, _SYNC_INFO),
        hop=_hkdf(master_key, _HOP_INFO),
    )


#: Published master key for the unkeyed rungs of the feature ladder (levels 0
#: and 1). It is a constant in a public repository and provides no secrecy
#: whatsoever -- that is the point. Those levels exist to prove the radio works
#: before any key is involved, so the frame tag degrades to a plain 32-bit
#: integrity check and the sync word to a public constant. Nothing that derives
#: from this key resists an adversary; :func:`derive_keys` guards the real path.
PUBLIC_MASTER_KEY = hashlib.sha256(b"gpsk_comms/public-ladder/v1").digest()


def public_keys():
    """Return the published key set used by ladder levels 0 and 1.

    Having the unkeyed levels carry a full key set rather than special-casing
    them keeps one code path through the transmitter and receiver: between level
    1 and level 2 the *secrecy* of the key changes and nothing else does, so a
    failure appearing at level 2 is a failure of authentication and not of some
    separate unkeyed code path that only runs at the bottom of the ladder.
    """
    return derive_keys(PUBLIC_MASTER_KEY)


def frame_mac(mac_key, body):
    """Return the truncated HMAC-SHA256 tag authenticating ``body``."""
    return hmac.new(mac_key, bytes(body), _HASH).digest()[:MAC_SIZE]


def verify_mac(mac_key, body, tag):
    """Constant-time tag check."""
    return hmac.compare_digest(frame_mac(mac_key, body), bytes(tag))


def sync_word(sync_key, epoch, hop_index):
    """Return the keyed frame sync word for one hop.

    Rotating the sync word per hop means an adversary who has not broken the
    key cannot use it as a trigger for a reactive jammer.
    """
    material = struct.pack(">QI", int(epoch) & 0xFFFFFFFFFFFFFFFF, int(hop_index) & 0xFFFFFFFF)
    return hmac.new(sync_key, material, _HASH).digest()[:SYNC_WORD_SIZE]


def _keystream(key, material, count):
    """Return ``count`` pseudorandom bytes from HMAC in counter mode."""
    output = bytearray()
    counter = 0
    while len(output) < count:
        output.extend(hmac.new(key, material + struct.pack(">I", counter), _HASH).digest())
        counter += 1
    return bytes(output[:count])


def _msequence_bits(degree, taps, state=1):
    """Return one full period of a maximal-length LFSR as 0/1 ``uint8``."""
    length = (1 << degree) - 1
    mask = length
    output = numpy.empty(length, dtype=numpy.uint8)
    for index in range(length):
        output[index] = state & 1
        feedback = 0
        for tap in taps:
            feedback ^= (state >> (tap - 1)) & 1
        state = ((state >> 1) | (feedback << (degree - 1))) & mask
    return output


def _code_degree(length):
    degree = int(length).bit_length()
    if (1 << degree) - 1 != int(length) or degree not in _PREFERRED_PAIRS:
        raise ValueError(f"code length must be one of {SUPPORTED_CODE_LENGTHS}; got {length}")
    return degree


def pn_code(pn_key, hop_index, length, kind="gold"):
    """Return a length-``length`` spreading code as a +/-1 ``int8`` array.

    ``gold`` (default) selects one member of the Gold family built from the
    degree's preferred pair. This is the same construction GPS C/A uses at
    N=1023, and it is the default for the reason GPS chose it: cross-correlation
    between any two codes in the family is bounded by t(n) -- 63 at N=1023, about
    24 dB below the peak -- so one hop's code cannot masquerade as another's.
    Autocorrelation sidelobes obey the same bound, which is ample for acquisition
    well below 0 dB SNR.

    ``msequence`` gives ideal -1 autocorrelation sidelobes (about 60 dB PSR at
    N=1023) but **no cross-correlation guarantee at all**: two codes drawn from
    the same tap set are cyclic shifts of one another and correlate perfectly.
    Use it only for a single-channel link with no hopping.

    ``crypto`` derives every chip from HMAC. It is the only genuinely
    unpredictable option, at the cost of sidelobes near sqrt(length) -- about
    20 dB PSR at N=1023, measured.

    Processing gain against a white jammer is ``length`` for all three; they
    differ in acquisition robustness and in code-to-code isolation. Code choice
    contributes only around 10 bits of entropy, so concealment rests on the hop
    sequence and forgery resistance rests on the MAC -- never on the code alone.
    """
    length = int(length)
    if length <= 0:
        raise ValueError("length must be positive")
    material = struct.pack(">I", int(hop_index) & 0xFFFFFFFF)

    if kind == "crypto":
        raw = numpy.frombuffer(_keystream(pn_key, material, (length + 7) // 8), dtype=numpy.uint8)
        bits = numpy.unpackbits(raw)[:length]
        return (1 - 2 * bits.astype(numpy.int16)).astype(numpy.int8)

    degree = _code_degree(length)
    taps_u, taps_v = _PREFERRED_PAIRS[degree]
    selector = _keystream(pn_key, material, 8)

    if kind == "msequence":
        # Nonzero phase only: zero is the register's absorbing state.
        state = (int.from_bytes(selector[1:5], "big") % length) + 1
        taps = taps_u if selector[0] & 1 else taps_v
        bits = _msequence_bits(degree, taps, state)
    elif kind == "gold":
        # The family has length + 2 members: u, v, and u ^ (v cyclically shifted).
        index = int.from_bytes(selector[0:4], "big") % (length + 2)
        if index == 0:
            bits = _msequence_bits(degree, taps_u)
        elif index == 1:
            bits = _msequence_bits(degree, taps_v)
        else:
            bits = _msequence_bits(degree, taps_u) ^ numpy.roll(
                _msequence_bits(degree, taps_v), -(index - 2)
            )
    else:
        raise ValueError("kind must be 'gold', 'msequence', or 'crypto'")

    return (1 - 2 * bits.astype(numpy.int16)).astype(numpy.int8)


def hop_sequence(hop_key, epoch, channel_count, blacklist=()):
    """Return the per-epoch visiting order over ``channel_count`` channels.

    The order is a keyed shuffle rather than independent draws, so every usable
    channel is visited exactly once per epoch. That bounds worst-case dwell on
    any single channel and stops the sequence from lingering where a jammer sits.

    ``blacklist`` is the injection point for adaptive channel avoidance. It is
    unused on a one-way link, but a future reverse channel can populate it
    without a protocol revision -- both ends stay in step as long as they agree
    on the blacklist contents.
    """
    channel_count = int(channel_count)
    if channel_count <= 0:
        raise ValueError("channel_count must be positive")
    excluded = {index for index in blacklist if 0 <= index < channel_count}
    usable = [index for index in range(channel_count) if index not in excluded]
    if not usable:
        raise ValueError("blacklist excludes every channel")

    material = struct.pack(">Q", int(epoch) & 0xFFFFFFFFFFFFFFFF)
    # Four bytes of randomness per Fisher-Yates step keeps the modulo bias far
    # below the point where it would skew dwell statistics.
    stream = _keystream(hop_key, material, 4 * len(usable))
    for position in range(len(usable) - 1, 0, -1):
        draw = int.from_bytes(stream[4 * position : 4 * position + 4], "big")
        swap = draw % (position + 1)
        usable[position], usable[swap] = usable[swap], usable[position]
    return tuple(usable)


def _main(argv=None):
    """Generate a master key. Run as ``python3 -m gpsk_comms.security``."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Generate a gpsk_comms master key.")
    parser.add_argument(
        "--output", help="write the key here instead of stdout, with 0600 permissions"
    )
    arguments = parser.parse_args(argv)

    key = generate_master_key().hex()
    if arguments.output is None:
        print(key)
        print(
            "\nBoth ends need this key. Store it in a file and point "
            "GPSK_COMMS_KEY_FILE at it,\nor set GPSK_COMMS_KEY. Never commit it.",
            file=sys.stderr,
        )
        return 0

    descriptor = os.open(arguments.output, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(key + "\n")
    print(f"wrote {arguments.output} (mode 0600); copy it to the other end over a trusted path")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
