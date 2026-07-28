#!/usr/bin/env python3

import unittest

import numpy

from gpsk_comms.security import (
    INSECURE_DEFAULT_KEY,
    KEY_SIZE,
    MAC_SIZE,
    SUPPORTED_CODE_LENGTHS,
    SYNC_WORD_SIZE,
    KeyError_,
    _PREFERRED_PAIRS,
    _msequence_bits,
    derive_keys,
    frame_mac,
    generate_master_key,
    hop_sequence,
    pn_code,
    preferred_pair_bound,
    sync_word,
    verify_mac,
)

MASTER = bytes(range(32))
OTHER = bytes(range(1, 33))


def circular_correlation(first, second):
    length = len(first)
    spectrum = numpy.fft.rfft(first.astype(numpy.float64)) * numpy.conj(
        numpy.fft.rfft(second.astype(numpy.float64))
    )
    return numpy.fft.irfft(spectrum, length)


def peak_off_zero(first, second):
    correlation = numpy.abs(circular_correlation(first, second))
    if first is second:
        correlation = correlation[1:]
    return int(round(correlation.max()))


class KeyDerivationTest(unittest.TestCase):
    def test_subkeys_are_distinct_and_deterministic(self):
        keys = derive_keys(MASTER)
        material = (keys.mac, keys.pn, keys.sync, keys.hop)
        self.assertEqual(len(set(material)), 4)
        self.assertTrue(all(len(key) == KEY_SIZE for key in material))
        self.assertEqual(derive_keys(MASTER).mac, keys.mac)
        self.assertNotEqual(derive_keys(OTHER).mac, keys.mac)

    def test_all_zero_key_is_refused_unless_explicitly_allowed(self):
        with self.assertRaises(KeyError_):
            derive_keys(INSECURE_DEFAULT_KEY)
        self.assertIsNotNone(derive_keys(INSECURE_DEFAULT_KEY, allow_insecure=True))

    def test_malformed_keys_are_refused(self):
        for bad in (b"short", "not-bytes", bytes(KEY_SIZE + 1)):
            with self.assertRaises(KeyError_):
                derive_keys(bad)

    def test_generated_keys_are_fresh_and_correctly_sized(self):
        first, second = generate_master_key(), generate_master_key()
        self.assertEqual(len(first), KEY_SIZE)
        self.assertNotEqual(first, second)

    def test_repr_does_not_leak_key_material(self):
        keys = derive_keys(MASTER)
        self.assertNotIn(keys.mac.hex()[:8], repr(keys))


class MacTest(unittest.TestCase):
    def test_valid_tag_is_accepted(self):
        keys = derive_keys(MASTER)
        body = b"version-counter-command"
        tag = frame_mac(keys.mac, body)
        self.assertEqual(len(tag), MAC_SIZE)
        self.assertTrue(verify_mac(keys.mac, body, tag))

    def test_every_single_bit_flip_is_rejected(self):
        keys = derive_keys(MASTER)
        body = bytearray(b"version-counter-command")
        tag = frame_mac(keys.mac, body)
        for index in range(len(body)):
            for bit in range(8):
                corrupted = bytearray(body)
                corrupted[index] ^= 1 << bit
                self.assertFalse(verify_mac(keys.mac, corrupted, tag))

    def test_wrong_key_is_rejected(self):
        body = b"version-counter-command"
        self.assertFalse(
            verify_mac(derive_keys(OTHER).mac, body, frame_mac(derive_keys(MASTER).mac, body))
        )


class SyncWordTest(unittest.TestCase):
    def test_sync_word_varies_with_hop_epoch_and_key(self):
        keys = derive_keys(MASTER)
        base = sync_word(keys.sync, 0, 0)
        self.assertEqual(len(base), SYNC_WORD_SIZE)
        self.assertEqual(base, sync_word(keys.sync, 0, 0))
        self.assertNotEqual(base, sync_word(keys.sync, 0, 1))
        self.assertNotEqual(base, sync_word(keys.sync, 1, 0))
        self.assertNotEqual(base, sync_word(derive_keys(OTHER).sync, 0, 0))


class PnCodeTest(unittest.TestCase):
    """The tap tables were found by search, so re-derive their properties here.

    A wrong entry in _PREFERRED_PAIRS must fail the suite rather than quietly
    degrade acquisition, which is invisible until it is measured over the air.
    """

    def test_tap_sets_are_maximal_length(self):
        for degree, pair in _PREFERRED_PAIRS.items():
            for taps in pair:
                bits = _msequence_bits(degree, taps)
                length = (1 << degree) - 1
                self.assertEqual(len(bits), length)
                # A maximal-length sequence is balanced to within one chip and
                # has ideal -1 autocorrelation sidelobes.
                symbols = 1 - 2 * bits.astype(numpy.int16)
                self.assertEqual(abs(int(symbols.sum())), 1, f"degree {degree} taps {taps}")
                sidelobes = circular_correlation(symbols, symbols)[1:]
                self.assertTrue(
                    numpy.allclose(sidelobes, -1.0, atol=1e-6), f"degree {degree} taps {taps}"
                )

    def test_preferred_pairs_meet_the_cross_correlation_bound(self):
        for degree, (taps_u, taps_v) in _PREFERRED_PAIRS.items():
            first = 1 - 2 * _msequence_bits(degree, taps_u).astype(numpy.int16)
            second = 1 - 2 * _msequence_bits(degree, taps_v).astype(numpy.int16)
            self.assertLessEqual(
                peak_off_zero(first, second), preferred_pair_bound(degree), f"degree {degree}"
            )

    def test_gold_codes_stay_within_bound_across_hops(self):
        keys = derive_keys(MASTER)
        for length in SUPPORTED_CODE_LENGTHS:
            bound = preferred_pair_bound(length.bit_length())
            codes = [pn_code(keys.pn, hop, length) for hop in range(6)]
            for index, code in enumerate(codes):
                self.assertEqual(len(code), length)
                self.assertEqual(set(numpy.unique(code)), {-1, 1})
                self.assertLessEqual(peak_off_zero(code, code), bound, f"autocorr N={length}")
                for other in codes[index + 1 :]:
                    self.assertLessEqual(
                        peak_off_zero(code, other), bound, f"crosscorr N={length}"
                    )

    def test_codes_are_deterministic_key_dependent_and_hop_dependent(self):
        keys = derive_keys(MASTER)
        reference = pn_code(keys.pn, 3, 1023)
        numpy.testing.assert_array_equal(reference, pn_code(keys.pn, 3, 1023))
        self.assertFalse(numpy.array_equal(reference, pn_code(keys.pn, 4, 1023)))
        self.assertFalse(numpy.array_equal(reference, pn_code(derive_keys(OTHER).pn, 3, 1023)))

    def test_hop_codes_are_almost_always_distinct(self):
        # The Gold family holds length + 2 codes, so a birthday collision over a
        # few dozen hops is expected and harmless: hops are separated in
        # frequency and disambiguated by a per-hop sync word. Guard only against
        # a degenerate generator that returns one code for everything.
        keys = derive_keys(MASTER)
        distinct = {pn_code(keys.pn, hop, 1023).tobytes() for hop in range(64)}
        self.assertGreaterEqual(len(distinct), 60)

    def test_crypto_codes_are_balanced_and_unpredictable(self):
        keys = derive_keys(MASTER)
        code = pn_code(keys.pn, 0, 1023, kind="crypto")
        self.assertEqual(set(numpy.unique(code)), {-1, 1})
        self.assertLess(abs(int(code.sum())), 150)
        self.assertFalse(numpy.array_equal(code, pn_code(keys.pn, 1, 1023, kind="crypto")))

    def test_unsupported_lengths_and_kinds_are_rejected(self):
        keys = derive_keys(MASTER)
        for length in (0, -1, 100, 1024):
            with self.assertRaises(ValueError):
                pn_code(keys.pn, 0, length)
        with self.assertRaises(ValueError):
            pn_code(keys.pn, 0, 1023, kind="unknown")


class HopSequenceTest(unittest.TestCase):
    def test_sequence_is_a_permutation_of_every_channel(self):
        keys = derive_keys(MASTER)
        order = hop_sequence(keys.hop, 0, 12)
        self.assertEqual(sorted(order), list(range(12)))

    def test_sequence_varies_with_epoch_and_key_but_is_reproducible(self):
        keys = derive_keys(MASTER)
        order = hop_sequence(keys.hop, 0, 12)
        self.assertEqual(order, hop_sequence(keys.hop, 0, 12))
        self.assertNotEqual(order, hop_sequence(keys.hop, 1, 12))
        self.assertNotEqual(order, hop_sequence(derive_keys(OTHER).hop, 0, 12))

    def test_blacklist_removes_channels_and_keeps_the_rest(self):
        keys = derive_keys(MASTER)
        order = hop_sequence(keys.hop, 0, 12, blacklist=(3, 7))
        self.assertEqual(sorted(order), [c for c in range(12) if c not in (3, 7)])

    def test_dwell_is_spread_across_channels_over_many_epochs(self):
        keys = derive_keys(MASTER)
        counts = numpy.zeros(12, dtype=int)
        for epoch in range(200):
            for position, channel in enumerate(hop_sequence(keys.hop, epoch, 12)):
                if position == 0:
                    counts[channel] += 1
        # Each channel should lead roughly 200/12 ~ 17 epochs; a generator stuck
        # on one channel would leave zeros and defeat the whole point of hopping.
        self.assertTrue((counts > 0).all())
        self.assertLess(counts.max(), 60)

    def test_degenerate_inputs_are_rejected(self):
        keys = derive_keys(MASTER)
        with self.assertRaises(ValueError):
            hop_sequence(keys.hop, 0, 0)
        with self.assertRaises(ValueError):
            hop_sequence(keys.hop, 0, 4, blacklist=(0, 1, 2, 3))


if __name__ == "__main__":
    unittest.main()
