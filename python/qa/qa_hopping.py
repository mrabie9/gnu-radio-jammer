#!/usr/bin/env python3

import unittest

from gpsk_comms.hopping import (
    BAND_2G4,
    BAND_5G2,
    BAND_WIDTH,
    DEFAULT_LO_CHANNELS,
    DEFAULT_SUBCHANNELS,
    BandEscapeController,
    HopPlan,
    TimedRetuner,
)
from gpsk_comms.security import derive_keys

KEYS = derive_keys(bytes(range(32)))
OTHER = derive_keys(bytes(range(1, 33)))


class HopPlanTest(unittest.TestCase):
    def setUp(self):
        self.plan = HopPlan(KEYS.hop)

    def test_slot_count_is_the_product_of_both_tiers(self):
        self.assertEqual(self.plan.slot_count, DEFAULT_LO_CHANNELS * DEFAULT_SUBCHANNELS)

    def test_order_is_a_permutation_of_every_slot(self):
        order = self.plan.order(0)
        self.assertEqual(sorted(order), list(range(self.plan.slot_count)))

    def test_order_is_reproducible_but_varies_with_epoch_and_key(self):
        self.assertEqual(self.plan.order(0), self.plan.order(0))
        self.assertNotEqual(self.plan.order(0), self.plan.order(1))
        self.assertNotEqual(self.plan.order(0), HopPlan(OTHER.hop).order(0))

    def test_decompose_round_trips(self):
        for slot in range(self.plan.slot_count):
            lo_index, sub_index = self.plan.decompose(slot)
            self.assertEqual(lo_index * DEFAULT_SUBCHANNELS + sub_index, slot)

    def test_lo_frequencies_stay_inside_the_allocation(self):
        for band in (BAND_2G4, BAND_5G2):
            frequencies = [
                self.plan.lo_frequency(band, slot) for slot in range(self.plan.slot_count)
            ]
            self.assertGreaterEqual(min(frequencies), band - BAND_WIDTH / 2)
            self.assertLessEqual(max(frequencies), band + BAND_WIDTH / 2)

    def test_lo_comb_is_centred_on_the_band(self):
        frequencies = [
            self.plan.lo_frequency(BAND_2G4, slot) for slot in range(self.plan.slot_count)
        ]
        self.assertAlmostEqual((min(frequencies) + max(frequencies)) / 2, BAND_2G4, delta=1.0)

    def test_lo_channels_are_distinct_and_evenly_spaced(self):
        frequencies = sorted(
            {
                self.plan.lo_frequency(BAND_2G4, slot * DEFAULT_SUBCHANNELS)
                for slot in range(DEFAULT_LO_CHANNELS)
            }
        )
        self.assertEqual(len(frequencies), DEFAULT_LO_CHANNELS)
        gaps = [b - a for a, b in zip(frequencies, frequencies[1:])]
        for gap in gaps:
            self.assertAlmostEqual(gap, gaps[0], delta=1.0)

    def test_subchannel_offsets_avoid_the_window_edges(self):
        sample_rate = 8_000_000
        plan = HopPlan(KEYS.hop, sample_rate=sample_rate)
        offsets = [plan.subchannel_offset(slot) for slot in range(plan.slot_count)]
        # Comfortably inside Nyquist, where the anti-alias response is flat.
        self.assertLess(max(abs(offset) for offset in offsets), 0.45 * sample_rate)
        self.assertAlmostEqual(sum(offsets) / len(offsets), 0.0, delta=1.0)

    def test_tuning_returns_a_consistent_pair(self):
        frequency, offset = self.plan.tuning(BAND_2G4, epoch=3, position=5)
        slot = self.plan.slot_at(3, 5)
        self.assertEqual(frequency, self.plan.lo_frequency(BAND_2G4, slot))
        self.assertEqual(offset, self.plan.subchannel_offset(slot))

    def test_position_wraps_within_an_epoch(self):
        count = self.plan.slot_count
        self.assertEqual(self.plan.slot_at(0, 0), self.plan.slot_at(0, count))

    def test_blacklist_excludes_slots_without_breaking_the_schedule(self):
        self.plan.set_blacklist({0, 1, 2})
        order = self.plan.order(5)
        self.assertEqual(len(order), self.plan.slot_count - 3)
        self.assertTrue(all(slot not in order for slot in (0, 1, 2)))
        self.assertEqual(self.plan.blacklist, {0, 1, 2})

    def test_two_plans_with_the_same_key_and_blacklist_stay_in_step(self):
        """The property a one-way link depends on: no negotiation, same order."""
        transmitter = HopPlan(KEYS.hop)
        receiver = HopPlan(KEYS.hop)
        transmitter.set_blacklist({7, 19})
        receiver.set_blacklist({7, 19})
        for epoch in range(20):
            self.assertEqual(transmitter.order(epoch), receiver.order(epoch))

    def test_invalid_geometry_is_rejected(self):
        for kwargs in ({"lo_channels": 0}, {"subchannels": 0}, {"lo_channels": -1}):
            with self.assertRaises(ValueError):
                HopPlan(KEYS.hop, **kwargs)


class BandEscapeTest(unittest.TestCase):
    def test_starts_on_the_first_band(self):
        self.assertEqual(BandEscapeController().band, BAND_2G4)

    def test_escapes_after_enough_consecutive_jammed_dwells(self):
        controller = BandEscapeController(dwells_before_escape=4)
        for _ in range(3):
            self.assertEqual(controller.observe("jammed"), BAND_2G4)
        self.assertEqual(controller.observe("jammed"), BAND_5G2)
        self.assertEqual(controller.escapes, 1)

    def test_a_good_dwell_resets_the_counter(self):
        controller = BandEscapeController(dwells_before_escape=4)
        controller.observe("jammed")
        controller.observe("jammed")
        controller.observe("ok")
        self.assertEqual(controller.consecutive_jammed, 0)
        for _ in range(3):
            controller.observe("jammed")
        self.assertEqual(controller.band, BAND_2G4)

    def test_signal_loss_never_triggers_an_escape(self):
        """An outage must not be answered by adding an acquisition gap to it."""
        controller = BandEscapeController(dwells_before_escape=2)
        for _ in range(50):
            controller.observe("no_signal")
        self.assertEqual(controller.escapes, 0)
        self.assertEqual(controller.band, BAND_2G4)

    def test_degraded_neither_escapes_nor_clears(self):
        controller = BandEscapeController(dwells_before_escape=3)
        controller.observe("jammed")
        controller.observe("degraded")
        self.assertEqual(controller.consecutive_jammed, 1)
        controller.observe("jammed")
        controller.observe("jammed")
        self.assertEqual(controller.escapes, 1)

    def test_escape_alternates_between_the_two_bands(self):
        controller = BandEscapeController(dwells_before_escape=1)
        self.assertEqual(controller.observe("jammed"), BAND_5G2)
        self.assertEqual(controller.observe("jammed"), BAND_2G4)

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            BandEscapeController(bands=(BAND_2G4,))
        with self.assertRaises(ValueError):
            BandEscapeController(dwells_before_escape=0)


class TimedRetunerTest(unittest.TestCase):
    def test_records_requests_without_a_device(self):
        retuner = TimedRetuner()
        retuner.retune(BAND_2G4, at_time=1.5)
        retuner.retune(BAND_5G2)
        self.assertEqual(retuner.history, [(BAND_2G4, 1.5), (BAND_5G2, None)])

    def test_drives_a_device_and_brackets_timed_commands(self):
        class FakeDevice:
            def __init__(self):
                self.calls = []

            def set_command_time(self, spec):
                self.calls.append(("set_command_time", spec))

            def clear_command_time(self):
                self.calls.append(("clear_command_time",))

            def set_center_freq(self, frequency, channel):
                self.calls.append(("set_center_freq", frequency, channel))

        device = FakeDevice()
        TimedRetuner(device).retune(BAND_2G4)
        self.assertEqual(device.calls, [("set_center_freq", float(BAND_2G4), 0)])

        device = FakeDevice()
        TimedRetuner(device).retune(BAND_5G2, at_time=2.0)
        names = [call[0] for call in device.calls]
        self.assertEqual(names, ["set_command_time", "set_center_freq", "clear_command_time"])


if __name__ == "__main__":
    unittest.main()
