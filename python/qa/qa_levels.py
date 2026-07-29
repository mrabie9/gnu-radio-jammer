#!/usr/bin/env python3
"""Tests for the feature ladder.

The point of the ladder is that a failure can be attributed to one layer, and
that only holds if two properties are true: every level really does carry
traffic, and every level really is the one below plus exactly one mechanism.
Both are checked here.

These run real flowgraphs in real time. The sample rate is deliberately low --
see the note in examples/level_ladder.py -- because what these tests need is
real-time headroom, not processing gain.
"""

import time
import unittest

import numpy
import pmt
from gnuradio import blocks, gr

from gpsk_comms.aj_command import (
    HOP_SWITCH_FRACTION,
    PAYLOAD_BITS,
    aj_command_rx,
    aj_command_tx,
    coded_bit_count,
    dwell_index,
    spreading_code,
    sync_bits,
)
from gpsk_comms.hopping import TimedRetuner
from gpsk_comms.levels import (
    FEATURES,
    LEVEL_NAMES,
    features_for_level,
    ladder,
    profile,
)
from gpsk_comms.protocol import monotonic_counter
from gpsk_comms.security import (
    KeyError_,
    generate_master_key,
    preferred_pair_bound,
    public_keys,
)

SAMPLE_RATE = 500_000
SPREADING_FACTOR = 127
DWELL_US = 200_000
REPEAT_RATE = 8.0


def command_message(command, pressed=True):
    message = pmt.dict_add(pmt.make_dict(), pmt.intern("command"), pmt.intern(command))
    return pmt.dict_add(message, pmt.intern("pressed"), pmt.from_bool(pressed))


def delivered(sink):
    return [
        pmt.symbol_to_string(
            pmt.dict_ref(sink.get_message(index), pmt.intern("command"), pmt.PMT_NIL)
        )
        for index in range(sink.num_messages())
    ]


def wait_for(predicate, timeout=10.0, interval=0.02):
    """Poll until ``predicate`` holds, or the timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


class Link:
    """A transmitter and receiver at one level, joined directly."""

    def __init__(self, level, tx_key=None, rx_key=None, rx_level=None, **rx_kwargs):
        self.top = gr.top_block()
        self.tx = aj_command_tx(
            tx_key,
            sample_rate=SAMPLE_RATE,
            repeat_rate=REPEAT_RATE,
            spreading_factor=SPREADING_FACTOR,
            dwell_us=DWELL_US,
            level=level,
        )
        options = {"watchdog_timeout": 30.0}
        options.update(rx_kwargs)
        self.rx = aj_command_rx(
            rx_key if rx_key is not None else tx_key,
            sample_rate=SAMPLE_RATE,
            spreading_factor=SPREADING_FACTOR,
            dwell_us=DWELL_US,
            level=level if rx_level is None else rx_level,
            **options,
        )
        self.commands = blocks.message_debug()
        self.top.connect(self.tx, self.rx)
        self.top.msg_connect((self.rx, "command"), (self.commands, "store"))

    def send(self, command, pressed=True):
        self.tx._source._handle_command(command_message(command, pressed))

    def __enter__(self):
        self.top.start()
        return self

    def __exit__(self, *exception):
        self.top.stop()
        self.top.wait()
        return False


class LadderStructureTest(unittest.TestCase):
    def test_every_level_is_a_superset_of_the_one_below(self):
        """The property the whole ladder rests on.

        If a level ever switched something off that a lower one had, then a
        failure at that level would no longer implicate the single mechanism it
        adds, and bisecting by level would stop being sound.
        """
        for lower, upper in zip(ladder(), ladder()[1:]):
            for feature in FEATURES:
                if getattr(lower, feature):
                    self.assertTrue(
                        getattr(upper, feature),
                        f"level {upper.level} dropped {feature!r} kept by {lower.level}",
                    )

    def test_each_level_adds_exactly_one_mechanism(self):
        for lower, upper in zip(ladder(), ladder()[1:]):
            added = set(upper.enabled) - set(lower.enabled)
            self.assertEqual(len(added), 1, f"level {upper.level} added {added}")

    def test_level_zero_enables_nothing(self):
        self.assertEqual(profile(0).enabled, ())

    def test_top_level_enables_everything(self):
        self.assertEqual(set(profile(len(LEVEL_NAMES) - 1).enabled), set(FEATURES))

    def test_levels_are_reachable_by_name_and_number(self):
        for number, name in enumerate(LEVEL_NAMES):
            self.assertEqual(profile(name), profile(number))

    def test_unknown_level_is_rejected(self):
        for bad in ("nonsense", -1, len(LEVEL_NAMES)):
            with self.assertRaises(ValueError):
                profile(bad)

    def test_excision_without_spreading_is_rejected(self):
        """Excision would delete an unspread signal, so this must not build."""
        with self.assertRaises(ValueError):
            profile(4, dsss=False)

    def test_hopping_without_spreading_is_rejected(self):
        with self.assertRaises(ValueError):
            profile(5, dsss=False)

    def test_a_feature_may_be_switched_off_for_bisection(self):
        bisected = profile(5, excision=False)
        self.assertFalse(bisected.excision)
        self.assertTrue(bisected.hopping and bisected.dsss)

    def test_none_override_leaves_the_level_alone(self):
        self.assertEqual(profile(3, fec=None, auth=None), profile(3))

    def test_only_the_top_two_levels_are_unkeyed(self):
        keyed = [link.level for link in ladder() if link.needs_key]
        self.assertEqual(keyed, [2, 3, 4, 5, 6])


class BurstLayoutTest(unittest.TestCase):
    """The waveform must change one thing at a time as the ladder is climbed."""

    def test_only_coding_changes_the_number_of_bits_on_the_air(self):
        uncoded = {
            level.level: coded_bit_count(level)
            for level in ladder()
            if not level.fec
        }
        coded = {level.level: coded_bit_count(level) for level in ladder() if level.fec}
        self.assertEqual(set(uncoded.values()), {PAYLOAD_BITS})
        self.assertEqual(len(set(coded.values())), 1)
        self.assertGreater(next(iter(coded.values())), PAYLOAD_BITS)

    def test_spreading_only_changes_the_chips_not_their_number(self):
        """Levels below 3 send the same burst length, just without a code.

        Keeping the spreading factor as the pulse length when spreading is off
        means the symbol rate, the burst length and every timing constant are
        the same either side of level 3, so a level-3 failure implicates the
        code and not a reshuffled waveform.
        """
        keys = public_keys()
        unspread = spreading_code(keys, profile(2), 0, SPREADING_FACTOR)
        spread = spreading_code(keys, profile(3), 0, SPREADING_FACTOR)
        self.assertEqual(len(unspread), len(spread))
        self.assertEqual(set(numpy.unique(unspread)), {1.0})
        self.assertEqual(set(numpy.unique(spread)), {-1.0, 1.0})

    def test_sync_word_is_fixed_until_hopping_turns_it_over(self):
        keys = public_keys()
        below = profile(4)
        self.assertTrue(
            numpy.array_equal(
                sync_bits(keys, below, 0, 0), sync_bits(keys, below, 7, 3)
            ),
            "sync word must not depend on the hop before hopping is enabled",
        )
        above = profile(5)
        self.assertFalse(
            numpy.array_equal(
                sync_bits(keys, above, 0, 0), sync_bits(keys, above, 7, 3)
            ),
            "sync word must rotate per hop once hopping is enabled",
        )


class KeyRequirementTest(unittest.TestCase):
    def test_unkeyed_levels_build_without_a_key(self):
        for level in (0, 1):
            transmitter = aj_command_tx(
                None,
                sample_rate=SAMPLE_RATE,
                repeat_rate=REPEAT_RATE,
                spreading_factor=SPREADING_FACTOR,
                level=level,
            )
            self.assertEqual(transmitter.level, level)

    def test_keyed_levels_refuse_to_build_without_a_key(self):
        for level in range(2, len(LEVEL_NAMES)):
            for factory in (aj_command_tx, aj_command_rx):
                with self.assertRaises(ValueError, msg=f"level {level}"):
                    factory(
                        None,
                        sample_rate=SAMPLE_RATE,
                        spreading_factor=SPREADING_FACTOR,
                        level=level,
                    )

    def test_keyed_levels_still_refuse_the_all_zero_key(self):
        with self.assertRaises(KeyError_):
            aj_command_tx(
                bytes(32),
                sample_rate=SAMPLE_RATE,
                repeat_rate=REPEAT_RATE,
                spreading_factor=SPREADING_FACTOR,
                level=2,
            )


class EveryLevelCarriesTrafficTest(unittest.TestCase):
    """The core claim: each rung of the ladder actually works.

    One test method per level rather than a loop, so a failure names the level
    directly instead of stopping the sweep at the first bad one.
    """

    def _carry(self, level):
        link_profile = profile(level)
        key = generate_master_key() if link_profile.needs_key else None
        with Link(level, tx_key=key) as link:
            link.send("forward")
            arrived = wait_for(lambda: "forward" in delivered(link.commands), timeout=15.0)
        self.assertTrue(
            arrived,
            f"level {level} ({LEVEL_NAMES[level]}) delivered nothing; "
            f"despreader={link.rx.despread_counts} decoder={link.rx.counts} "
            f"tx_bursts={link.tx.bursts} reanchors={link.tx.reanchors}",
        )
        self.assertGreater(link.rx.counts["valid"], 0)

    def test_level_0_basic(self):
        self._carry(0)

    def test_level_1_fec(self):
        self._carry(1)

    def test_level_2_auth(self):
        self._carry(2)

    def test_level_3_dsss(self):
        self._carry(3)

    def test_level_4_excision(self):
        self._carry(4)

    def test_level_5_hop(self):
        self._carry(5)

    def test_level_6_escape(self):
        self._carry(6)


class HoppingTest(unittest.TestCase):
    def test_the_receiver_follows_the_hop_across_several_dwells(self):
        """Regression guard for the bug that made hopping useless.

        The receiver used to derive its spreading code and sync word once, at
        construction, while the transmitter re-derived them every burst. The two
        agreed for exactly one dwell and then diverged for good, so every frame
        after the first dwell failed its MAC -- a hundred per cent authentication
        failure rate on a noiseless loopback.

        One burst is sent per dwell, so requiring frames from several dwells is
        precisely a requirement that the receiver retuned along with the
        transmitter rather than sitting on its first code.
        """
        key = generate_master_key()
        with Link(5, tx_key=key) as link:
            link.send("forward")
            followed = wait_for(lambda: link.rx.counts["valid"] >= 4, timeout=25.0)
            counts = link.rx.counts
        self.assertTrue(
            followed,
            f"receiver stopped following the hop schedule: {counts}",
        )
        # Some loss around a retune is tolerable; wholesale failure is not.
        self.assertGreater(counts["valid"], counts["auth_failure"])

    def test_the_schedule_visits_more_than_one_slot(self):
        """A 'hop' that never moves would pass the test above trivially."""
        key = generate_master_key()
        transmitter = aj_command_tx(
            key,
            sample_rate=SAMPLE_RATE,
            repeat_rate=REPEAT_RATE,
            spreading_factor=SPREADING_FACTOR,
            dwell_us=DWELL_US,
            level=5,
        )
        start = dwell_index(monotonic_counter(), DWELL_US)
        slots = set()
        for index in range(start, start + 16):
            epoch, position = divmod(index, 1 << 16)
            slots.add(transmitter.hop_plan.slot_at(epoch, position))
        self.assertGreater(len(slots), 1)

    def test_a_burst_that_cannot_fit_a_dwell_is_refused(self):
        """Better a clear error at build time than a link that never decodes."""
        with self.assertRaises(ValueError):
            aj_command_tx(
                generate_master_key(),
                sample_rate=SAMPLE_RATE,
                repeat_rate=REPEAT_RATE,
                spreading_factor=SPREADING_FACTOR,
                dwell_us=1_000,
                level=5,
            )

    def test_a_search_window_longer_than_a_dwell_is_refused(self):
        with self.assertRaises(ValueError):
            aj_command_rx(
                generate_master_key(),
                sample_rate=SAMPLE_RATE,
                spreading_factor=SPREADING_FACTOR,
                dwell_us=20_000,
                level=5,
            )

    def test_the_receiver_switches_code_inside_the_dwell_not_on_its_edge(self):
        self.assertGreater(HOP_SWITCH_FRACTION, 0.5)
        self.assertLess(HOP_SWITCH_FRACTION, 1.0)


class BandEscapeTest(unittest.TestCase):
    def test_both_ends_derive_the_same_lo_schedule(self):
        """There is no reverse channel, so agreement has to be derived, not
        negotiated. Both ends compute the LO from the same key and the same
        dwell index; if they ever disagreed the link would fail on a real radio
        and pass in loopback, where no retune actually happens.
        """
        key = generate_master_key()
        tx_retuner, rx_retuner = TimedRetuner(), TimedRetuner()
        transmitter = aj_command_tx(
            key,
            sample_rate=SAMPLE_RATE,
            repeat_rate=REPEAT_RATE,
            spreading_factor=SPREADING_FACTOR,
            dwell_us=DWELL_US,
            level=6,
            retuner=tx_retuner,
        )
        receiver = aj_command_rx(
            key,
            sample_rate=SAMPLE_RATE,
            spreading_factor=SPREADING_FACTOR,
            dwell_us=DWELL_US,
            level=6,
            retuner=rx_retuner,
        )
        start = dwell_index(monotonic_counter(), DWELL_US)
        for index in range(start, start + 12):
            transmitter._retune(index)
            receiver._tune_to(index)
        transmitted = [frequency for frequency, _ in tx_retuner.history[-12:]]
        received = [frequency for frequency, _ in rx_retuner.history[-12:]]
        self.assertEqual(transmitted, received)
        self.assertGreater(len(set(transmitted)), 1, "the LO never moved")

    def test_lower_levels_do_not_touch_the_lo(self):
        key = generate_master_key()
        retuner = TimedRetuner()
        aj_command_rx(
            key,
            sample_rate=SAMPLE_RATE,
            spreading_factor=SPREADING_FACTOR,
            dwell_us=DWELL_US,
            level=5,
            retuner=retuner,
        )
        self.assertEqual(retuner.history, [])


class LevelAgreementTest(unittest.TestCase):
    """Both ends must be set to the same level.

    Most mismatches fail outright, and those are checked here directly. The
    spreading mismatch is the exception and is checked by measurement instead --
    see :meth:`test_a_spreading_mismatch_throws_away_the_processing_gain`.
    """

    def test_a_coding_mismatch_decodes_nothing(self):
        key = generate_master_key()
        with Link(1, tx_key=key, rx_level=0) as link:
            link.send("forward")
            time.sleep(3.0)
            counts = link.rx.counts
        self.assertEqual(counts["valid"], 0)
        self.assertEqual(delivered(link.commands), [])

    def test_an_authentication_mismatch_decodes_nothing(self):
        """A level-1 receiver checks the frame tag against the published key,
        so a level-2 transmitter's tags never verify."""
        key = generate_master_key()
        with Link(2, tx_key=key, rx_level=1) as link:
            link.send("forward")
            time.sleep(3.0)
            counts = link.rx.counts
        self.assertEqual(counts["valid"], 0)
        self.assertEqual(delivered(link.commands), [])

    def test_a_spreading_mismatch_throws_away_the_processing_gain(self):
        """A spreading mismatch is degradation, not a clean failure.

        An unspread receiver correlates against a constant, and that is not
        orthogonal to a Gold code. Correlating against a constant just sums the
        code, and a Gold sequence's imbalance is one of -1, -t(n) or t(n)-2 --
        bounded by the same t(n) that bounds its cross-correlation, but not
        anywhere near zero. So the correlation peak collapses from N to at most
        t(n), which on a noiseless loopback is still enough to carry a frame: a
        mismatched pair can look like it is working.

        It would not survive a real channel, having given up essentially all of
        its processing gain, which is why this is measured as a gain rather than
        asserted as a delivery count. The bound is taken from theory rather than
        from a measured constant so that changing the code length cannot quietly
        turn this into a test of nothing.
        """
        keys = public_keys()
        code = spreading_code(keys, profile(3), 0, SPREADING_FACTOR)
        constant = spreading_code(keys, profile(2), 0, SPREADING_FACTOR)
        degree = int(SPREADING_FACTOR).bit_length()
        bound = preferred_pair_bound(degree)

        matched = abs(float(numpy.dot(code, code)))
        mismatched = abs(float(numpy.dot(code, constant)))
        self.assertEqual(matched, float(SPREADING_FACTOR))
        self.assertLessEqual(mismatched, bound, "code imbalance exceeded the Gold bound")

        loss_db = 20 * numpy.log10(matched / max(mismatched, 1e-12))
        worst_case_db = 20 * numpy.log10(SPREADING_FACTOR / bound)
        self.assertGreaterEqual(
            loss_db, worst_case_db, f"only {loss_db:.1f} dB of gain was at stake"
        )


class AuthenticationBoundaryTest(unittest.TestCase):
    def test_a_wrong_key_delivers_nothing_from_level_2_up(self):
        with Link(2, tx_key=generate_master_key(), rx_key=generate_master_key()) as link:
            link.send("forward")
            rejected = wait_for(lambda: link.rx.counts["auth_failure"] > 2, timeout=15.0)
        self.assertTrue(rejected, "the receiver never even saw the forged traffic")
        self.assertEqual(link.rx.counts["valid"], 0)
        self.assertEqual(delivered(link.commands), [])

    def test_replay_rejection_is_off_below_level_2(self):
        """Replay rejection without authentication is theatre, and it would add
        a failure mode to the rungs that exist to have as few as possible."""
        self.assertIsNone(
            aj_command_rx(
                None,
                sample_rate=SAMPLE_RATE,
                spreading_factor=SPREADING_FACTOR,
                level=1,
            )._decoder._replay
        )
        self.assertIsNotNone(
            aj_command_rx(
                generate_master_key(),
                sample_rate=SAMPLE_RATE,
                spreading_factor=SPREADING_FACTOR,
                level=2,
            )._decoder._replay
        )


class BackwardCompatibilityTest(unittest.TestCase):
    """The pre-ladder keyword arguments must keep working."""

    def test_hopping_enabled_still_switches_hopping_off(self):
        receiver = aj_command_rx(
            generate_master_key(),
            sample_rate=SAMPLE_RATE,
            spreading_factor=SPREADING_FACTOR,
            hopping_enabled=False,
        )
        self.assertFalse(receiver.profile.hopping)
        self.assertTrue(receiver.profile.dsss)

    def test_excision_enabled_still_switches_excision_off(self):
        receiver = aj_command_rx(
            generate_master_key(),
            sample_rate=SAMPLE_RATE,
            spreading_factor=SPREADING_FACTOR,
            dwell_us=DWELL_US,
            excision_enabled=False,
        )
        self.assertFalse(receiver.profile.excision)

    def test_the_default_level_is_everything_but_band_escape(self):
        expected = features_for_level(5)
        transmitter = aj_command_tx(
            generate_master_key(),
            sample_rate=SAMPLE_RATE,
            repeat_rate=REPEAT_RATE,
            spreading_factor=SPREADING_FACTOR,
        )
        for feature, enabled in expected.items():
            self.assertEqual(getattr(transmitter.profile, feature), enabled, feature)


if __name__ == "__main__":
    unittest.main()
