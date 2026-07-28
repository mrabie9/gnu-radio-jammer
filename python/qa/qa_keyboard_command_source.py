#!/usr/bin/env python3

import unittest

import pmt
from gnuradio import blocks, gr
from PyQt5 import Qt, QtTest

from gpsk_comms.keyboard_command_source import keyboard_command_source


def _field(message, name):
    return pmt.dict_ref(message, pmt.intern(name), pmt.PMT_NIL)


class KeyboardCommandSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = Qt.QApplication.instance() or Qt.QApplication([])

    def setUp(self):
        self.top_block = gr.top_block()
        # State-transition tests disable periodic command refresh so their
        # expected message lists remain deterministic.
        self.source = keyboard_command_source(command_refresh_ms=0)
        self.debug = blocks.message_debug()
        self.top_block.msg_connect((self.source, "command"), (self.debug, "store"))
        self.top_block.start()

    def tearDown(self):
        self.source._release_all()
        self.top_block.stop()
        self.top_block.wait()
        self.source.deleteLater()
        self.application.processEvents()

    def _send_key(self, event_type, key):
        event = Qt.QKeyEvent(event_type, key, Qt.Qt.NoModifier)
        self.application.sendEvent(self.source, event)
        self.application.processEvents()

    def _messages(self):
        return [
            (
                pmt.symbol_to_string(_field(self.debug.get_message(index), "command")),
                pmt.to_bool(_field(self.debug.get_message(index), "pressed")),
            )
            for index in range(self.debug.num_messages())
        ]

    def test_press_release_and_last_held_key(self):
        self._send_key(Qt.QEvent.KeyPress, Qt.Qt.Key_A)
        self._send_key(Qt.QEvent.KeyPress, Qt.Qt.Key_D)
        self._send_key(Qt.QEvent.KeyRelease, Qt.Qt.Key_D)
        self._send_key(Qt.QEvent.KeyRelease, Qt.Qt.Key_A)
        QtTest.QTest.qWait(180)

        self.assertEqual(
            self._messages(),
            [
                ("left", True),
                ("right", True),
                ("left", True),
                ("stop", False),
            ],
        )
        self.assertEqual(
            self.source._status.text(),
            "IDLE — no command packets are being sent",
        )

    def test_auto_repeat_is_ignored(self):
        press = Qt.QKeyEvent(
            Qt.QEvent.KeyPress,
            Qt.Qt.Key_W,
            Qt.Qt.NoModifier,
            "w",
            True,
            2,
        )
        self.application.sendEvent(self.source, press)
        self.application.processEvents()
        self.assertEqual(self.debug.num_messages(), 0)

    def test_window_deactivate_does_not_cancel_held_key(self):
        self._send_key(Qt.QEvent.KeyPress, Qt.Qt.Key_W)
        self.application.sendEvent(
            self.source,
            Qt.QEvent(Qt.QEvent.WindowDeactivate),
        )
        self.application.processEvents()

        self.assertEqual(self.source._active_token, f"key:{Qt.Qt.Key_W}")
        self.assertEqual(self._messages(), [("forward", True)])

        self._send_key(Qt.QEvent.KeyRelease, Qt.Qt.Key_W)
        QtTest.QTest.qWait(180)
        self.assertEqual(self._messages()[-1], ("stop", False))

    def test_windows_repeat_pair_does_not_release_held_key(self):
        self._send_key(Qt.QEvent.KeyPress, Qt.Qt.Key_W)

        # Some Windows/Qt combinations do not mark the synthetic repeat
        # release/press pair as auto-repeat. The subsequent press must cancel
        # the pending release instead of briefly stopping the transmitter.
        self._send_key(Qt.QEvent.KeyRelease, Qt.Qt.Key_W)
        QtTest.QTest.qWait(30)
        self._send_key(Qt.QEvent.KeyPress, Qt.Qt.Key_W)
        QtTest.QTest.qWait(180)

        self.assertEqual(self.source._active_token, f"key:{Qt.Qt.Key_W}")
        self.assertEqual(self._messages(), [("forward", True)])

        self._send_key(Qt.QEvent.KeyRelease, Qt.Qt.Key_W)
        QtTest.QTest.qWait(180)
        self.assertEqual(self._messages()[-1], ("stop", False))

    def test_held_key_periodically_rearms_command_until_release(self):
        self.source._command_refresh_ms = 40
        self._send_key(Qt.QEvent.KeyPress, Qt.Qt.Key_W)
        QtTest.QTest.qWait(110)

        messages = self._messages()
        self.assertEqual(messages[0], ("forward", True))
        self.assertGreaterEqual(messages.count(("forward", False)), 2)
        self.assertGreaterEqual(messages.count(("forward", True)), 3)
        self.assertEqual(self.source._status.text(), "SENDING: FORWARD")

        self._send_key(Qt.QEvent.KeyRelease, Qt.Qt.Key_W)
        QtTest.QTest.qWait(180)
        self.assertEqual(self._messages()[-1], ("stop", False))
        message_count = self.debug.num_messages()
        QtTest.QTest.qWait(100)
        self.assertEqual(self.debug.num_messages(), message_count)


if __name__ == "__main__":
    unittest.main()
