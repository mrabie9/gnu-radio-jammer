"""Qt keyboard command source for GNU Radio Companion flowgraphs."""

import pmt
from gnuradio import gr
from PyQt5 import Qt


class keyboard_command_source(gr.sync_block, Qt.QFrame):
    """Publish press/release command dictionaries from W/A/S/D keyboard events."""

    _KEY_COMMANDS = {
        Qt.Qt.Key_W: "forward",
        Qt.Qt.Key_S: "backward",
        Qt.Qt.Key_A: "left",
        Qt.Qt.Key_D: "right",
        Qt.Qt.Key_Space: "stop",
    }

    def __init__(self, release_debounce_ms=120, command_refresh_ms=100):
        gr.sync_block.__init__(self, name="keyboard command source", in_sig=None, out_sig=None)
        Qt.QFrame.__init__(self)

        application = Qt.QApplication.instance()
        if application is None:
            raise RuntimeError(
                "Keyboard Command Source requires a QT GUI GNU Radio flowgraph"
            )

        self._command_port = pmt.intern("command")
        self.message_port_register_out(self._command_port)
        self._held_tokens = []
        self._token_commands = {}
        self._active_token = None
        self._release_debounce_ms = max(0, int(release_debounce_ms))
        self._release_timers = {}
        self._command_refresh_ms = max(0, int(command_refresh_ms))

        self.setFrameShape(Qt.QFrame.StyledPanel)
        self.setFocusPolicy(Qt.Qt.StrongFocus)
        self.setMinimumSize(500, 300)
        self._build_panel()

        self._command_refresh_timer = Qt.QTimer(self)
        self._command_refresh_timer.setSingleShot(False)
        self._command_refresh_timer.timeout.connect(self._refresh_active_command)

        self._application = application
        self._application.installEventFilter(self)

    def _build_panel(self):
        layout = Qt.QVBoxLayout(self)

        title = Qt.QLabel("Hold W / A / S / D to transmit")
        title.setAlignment(Qt.Qt.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title)

        help_text = Qt.QLabel(
            "W = forward    S = backward    A = left    D = right\n"
            "Space = stop    Release all controls = no command packets\n"
            "Held command refresh: 10 updates/second"
        )
        help_text.setAlignment(Qt.Qt.AlignCenter)
        help_text.setStyleSheet("font-size: 13px;")
        layout.addWidget(help_text)

        button_grid = Qt.QGridLayout()
        button_specs = (
            ("W\nForward", "forward", 0, 1),
            ("A\nLeft", "left", 1, 0),
            ("S\nBackward", "backward", 1, 1),
            ("D\nRight", "right", 1, 2),
            ("Space\nStop", "stop", 2, 0, 1, 3),
        )
        for spec in button_specs:
            label, command, row, column, *span = spec
            row_span, column_span = span if span else (1, 1)
            button = Qt.QPushButton(label)
            button.setMinimumHeight(48)
            token = f"button:{command}"
            button.pressed.connect(
                lambda item=token, value=command: self._press(item, value)
            )
            button.released.connect(lambda item=token: self._release(item))
            button_grid.addWidget(button, row, column, row_span, column_span)
        layout.addLayout(button_grid)

        self._status = Qt.QLabel()
        self._status.setAlignment(Qt.Qt.AlignCenter)
        layout.addWidget(self._status)
        self._show_idle()

    def _publish(self, command, pressed):
        message = pmt.make_dict()
        message = pmt.dict_add(message, pmt.intern("command"), pmt.intern(command))
        message = pmt.dict_add(message, pmt.intern("pressed"), pmt.from_bool(pressed))
        self.message_port_pub(self._command_port, message)

    def _show_active(self, command):
        self._status.setText(f"SENDING: {command.upper()}")
        self._status.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #137333;"
        )

    def _show_idle(self):
        self._status.setText("IDLE — no command packets are being sent")
        self._status.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #9b1c1c;"
        )

    def _press(self, token, command):
        if token in self._held_tokens:
            return
        self._held_tokens.append(token)
        self._token_commands[token] = command
        self._active_token = token
        self._publish(command, True)
        self._show_active(command)
        if self._command_refresh_ms:
            self._command_refresh_timer.start(self._command_refresh_ms)

    def _refresh_active_command(self):
        if self._active_token is None:
            self._command_refresh_timer.stop()
            return
        command = self._token_commands[self._active_token]
        # This deliberately mirrors the effective behavior of the working
        # Tkinter keyboard during OS key repeat. Releasing and immediately
        # re-pressing re-arms the unchanged TX block, increments its sequence,
        # and lets the unchanged RX publish another command for controllers
        # that require continuous updates while a control is held.
        self._publish(command, False)
        self._publish(command, True)

    def _release(self, token):
        if token not in self._held_tokens:
            return
        self._held_tokens.remove(token)
        self._token_commands.pop(token, None)
        if token != self._active_token:
            return
        if self._held_tokens:
            self._active_token = self._held_tokens[-1]
            command = self._token_commands[self._active_token]
            self._publish(command, True)
            self._show_active(command)
        else:
            self._publish("stop", False)
            self._active_token = None
            self._command_refresh_timer.stop()
            self._show_idle()

    def _cancel_pending_release(self, token):
        timer = self._release_timers.pop(token, None)
        if timer is not None:
            timer.stop()
            timer.deleteLater()

    def _finish_key_release(self, token):
        timer = self._release_timers.pop(token, None)
        if timer is not None:
            timer.deleteLater()
        self._release(token)

    def _schedule_key_release(self, token):
        self._cancel_pending_release(token)
        if self._release_debounce_ms == 0:
            self._release(token)
            return
        timer = Qt.QTimer(self)
        timer.setSingleShot(True)
        timer.timeout.connect(lambda item=token: self._finish_key_release(item))
        self._release_timers[token] = timer
        timer.start(self._release_debounce_ms)

    def _release_all(self):
        self._command_refresh_timer.stop()
        for token in list(self._release_timers):
            self._cancel_pending_release(token)
        if self._active_token is not None:
            self._publish("stop", False)
        self._held_tokens.clear()
        self._token_commands.clear()
        self._active_token = None
        self._show_idle()

    def eventFilter(self, watched, event):
        event_type = event.type()
        if event_type == Qt.QEvent.KeyPress:
            key = event.key()
            if key == Qt.Qt.Key_Escape:
                self._release_all()
                Qt.QTimer.singleShot(0, self.window().close)
                return True
            command = self._KEY_COMMANDS.get(key)
            if command is not None:
                token = f"key:{key}"
                # Windows may report the release/press pair produced by key
                # repeat as ordinary events. Any repeat press cancels the
                # short pending release, while a real release has no matching
                # press and is applied after the debounce interval.
                self._cancel_pending_release(token)
                if not event.isAutoRepeat():
                    self._press(token, command)
                return True
        elif event_type == Qt.QEvent.KeyRelease:
            key = event.key()
            if key in self._KEY_COMMANDS:
                self._schedule_key_release(f"key:{key}")
                return True
        elif event_type == Qt.QEvent.ApplicationDeactivate:
            self._release_all()
        return False

    def closeEvent(self, event):
        self._release_all()
        super().closeEvent(event)
