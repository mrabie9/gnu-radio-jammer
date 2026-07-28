#!/usr/bin/env python3
"""Control the UHD GMSK transmitter from press/release keyboard events."""

import signal
import tkinter as tk

import pmt
from gnuradio import gr, uhd

from gpsk_comms import gmsk_command_tx


SAMPLE_RATE = 1_000_000
CENTER_FREQUENCY = 2_440_000_000
TX_GAIN = 80
DEVICE_ADDRESS = "serial=337D3DF"
ANTENNA = "TX/RX"

KEY_COMMANDS = {
    "w": "forward",
    "s": "backward",
    "a": "left",
    "d": "right",
    "space": "stop",
}


class KeyboardMessageSource(gr.basic_block):
    """Message-only block used to safely publish GUI key events."""

    def __init__(self):
        super().__init__(name="keyboard command source", in_sig=None, out_sig=None)
        self._command_port = pmt.intern("command")
        self.message_port_register_out(self._command_port)

    def set_key_state(self, command, pressed):
        message = pmt.make_dict()
        message = pmt.dict_add(message, pmt.intern("command"), pmt.intern(command))
        message = pmt.dict_add(message, pmt.intern("pressed"), pmt.from_bool(pressed))
        self.message_port_pub(self._command_port, message)


class KeyboardTxFlowgraph(gr.top_block):
    def __init__(self):
        super().__init__("GMSK keyboard command TX", catch_exceptions=True)
        self.tx = gmsk_command_tx(
            sample_rate=SAMPLE_RATE,
            samples_per_symbol=4,
            bt=0.35,
            repeat_rate=100,
            access_code="D391DA26",
            hold_to_send=True,
        )
        self.keyboard_source = KeyboardMessageSource()
        self.usrp_sink = uhd.usrp_sink(
            ",".join((DEVICE_ADDRESS, "")),
            uhd.stream_args(cpu_format="fc32", channels=[0]),
            "",
        )
        self.usrp_sink.set_clock_source("internal", 0)
        self.usrp_sink.set_time_source("internal", 0)
        self.usrp_sink.set_samp_rate(SAMPLE_RATE)
        self.usrp_sink.set_time_unknown_pps(uhd.time_spec(0))
        self.usrp_sink.set_center_freq(CENTER_FREQUENCY, 0)
        self.usrp_sink.set_antenna(ANTENNA, 0)
        self.usrp_sink.set_bandwidth(SAMPLE_RATE, 0)
        self.usrp_sink.set_gain(TX_GAIN, 0)
        self.connect(self.tx, self.usrp_sink)
        self.msg_connect(
            (self.keyboard_source, "command"),
            (self.tx, "command"),
        )

    def set_key_state(self, command, pressed):
        self.keyboard_source.set_key_state(command, pressed)


class KeyboardWindow:
    def __init__(self, flowgraph):
        self.flowgraph = flowgraph
        self.root = tk.Tk()
        self.root.title("GMSK Command TX")
        self.root.geometry("520x260")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.held_keys = []
        self.active_key = None
        self.closed = False

        tk.Label(
            self.root,
            text="Click here, then hold W / A / S / D",
            font=("Sans", 18, "bold"),
        ).pack(pady=(24, 8))
        tk.Label(
            self.root,
            text="W = forward    S = backward    A = left    D = right\n"
            "Space = stop    Release all keys = no command packets",
            font=("Sans", 12),
        ).pack(pady=8)
        self.status = tk.StringVar(value="IDLE — no command is being sent")
        self.status_label = tk.Label(
            self.root,
            textvariable=self.status,
            font=("Sans", 15, "bold"),
            fg="#9b1c1c",
        )
        self.status_label.pack(pady=22)
        tk.Label(self.root, text="Esc closes the transmitter").pack()

        self.root.bind_all("<KeyPress>", self.key_press)
        self.root.bind_all("<KeyRelease>", self.key_release)
        self.root.bind("<FocusOut>", self.focus_lost)
        self.root.after(100, self.root.focus_force)

    @staticmethod
    def _key_name(event):
        return event.keysym.lower()

    def _activate(self, key):
        command = KEY_COMMANDS[key]
        self.flowgraph.set_key_state(command, True)
        self.active_key = key
        self.status.set(f"SENDING: {command.upper()}")
        self.status_label.configure(fg="#137333")

    def _release_active(self):
        if self.active_key is not None:
            self.flowgraph.set_key_state(KEY_COMMANDS[self.active_key], False)
        self.active_key = None
        self.status.set("IDLE — no command is being sent")
        self.status_label.configure(fg="#9b1c1c")

    def key_press(self, event):
        key = self._key_name(event)
        if key == "escape":
            self.close()
            return
        if key not in KEY_COMMANDS or key in self.held_keys:
            return
        self.held_keys.append(key)
        if key != self.active_key:
            self._activate(key)

    def key_release(self, event):
        key = self._key_name(event)
        if key not in KEY_COMMANDS or key not in self.held_keys:
            return
        self.held_keys.remove(key)
        if key != self.active_key:
            return
        if self.held_keys:
            self._activate(self.held_keys[-1])
        else:
            self._release_active()

    def focus_lost(self, _event=None):
        self.held_keys.clear()
        self._release_active()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.held_keys.clear()
        self._release_active()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    flowgraph = KeyboardTxFlowgraph()
    window = KeyboardWindow(flowgraph)

    def stop_from_signal(_signal_number, _frame):
        window.root.after(0, window.close)

    signal.signal(signal.SIGINT, stop_from_signal)
    signal.signal(signal.SIGTERM, stop_from_signal)
    flowgraph.start()
    try:
        window.run()
    finally:
        flowgraph.stop()
        flowgraph.wait()


if __name__ == "__main__":
    main()
