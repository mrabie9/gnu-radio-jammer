#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: GMSK Keyboard Command TX - UHD
# Author: gpsk_comms
# Description: Keyboard-controlled GMSK command transmitter using a UHD USRP Sink
# GNU Radio version: 3.10.6.0

from packaging.version import Version as StrictVersion
from PyQt5 import Qt
from gnuradio import qtgui
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import uhd
import time
from gpsk_comms import gmsk_command_tx
from gpsk_comms.keyboard_command_source import keyboard_command_source



class gmsk_keyboard_tx_uhd(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "GMSK Keyboard Command TX - UHD", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("GMSK Keyboard Command TX - UHD")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except BaseException as exc:
            print(f"Qt GUI: Could not set Icon: {str(exc)}", file=sys.stderr)
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("GNU Radio", "gmsk_keyboard_tx_uhd")

        try:
            if StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
                self.restoreGeometry(self.settings.value("geometry").toByteArray())
            else:
                self.restoreGeometry(self.settings.value("geometry"))
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 1000000
        self.time_source = time_source = "internal"
        self.gain = gain = 80
        self.device_addr = device_addr = ""
        self.clock_source = clock_source = "internal"
        self.center_freq = center_freq = 2440000000
        self.bandwidth = bandwidth = samp_rate
        self.antenna = antenna = "TX/RX"

        ##################################################
        # Blocks
        ##################################################

        self.usrp_sink = uhd.usrp_sink(
            ",".join((device_addr, '')),
            uhd.stream_args(
                cpu_format="fc32",
                args='',
                channels=list(range(0,1)),
            ),
            "",
        )
        self.usrp_sink.set_clock_source(clock_source, 0)
        self.usrp_sink.set_time_source(time_source, 0)
        self.usrp_sink.set_samp_rate(samp_rate)
        self.usrp_sink.set_time_unknown_pps(uhd.time_spec(0))

        self.usrp_sink.set_center_freq(center_freq, 0)
        self.usrp_sink.set_antenna(antenna, 0)
        self.usrp_sink.set_bandwidth(bandwidth, 0)
        self.usrp_sink.set_gain(gain, 0)
        self.tx = gmsk_command_tx(
            sample_rate=samp_rate,
            samples_per_symbol=4,
            bt=0.35,
            repeat_rate=100,
            access_code='D391DA26',
            command_cycle=("forward", "backward", "right", "left") if False else None,
            cycle_period=0.5,
            hold_to_send=True)
        self.keyboard_source = _keyboard_source_win = keyboard_command_source()
        self.keyboard_source = _keyboard_source_win
        self.top_grid_layout.addWidget(_keyboard_source_win, 0, 0, 1, 1)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 1):
            self.top_grid_layout.setColumnStretch(c, 1)


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.keyboard_source, 'command'), (self.tx, 'command'))
        self.connect((self.tx, 0), (self.usrp_sink, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "gmsk_keyboard_tx_uhd")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.set_bandwidth(self.samp_rate)
        self.usrp_sink.set_samp_rate(self.samp_rate)

    def get_time_source(self):
        return self.time_source

    def set_time_source(self, time_source):
        self.time_source = time_source

    def get_gain(self):
        return self.gain

    def set_gain(self, gain):
        self.gain = gain
        self.usrp_sink.set_gain(self.gain, 0)

    def get_device_addr(self):
        return self.device_addr

    def set_device_addr(self, device_addr):
        self.device_addr = device_addr

    def get_clock_source(self):
        return self.clock_source

    def set_clock_source(self, clock_source):
        self.clock_source = clock_source

    def get_center_freq(self):
        return self.center_freq

    def set_center_freq(self, center_freq):
        self.center_freq = center_freq
        self.usrp_sink.set_center_freq(self.center_freq, 0)

    def get_bandwidth(self):
        return self.bandwidth

    def set_bandwidth(self, bandwidth):
        self.bandwidth = bandwidth
        self.usrp_sink.set_bandwidth(self.bandwidth, 0)

    def get_antenna(self):
        return self.antenna

    def set_antenna(self, antenna):
        self.antenna = antenna
        self.usrp_sink.set_antenna(self.antenna, 0)




def main(top_block_cls=gmsk_keyboard_tx_uhd, options=None):

    if StrictVersion("4.5.0") <= StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
        style = gr.prefs().get_string('qtgui', 'style', 'raster')
        Qt.QApplication.setGraphicsSystem(style)
    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()
