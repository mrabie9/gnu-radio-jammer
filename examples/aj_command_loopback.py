#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: AJ Command Loopback
# Author: gpsk_comms
# Description: Anti-jam command link loopback with a switchable jammer, no USRP hardware
# GNU Radio version: 3.10.10.0

from gnuradio import analog
from gnuradio import blocks
import pmt
from gnuradio import blocks, gr
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gpsk_comms import aj_command_rx
from gpsk_comms.security import load_master_key
from gpsk_comms import aj_command_tx
from gpsk_comms.security import generate_master_key, load_master_key




class aj_command_loopback(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "AJ Command Loopback", catch_exceptions=True)

        ##################################################
        # Variables
        ##################################################
        self.spreading_factor = spreading_factor = 255
        self.samp_rate = samp_rate = 2000000
        self.master_key = master_key = generate_master_key()
        self.jam_amplitude = jam_amplitude = 0.0

        ##################################################
        # Blocks
        ##################################################

        self.tx = aj_command_tx(
            master_key=master_key,
            sample_rate=samp_rate,
            repeat_rate=20,
            spreading_factor=spreading_factor,
            interleave_depth=17,
            dwell_us=100000,
            hopping_enabled=False,
            hold_to_send=False)
        self.throttle = blocks.throttle( gr.sizeof_gr_complex*1, samp_rate, True, 0 if "auto" == "auto" else max( int(float(0.1) * samp_rate) if "auto" == "time" else int(0.1), 1) )
        self.rx = aj_command_rx(
            master_key=master_key,
            sample_rate=samp_rate,
            spreading_factor=spreading_factor,
            interleave_depth=17,
            watchdog_timeout=1.0,
            grace_period=0.0,
            dwell_us=100000,
            hopping_enabled=False,
            excision_enabled=True,
            detection_threshold=0.25)
        self.jammer = analog.sig_source_c(samp_rate, analog.GR_COS_WAVE, (samp_rate*0.13), jam_amplitude, 0, 0)
        self.command_strobe = blocks.message_strobe(pmt.dict_add(pmt.make_dict(), pmt.intern("command"), pmt.intern("forward")), 1000)
        self.command_debug = blocks.message_debug(True, gr.log_levels.info)
        self.combiner = blocks.add_vcc(1)


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.command_strobe, 'strobe'), (self.tx, 'command'))
        self.msg_connect((self.rx, 'command'), (self.command_debug, 'print'))
        self.connect((self.combiner, 0), (self.throttle, 0))
        self.connect((self.jammer, 0), (self.combiner, 1))
        self.connect((self.throttle, 0), (self.rx, 0))
        self.connect((self.tx, 0), (self.combiner, 0))


    def get_spreading_factor(self):
        return self.spreading_factor

    def set_spreading_factor(self, spreading_factor):
        self.spreading_factor = spreading_factor

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.jammer.set_sampling_freq(self.samp_rate)
        self.jammer.set_frequency((self.samp_rate*0.13))
        self.throttle.set_sample_rate(self.samp_rate)

    def get_master_key(self):
        return self.master_key

    def set_master_key(self, master_key):
        self.master_key = master_key

    def get_jam_amplitude(self):
        return self.jam_amplitude

    def set_jam_amplitude(self, jam_amplitude):
        self.jam_amplitude = jam_amplitude
        self.jammer.set_amplitude(self.jam_amplitude)




def main(top_block_cls=aj_command_loopback, options=None):
    tb = top_block_cls()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()

    try:
        input('Press Enter to quit: ')
    except EOFError:
        pass
    tb.stop()
    tb.wait()


if __name__ == '__main__':
    main()
