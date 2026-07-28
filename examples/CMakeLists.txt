install(
    FILES
        gmsk_command_loopback.grc
        aj_command_loopback.grc
        gmsk_command_tx_uhd.grc
        gmsk_keyboard_tx_uhd.grc
        gmsk_command_rx_uhd.grc
    DESTINATION ${CMAKE_INSTALL_DATADIR}/gnuradio/examples/gpsk_comms
)

install(
    PROGRAMS ota_smoke_test.py keyboard_tx.py jammer_harness.py
    DESTINATION ${CMAKE_INSTALL_DATADIR}/gnuradio/examples/gpsk_comms
)
