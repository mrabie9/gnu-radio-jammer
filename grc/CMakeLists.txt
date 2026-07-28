install(
    FILES
        gpsk_comms_gmsk_command_tx.block.yml
        gpsk_comms_gmsk_command_rx.block.yml
        gpsk_comms_aj_command_tx.block.yml
        gpsk_comms_aj_command_rx.block.yml
        gpsk_comms_keyboard_command_source.block.yml
    DESTINATION ${CMAKE_INSTALL_DATADIR}/gnuradio/grc/blocks
)
