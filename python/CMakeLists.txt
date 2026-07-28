install(
    FILES
        gpsk_comms/__init__.py
        gpsk_comms/protocol.py
        gpsk_comms/gmsk_command_tx.py
        gpsk_comms/gmsk_command_rx.py
        gpsk_comms/keyboard_command_source.py
    DESTINATION ${GR_PYTHON_DIR}/gpsk_comms
)

if(ENABLE_TESTING)
    add_subdirectory(qa)
endif()
