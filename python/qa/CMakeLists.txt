find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_test(
    NAME qa_protocol
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/qa_protocol.py
)
add_test(
    NAME qa_gmsk_loopback
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/qa_gmsk_loopback.py
)
add_test(
    NAME qa_keyboard_command_source
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/qa_keyboard_command_source.py
)

set(_qa_pythonpath "${CMAKE_CURRENT_SOURCE_DIR}/..")
set_tests_properties(qa_protocol qa_gmsk_loopback PROPERTIES
    ENVIRONMENT "PYTHONPATH=${_qa_pythonpath}"
)
set_tests_properties(qa_keyboard_command_source PROPERTIES
    ENVIRONMENT "PYTHONPATH=${_qa_pythonpath};QT_QPA_PLATFORM=offscreen"
)
