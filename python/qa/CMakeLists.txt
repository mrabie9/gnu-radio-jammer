find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(_qa_pythonpath "${CMAKE_CURRENT_SOURCE_DIR}/..")

# Tests that need only PYTHONPATH.
set(_qa_plain
    qa_protocol
    qa_protocol_v2
    qa_security
    qa_dsss
    qa_fec
    qa_excision
    qa_hopping
    qa_gmsk_loopback
    qa_aj_command
    qa_levels
)

foreach(_test ${_qa_plain})
    add_test(
        NAME ${_test}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${_test}.py
    )
    set_tests_properties(${_test} PROPERTIES ENVIRONMENT "PYTHONPATH=${_qa_pythonpath}")
endforeach()

# These run several real-time flowgraphs end to end, so they need more than the
# default CTest timeout allows. qa_levels runs one per rung of the ladder.
set_tests_properties(qa_aj_command PROPERTIES TIMEOUT 300)
set_tests_properties(qa_levels PROPERTIES TIMEOUT 600)

# The keyboard source needs a Qt platform plugin even when headless.
add_test(
    NAME qa_keyboard_command_source
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/qa_keyboard_command_source.py
)
set_tests_properties(qa_keyboard_command_source PROPERTIES
    ENVIRONMENT "PYTHONPATH=${_qa_pythonpath};QT_QPA_PLATFORM=offscreen"
)
