"""Forward error correction and interleaving.

Rate-1/2, constraint-length-7 convolutional coding with soft-decision Viterbi
decoding, plus a block interleaver.

This deliberately does not use ``gnuradio.fec``. Frames here are around 200 bits
and arrive at tens of hertz, well off the per-sample path, so a vectorised numpy
decoder costs microseconds per frame while being far easier to test in isolation
and to feed with the soft values :mod:`gpsk_comms.dsss` already produces. The
GNU Radio blocks would add stream-framing machinery for no benefit at this rate.

Soft decisions matter here: hard-slicing the despread symbols before decoding
throws away roughly 2 dB, which is a significant fraction of what this layer
contributes in the first place.

The interleaver is what makes this layer work against *pulsed* and *swept*
jammers rather than only against white noise. A jammer that erases a contiguous
run of chips produces a contiguous run of bad symbols, and a convolutional code
handles scattered errors far better than bursts. Interleaving converts one into
the other.
"""

import numpy

#: Standard NASA / CCSDS rate-1/2 K=7 generator polynomials, octal 171 and 133.
#: The same pair used by almost every deep-space and 802.11 legacy link, chosen
#: here for the same reason: it is the best-studied short constraint length code
#: and its ~5 dB soft-decision coding gain is well characterised.
POLYNOMIALS = (0o171, 0o133)
CONSTRAINT_LENGTH = 7
TAIL_BITS = CONSTRAINT_LENGTH - 1
STATE_COUNT = 1 << TAIL_BITS
RATE_DENOMINATOR = len(POLYNOMIALS)


def _parity(values):
    """Return the parity of each integer in ``values`` as 0/1."""
    values = numpy.asarray(values, dtype=numpy.int64)
    result = numpy.zeros_like(values)
    for shift in range(CONSTRAINT_LENGTH):
        result ^= (values >> shift) & 1
    return result


def _build_trellis():
    """Precompute next state and output symbols for every (state, input) pair.

    Outputs are stored as +/-1 floats so the decoder's branch metric is a plain
    dot product against the received soft values, with no per-branch branching.
    """
    states = numpy.arange(STATE_COUNT, dtype=numpy.int64)
    next_state = numpy.empty((STATE_COUNT, 2), dtype=numpy.int64)
    outputs = numpy.empty((STATE_COUNT, 2, RATE_DENOMINATOR), dtype=numpy.float32)
    for bit in (0, 1):
        register = (states << 1) | bit
        next_state[:, bit] = register & (STATE_COUNT - 1)
        for index, polynomial in enumerate(POLYNOMIALS):
            outputs[:, bit, index] = 1.0 - 2.0 * _parity(register & polynomial)
    return next_state, outputs


_NEXT_STATE, _OUTPUTS = _build_trellis()

# For the decoder: for each destination state, which (source state, input bit)
# pairs lead into it. A rate-1/2 binary trellis always has exactly two.
_PREDECESSORS = numpy.empty((STATE_COUNT, 2), dtype=numpy.int64)
_PREDECESSOR_BITS = numpy.empty((STATE_COUNT, 2), dtype=numpy.int64)
_fill = numpy.zeros(STATE_COUNT, dtype=numpy.int64)
for _state in range(STATE_COUNT):
    for _bit in (0, 1):
        _destination = int(_NEXT_STATE[_state, _bit])
        _slot = int(_fill[_destination])
        _PREDECESSORS[_destination, _slot] = _state
        _PREDECESSOR_BITS[_destination, _slot] = _bit
        _fill[_destination] += 1
del _fill, _state, _bit, _destination, _slot


def encoded_length(payload_bits):
    """Coded bit count for a payload, including the zero-termination tail."""
    return (int(payload_bits) + TAIL_BITS) * RATE_DENOMINATOR


def convolutional_encode(bits):
    """Encode ``bits`` at rate 1/2, zero-terminating the trellis.

    Termination costs six extra bits but lets the decoder start *and* finish at
    a known state, which is worth appreciably more than six bits' redundancy on
    frames this short.
    """
    bits = numpy.asarray(bits, dtype=numpy.uint8)
    padded = numpy.concatenate((bits, numpy.zeros(TAIL_BITS, dtype=numpy.uint8)))
    output = numpy.empty(len(padded) * RATE_DENOMINATOR, dtype=numpy.uint8)
    state = 0
    for index, bit in enumerate(padded):
        register = (state << 1) | int(bit)
        for slot, polynomial in enumerate(POLYNOMIALS):
            output[index * RATE_DENOMINATOR + slot] = int(_parity(register & polynomial))
        state = register & (STATE_COUNT - 1)
    return output


def viterbi_decode(soft_values, payload_bits):
    """Soft-decision Viterbi decode.

    ``soft_values`` are signed reals in the convention produced by
    :func:`gpsk_comms.dsss.differential_decode`: positive means a zero bit.
    Magnitudes carry confidence and are used as such -- passing hard +/-1 values
    still works but gives up about 2 dB.

    The state update is vectorised across all 64 states, so the only Python loop
    runs once per coded symbol pair (a few hundred iterations per frame).
    """
    payload_bits = int(payload_bits)
    total_steps = payload_bits + TAIL_BITS
    expected = total_steps * RATE_DENOMINATOR
    soft = numpy.asarray(soft_values, dtype=numpy.float32)
    if len(soft) < expected:
        raise ValueError(f"need {expected} soft values, got {len(soft)}")
    received = soft[:expected].reshape(total_steps, RATE_DENOMINATOR)

    # Path metrics: start pinned to state 0, everything else unreachable.
    metrics = numpy.full(STATE_COUNT, -numpy.inf, dtype=numpy.float32)
    metrics[0] = 0.0
    decisions = numpy.empty((total_steps, STATE_COUNT), dtype=numpy.uint8)

    for step in range(total_steps):
        # Correlation metric for both branches entering every state at once.
        branch = _OUTPUTS[_PREDECESSORS, _PREDECESSOR_BITS] @ received[step]
        candidates = metrics[_PREDECESSORS] + branch
        choice = numpy.argmax(candidates, axis=1)
        decisions[step] = choice.astype(numpy.uint8)
        metrics = candidates[numpy.arange(STATE_COUNT), choice]

    # Traceback from state 0, which termination guarantees is the end state.
    bits = numpy.empty(total_steps, dtype=numpy.uint8)
    state = 0
    for step in range(total_steps - 1, -1, -1):
        slot = int(decisions[step, state])
        bits[step] = _PREDECESSOR_BITS[state, slot]
        state = int(_PREDECESSORS[state, slot])
    return bits[:payload_bits]


def interleave(values, depth):
    """Block-interleave by writing rows of width ``depth`` and reading columns.

    ``depth`` is the burst length in symbols that gets spread out. Zero or one
    disables interleaving. The input is zero-padded to fill the block and the
    padding is removed again by :func:`deinterleave`.
    """
    values = numpy.asarray(values)
    depth = int(depth)
    if depth <= 1:
        return values
    rows = -(-len(values) // depth)
    padded = numpy.zeros(rows * depth, dtype=values.dtype)
    padded[: len(values)] = values
    return padded.reshape(rows, depth).T.ravel()


def deinterleave(values, depth, length=None):
    """Inverse of :func:`interleave`."""
    values = numpy.asarray(values)
    depth = int(depth)
    if depth <= 1:
        return values if length is None else values[:length]
    rows = len(values) // depth
    restored = values[: rows * depth].reshape(depth, rows).T.ravel()
    return restored if length is None else restored[:length]


def encode_frame(bits, depth=0):
    """Convolutionally encode then interleave."""
    return interleave(convolutional_encode(bits), depth)


def decode_frame(soft_values, payload_bits, depth=0):
    """Deinterleave then soft-decision decode."""
    expected = encoded_length(payload_bits)
    return viterbi_decode(deinterleave(soft_values, depth, expected), payload_bits)
