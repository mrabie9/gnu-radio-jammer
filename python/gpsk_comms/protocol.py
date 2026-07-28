"""Wire format shared by the GMSK command transmitter and receiver."""

import struct
import zlib

PROTOCOL_VERSION = 1
DEFAULT_ACCESS_CODE = "D391DA26"
ACCESS_CODE_BYTES = 4
BODY_FORMAT = ">BHHB"
BODY_SIZE = struct.calcsize(BODY_FORMAT)
PAYLOAD_SIZE = BODY_SIZE + 4

COMMAND_TO_ID = {
    "stop": 0,
    "forward": 1,
    "backward": 2,
    "left": 3,
    "right": 4,
}
ID_TO_COMMAND = {value: key for key, value in COMMAND_TO_ID.items()}


class PacketError(ValueError):
    """Base class for rejected command payloads."""


class PacketLengthError(PacketError):
    """Raised when a payload is not the fixed protocol length."""


class PacketCRCError(PacketError):
    """Raised when a payload CRC does not match its body."""


class PacketFormatError(PacketError):
    """Raised when a CRC-valid payload has unsupported field values."""


def normalise_access_code(value):
    """Return a four-byte access code from an 8-digit hexadecimal string."""
    if isinstance(value, bytes):
        raw = value
    else:
        text = str(value).strip().replace("0x", "").replace("_", "")
        if len(text) != ACCESS_CODE_BYTES * 2:
            raise ValueError("access_code must contain exactly 8 hexadecimal digits")
        try:
            raw = bytes.fromhex(text)
        except ValueError as error:
            raise ValueError("access_code must be hexadecimal") from error
    if len(raw) != ACCESS_CODE_BYTES:
        raise ValueError("access_code must contain exactly 4 bytes")
    return raw


def encode_payload(session_id, sequence, command):
    """Encode a command payload and append its big-endian CRC-32."""
    if command not in COMMAND_TO_ID:
        raise ValueError(f"unsupported command: {command!r}")
    body = struct.pack(
        BODY_FORMAT,
        PROTOCOL_VERSION,
        int(session_id) & 0xFFFF,
        int(sequence) & 0xFFFF,
        COMMAND_TO_ID[command],
    )
    return body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def decode_payload(payload):
    """Validate and decode a payload.

    Returns ``(session_id, sequence, command)``. Invalid data raises ValueError.
    """
    payload = bytes(payload)
    if len(payload) != PAYLOAD_SIZE:
        raise PacketLengthError(f"payload must be exactly {PAYLOAD_SIZE} bytes")
    body, crc_bytes = payload[:-4], payload[-4:]
    expected_crc = struct.unpack(">I", crc_bytes)[0]
    if zlib.crc32(body) & 0xFFFFFFFF != expected_crc:
        raise PacketCRCError("CRC mismatch")
    version, session_id, sequence, command_id = struct.unpack(BODY_FORMAT, body)
    if version != PROTOCOL_VERSION:
        raise PacketFormatError(f"unsupported protocol version: {version}")
    try:
        command = ID_TO_COMMAND[command_id]
    except KeyError as error:
        raise PacketFormatError(f"unsupported command id: {command_id}") from error
    return session_id, sequence, command


def bytes_to_bits(data):
    """Expand packed bytes into an MSB-first list of zero/one integers."""
    return [(byte >> bit) & 1 for byte in data for bit in range(7, -1, -1)]
