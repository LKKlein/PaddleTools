import six

from ..config import type2id


def _VarintEncoder():
    """Return an encoder for a basic varint value (does not include tag)."""

    def EncodeVarint(write, value, unused_deterministic=None):
        bits = value & 0x7f
        value >>= 7
        while value:
            write(0x80 | bits)
            bits = value & 0x7f
            value >>= 7
        return write(bits)

    return EncodeVarint


_EncodeVarint = _VarintEncoder()


def _VarintBytes(value):
    """Encode the given integer as a varint and return the bytes"""
    pieces = []
    _EncodeVarint(pieces.append, value, True)
    return pieces


def _encode_tensor_desc(data_type, dims):
    type_id = type2id.get(str(data_type), 5)
    encode_num = [8, type_id]
    for dim in dims:
        encode_num.append(16)
        encode_num += _VarintBytes(dim)

    encode_str = [six.int2byte(num) for num in encode_num]
    return encode_str
