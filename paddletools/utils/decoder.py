import six

from ..config import id2type


def _VarintDecoder(mask, result_type):
    def DecodeVarint(buffer, pos):
        result = 0
        shift = 0
        while 1:
            b = six.indexbytes(buffer, pos)
            result |= ((b & 0x7f) << shift)
            pos += 1
            if not (b & 0x80):
                result &= mask
                result = result_type(result)
                return (result, pos)
            shift += 7
            if shift >= 64:
                raise Exception('Too many bytes when decoding varint.')
    return DecodeVarint


_DecodeVarint = _VarintDecoder((1 << 64) - 1, int)


def _decode_buf(buf):
    pos = 0
    out = []
    while pos < len(buf):
        res, pos = _DecodeVarint(buf, pos)
        out.append(res)

    dims = []
    type_id = 5
    for idx, b in enumerate(out):
        if idx == 1:
            type_id = b
        elif (idx - 1) % 2 == 0:
            dims.append(b)
    if type_id not in id2type:
        type_id = 5
    return id2type[type_id], dims
