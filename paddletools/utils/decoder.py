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
    dim_flag = type_flag = False
    type_id = 5
    for b in out:
        if b == 8 and not type_flag:
            type_flag = True
            continue

        if b == 16 and not dim_flag:
            dim_flag = True
            continue

        if type_flag:
            type_id = b
            type_flag = False

        if dim_flag:
            dims.append(b)
            dim_flag = False
    if type_id not in id2type:
        type_id = 5
    return id2type[type_id], dims
