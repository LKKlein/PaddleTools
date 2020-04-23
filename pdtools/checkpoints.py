import logging
import os
import struct

import numpy as np
import paddle.fluid as fluid

from .config import short2size, type2short
from .decoder import _decode_buf

place = fluid.CPUPlace()
logger = logging.getLogger("pdtools")
logger.setLevel(logging.INFO)


def _read_params(param_file):
    with open(param_file, 'rb') as f:
        lod_tensor_version, = struct.unpack("I" * 1, f.read(4))
        lod_info, = struct.unpack("q" * 1, f.read(8))
        tensor_version, = struct.unpack("I" * 1, f.read(4))
        buf_size, = struct.unpack("i" * 1, f.read(4))
        buf = struct.unpack("c" * buf_size, f.read(buf_size))
        buf_str = b''
        for b in buf:
            buf_str += b
        data_type, dims = _decode_buf(buf_str)
        dim_size = np.product(dims)
        type_short = type2short[data_type]
        short_size = short2size[type_short]
        data = struct.unpack(type_short * dim_size, f.read(short_size * dim_size))
        data = np.asarray(data).astype(data_type).reshape(dims)
    return data, data_type, lod_info


def static2dynamic(params_dir, save_path=None):
    params = os.listdir(params_dir)
    logger.info("found {} parameters. start to read.".format(len(params)))
    state_dict = {}
    dtype = ""
    for param in params:
        param_path = os.path.join(params_dir, param)
        if os.path.isdir(param_path):
            continue
        data, data_type, lod_info = _read_params(param_path)
        logger.debug("param: {}, shape: {}, data type: {}".format(param, data.shape, data_type))
        state_dict[param] = data
        dtype = data_type

    logger.info("parameters read finished! start to transform to dynamic!")
    dynamic_state_dict = _make_dynamic_state_dict(state_dict, dtype)
    if save_path:
        with fluid.dygraph.guard(place):
            fluid.save_dygraph(dynamic_state_dict, save_path)
        logger.info("dynamic parameters has been saved to {}.pdparams.".format(save_path))
    else:
        return dynamic_state_dict


def _make_dynamic_state_dict(state_dict, data_type="float32"):
    with fluid.dygraph.guard(place):
        layer_helper = fluid.dygraph.layer_object_helper.LayerObjectHelper("transform")

        model_state_dict = {}

        for name, value in state_dict.items():
            temp_attr = fluid.ParamAttr(name=name)
            shape = value.shape
            is_bias = 'bias' in name
            initializer = fluid.initializer.NumpyArrayInitializer(value)

            param = layer_helper.create_parameter(temp_attr, shape, data_type, is_bias, initializer)
            model_state_dict[name] = param
    logger.info("dynamic parameters make finished!")
    return model_state_dict
