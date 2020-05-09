import os
import struct

import numpy as np
import paddle.fluid as fluid
from paddletools import logger
from paddletools.config import short2size, type2short
from paddletools.utils.decoder import _decode_buf
from paddletools.utils.encoder import _encode_tensor_desc

__all__ = ["static2dynamic", "dynamic2static", "torch2dynamic"]

place = fluid.CPUPlace()


def _read_torch_dict(param_file):
    import torch
    model_state_dict = torch.load(param_file)
    state_dict = {}
    if "network" in model_state_dict:
        model_state_dict = model_state_dict["network"]
    for name, data in model_state_dict.items():
        state_dict[name] = data.numpy()
    return state_dict


def _read_static_params(param_file):
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


def _read_dynamic_params(param_file):
    state_dict = {}
    with fluid.dygraph.guard(place):
        model_state_dict, _ = fluid.load_dygraph(param_file, keep_name_table=True)
        name_table = model_state_dict.pop("StructuredToParameterName@@")
        for name, data in model_state_dict.items():
            state_dict[name_table[name]] = data
    return state_dict


def static2dynamic(params_dir, save_path=None):
    params = os.listdir(params_dir)
    logger.info("found {} parameters. start to read.".format(len(params)))
    state_dict = {}
    dtype = ""
    for param in params:
        param_path = os.path.join(params_dir, param)
        if os.path.isdir(param_path):
            continue
        data, data_type, lod_info = _read_static_params(param_path)
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


def torch2dynamic(param_file, save_path=None):
    assert os.path.exists(param_file), "{} not exists!".format(param_file)
    logger.info("start to read torch params...")
    state_dict = _read_torch_dict(param_file)
    logger.info("found {} parameters. start to transform...".format(len(state_dict)))

    dynamic_state_dict = _make_dynamic_state_dict(state_dict)
    if save_path:
        with fluid.dygraph.guard(place):
            fluid.save_dygraph(dynamic_state_dict, save_path)
        logger.info("dynamic parameters has been saved to {}.pdparams.".format(save_path))
    else:
        return dynamic_state_dict


def dynamic2static(param_file, filename):
    assert os.path.exists(param_file + ".pdparams"), "{}.pdparams not exists!".format(param_file)
    if not os.path.exists(filename):
        os.makedirs(filename)
    assert len(os.listdir(filename)) == 0, "dir {} should be empty!".format(filename)
    logger.info("start to read dynamic params...")
    static_dict = _read_dynamic_params(param_file)
    logger.info("found {} parameters. start to save to {}...".format(len(static_dict), filename))
    for name, data in static_dict.items():
        _make_static_output(filename, name, data)
    logger.info("finish!")


def _make_static_output(filename, param_name, param,
                        lod_tensor_version=0, lod_info=0, tensor_version=0):
    param_shape = param.shape
    param_type = str(param.dtype)
    save_path = os.path.join(filename, param_name)
    with open(save_path, "wb") as f:
        lod_tensor_version_str = struct.pack("I" * 1, lod_tensor_version)
        f.write(lod_tensor_version_str)
        lod_info_str = struct.pack("q" * 1, lod_info)
        f.write(lod_info_str)
        tensor_version_str = struct.pack("I" * 1, tensor_version)
        f.write(tensor_version_str)
        tensor_desc = _encode_tensor_desc(param_type, param_shape)
        buf_size = len(tensor_desc)
        buf_size_str = struct.pack("i" * 1, buf_size)
        f.write(buf_size_str)
        buf = struct.pack("c" * buf_size, *tensor_desc)
        f.write(buf)
        param_flat = param.flatten()
        short_type = type2short[param_type]
        param_str = struct.pack(short_type * len(param_flat), *param_flat)
        f.write(param_str)
    logger.debug("param: {} shape: {} save: {}".format(param_name, param_shape, save_path))


def _make_dynamic_state_dict(state_dict, data_type="float32"):
    with fluid.dygraph.guard(place):
        layer_helper = fluid.dygraph.layer_object_helper.LayerObjectHelper("transform")

        model_state_dict = {}

        for name, value in state_dict.items():
            temp_attr = fluid.ParamAttr(name=name)
            shape = value.shape
            if len(shape) < 1:
                continue
            is_bias = 'bias' in name
            initializer = fluid.initializer.NumpyArrayInitializer(value)

            logger.debug("[ToDynamic] param: {}, shape: {}".format(name, shape))
            param = layer_helper.create_parameter(temp_attr, shape, data_type, is_bias, initializer)
            model_state_dict[name] = param
    logger.info("dynamic parameters make finished!")
    return model_state_dict
