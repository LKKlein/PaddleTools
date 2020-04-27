import os
import warnings

import numpy as np
import paddle.fluid as fluid
import logging
from paddle.fluid.dygraph import Conv2D, BatchNorm, Linear
from paddletools.checkpoints import static2dynamic, dynamic2static

place = fluid.CPUPlace()
warnings.filterwarnings(action="ignore")


def reader():
    return np.ones((1, 3, 16, 16), dtype=np.float32)


def build_static_network(save_params=False, load_pretrain=None):
    main_prog = fluid.Program()
    startup_prog = fluid.Program()
    exe = fluid.Executor(place)
    with fluid.program_guard(main_prog, startup_prog):
        data = fluid.layers.data(name="img", shape=[1, 3, 16, 16], append_batch_size=False)
        conv = fluid.layers.conv2d(
            input=data, num_filters=16, filter_size=3,
            param_attr=fluid.ParamAttr(name='conv.weight'),
            bias_attr=fluid.ParamAttr(name='conv.bias'))
        bn = fluid.layers.batch_norm(
            input=conv, act="relu",
            param_attr=fluid.ParamAttr(name='bn.scale'),
            bias_attr=fluid.ParamAttr(name='bn.offset'),
            moving_mean_name='bn.mean',
            moving_variance_name='bn.variance')
        batch_size = bn.shape[0]
        f = fluid.layers.reshape(bn, [batch_size, -1])
        fc = fluid.layers.fc(
            input=f, size=3,
            param_attr=fluid.ParamAttr(name='fc.weight'),
            bias_attr=fluid.ParamAttr(name='fc.bias'))
        logits = fluid.layers.softmax(fc)
    eval_prog = main_prog.clone(True)
    exe.run(startup_prog)

    if load_pretrain:
        fluid.io.load_persistables(exe, load_pretrain, main_prog)

    d = {"img": reader()}
    result = exe.run(eval_prog, feed=d, fetch_list=[logits.name])
    logging.info(result[0])
    if save_params:
        if not os.path.exists("params"):
            os.mkdir("params")
        fluid.io.save_persistables(exe, "params", main_prog)


class TestModel(fluid.dygraph.Layer):

    def __init__(self, name):
        super(TestModel, self).__init__(name)
        self.conv = Conv2D(3, 16, filter_size=3,
                           param_attr=fluid.ParamAttr(name='conv.weight'),
                           bias_attr=fluid.ParamAttr(name='conv.bias'))
        self.bn = BatchNorm(16, act="relu",
                            param_attr=fluid.ParamAttr(name="bn.scale"),
                            bias_attr=fluid.ParamAttr(name="bn.offset"),
                            moving_mean_name="bn.mean",
                            moving_variance_name="bn.variance")
        self.fc = Linear(3136, 3,
                         param_attr=fluid.ParamAttr(name='fc.weight'),
                         bias_attr=fluid.ParamAttr(name='fc.bias'))

    def forward(self, x):
        b = x.shape[0]
        x = self.conv(x)
        x = self.bn(x)
        x = fluid.layers.reshape(x, [b, -1])
        x = self.fc(x)
        x = fluid.layers.softmax(x)
        return x


def build_dynamic_network(load_params=None, save_params=False):
    with fluid.dygraph.guard(place):
        model = TestModel("test")
        if load_params:
            model_state_dict, _ = fluid.load_dygraph(load_params)
            model.load_dict(model_state_dict, use_structured_name=False)
        model.eval()
        d = fluid.dygraph.to_variable(reader())
        p = model(d)
        logging.info(p.numpy())
        if save_params:
            fluid.save_dygraph(model.state_dict(), "dynamic_params")


if __name__ == "__main__":
    logging.info(">>> build satic network & save params...")
    build_static_network(save_params=True)
    logging.info(">>> read static params & build dynamic network...")
    static2dynamic("params", "dynamic")
    build_dynamic_network(load_params="dynamic")

    print("\n<========================>\n")

    logging.info(">>> build dynamic network & save params...")
    build_dynamic_network(save_params=True)
    logging.info(">>> read dynamic params & build static network...")
    dynamic2static("dynamic_params", "static_params")
    build_static_network(load_pretrain="static_params")
