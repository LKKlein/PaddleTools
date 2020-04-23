# PaddleTools

Paddle动态图是不是很方便？是不是很好用？嗯？可是官方预训练参数都是静态图参数，自己从零开始又达不到精度，哎，还是先用静态图吧。不要方，`PaddleTools`可以帮到你！  
`PaddleTools`是非官方出品的`Paddle`工具类，主要是为了添加一些日常使用`Paddle`时的常用工具，方便你我他！`PaddleTools`开源的第一款工具就是可以将`Paddle`的静态图参数转换为当下热门方便的动态图参数，如果大家有兴趣，欢迎一起开发这个工具类，如果没兴趣，往下看看也是可以的！


## 安装

- pip安装

```shell
pip3 install paddletools --upgrade
```

- 源码安装

```shell
git clone https://github.com/LKKlein/PaddleTools.git
cd PaddleTools
pip3 install -r requirments.txt
python3 setup.py install
```

## 使用

- 命令行使用

```shell
>>> pdtools --help
Usage:
    pdtools param to_dynamic --src=<source_param> --dst=<destination_param> [--verbose]

Arguments:
    param                    paddle parameters operations.

Options:
    -s <source_param>, --src=<source_param>              dir path for static params.
    -d <destination_param>, --dst=<destination_param>    where should dynamic params to store.
    -v, --verbose                                        whether to show more logs. [default: True]

Example:
    pdtools param to_dynamic -s yolov3_pretrain/ -d yolov3 -v
```

- 代码中使用

```python
import paddle.fluid as fluid
from paddletools.checkpoints import static2dynamic

# 保存到文件
static2dynamic(params_dir="yolov3_pretrain/", save_path="yolov3")

# 读取参数到内存
model_state_dict = static2dynamic(params_dir="yolov3_pretrain/")

place = fluid.CPUPlace()
with fluid.dygraph.guard(place):
    model = YOLOv3()  # 初始化定义的动态图网络，这里假定为YOLOv3
    model.load_dict(model_state_dict)  # 将读取的参数加载到网络中
```

## 建议与意见

1. 如果你有什么关于`PaddleTools`需要实现的需求，请尽管提出你的issue，或者发邮件给我也行(lkklein@163.com)。
2. 欢迎各位同学一起开发，随时在线。
3. 欢迎大家使用，star和fork。

