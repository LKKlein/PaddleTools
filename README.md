# PaddleTools

Paddle动态图是不是很方便？是不是很好用？嗯？可是官方预训练参数都是静态图参数，自己从零开始又达不到精度，哎，还是先用静态图吧。不要方，`PaddleTools`可以帮到你！  
`PaddleTools`是非官方出品的`Paddle`工具类，主要是为了添加一些日常使用`Paddle`时的常用工具，方便你我他！`PaddleTools`开源的第一款工具就是可以将`Paddle`的静态图参数转换为当下热门方便的动态图参数。目前，`PaddleTools`已支持将`Paddle`的参数动静态相互转换，同时还能将`Pytorch`的参数转到`Paddle`的动态图。如果大家有兴趣，欢迎一起开发这个工具类，如果没兴趣，往下看看也是可以的！

## 工具列表

- [x] `PaddlePaddle`静态图参数转动态图参数
- [x] `PaddlePaddle`动态图参数转静态图参数
- [x] `Pytorch`参数转`PaddlePaddle`动态图参数



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

**请注意：**  
**1. 我们在搭建动态网络时，请将参数命名(ParamAttr)与静态图保持一致，这样才能正确读取参数！！！**  
**2. 静态图参数只需要传递参数所在的文件夹名就可以了，动态图需要传递参数文件的名字，不包括`.pdparams`**  

- 命令行使用

```shell
>>> pdtools --help
Usage:
    pdtools param (to_dynamic | to_static | from_torch) --src=<source_param> --dst=<destination_param> [--verbose]

Arguments:
    param                    paddle parameters operations.

Options:
    -s <source_param>, --src=<source_param>              dir path for source params.
    -d <destination_param>, --dst=<destination_param>    where should dest params to store.
    -v, --verbose                                        whether to show more logs. [default: True]

Example:
    pdtools param to_dynamic -s yolov3_pretrain/ -d yolov3 -v
    pdtools param to_static -s yolov3 -d yolov3_pretrain/ -v
    pdtools param from_torch -s yolov3.pth -d yolov3 -v
```

- 代码中使用  


1. 静态图转动态图

```python
import paddle.fluid as fluid
from paddletools.checkpoints import static2dynamic

# 第一种方式，保存到文件，然后使用动态网络从文件加载参数
static2dynamic(params_dir="yolov3_pretrain/", save_path="yolov3")

# 第二种方式，读取参数到内存，直接加载到网络中
model_state_dict = static2dynamic(params_dir="yolov3_pretrain/")

place = fluid.CPUPlace()
with fluid.dygraph.guard(place):
    model = YOLOv3()  # 初始化定义的动态图网络，这里假定为YOLOv3
    # 注意，这里的use_structured_name一定要设置为false，structured_name是由系统自动取的，与我们自己的命名不同
    # 这里可能会出现一个warning，但是没有关系，我们的参数已经是成功读取了的
    model.load_dict(model_state_dict, use_structured_name=False)  # 将读取的参数加载到网络中
```

2. 动态图转静态图

```python
from paddletools.checkpoints import dynamic2static

dynamic2static(param_file="yolov3", filename="yolov3_pretrain/")
```

3. torch参数转动态图

```python
from paddletools.checkpoints import torch2dynamic

# 第一种方式，保存到文件，然后使用动态网络从文件加载参数
torch2dynamic(param_file="yolov3_pretrain.pth", save_path="yolov3")

# 第二种方式，读取参数到内存，直接加载到网络中
model_state_dict = torch2dynamic(param_file="yolov3_pretrain.pth")
```


## 测试

1. 使用`paddlepaddle`在`obj365`上预训练的`YoloV3`参数进行测试

```shell
wget https://paddlemodels.bj.bcebos.com/object_detection/ResNet50_vd_dcn_db_obj365_pretrained.tar
tar -xvf ResNet50_vd_dcn_db_obj365_pretrained.tar
pdtools param to_dynamic -s ResNet50_vd_dcn_db_obj365_pretrained -d yolov3
```
转换完成会在当前目录下生成一个`yolov3.pdparams`的动态图参数文件

2. 搭建网络测试转换精度

- 静态图转动态图测试

网络详情见`test/demo.py`，主要是一层卷积、一层批归一化、一层全连接，测试结果是静态图与动态图输出完全一致。
```shell
cd test
python3 demo.py
```


## 建议与意见

1. 如果你有什么关于`PaddleTools`需要实现的需求，请尽管提出你的issue，或者发邮件给我也行(lkklein@163.com)。
2. 欢迎各位同学一起开发，随时在线。
3. 欢迎大家使用，star和fork。

