# PaddleTools

Paddle动态图是不是很方便？是不是很好用？嗯？可是官方预训练参数都是静态图参数，自己从零开始又达不到精度，哎，还是先用静态图吧。不要方，`PaddleTools`可以帮到你！  
`PaddleTools`是非官方出品的`Paddle`工具类，主要是为了添加一些日常使用`Paddle`时的常用工具，方便你我他！`PaddleTools`开源的第一款工具就是可以将`Paddle`的静态图参数转换为当下热门方便的动态图参数。目前，`PaddleTools`已支持将`Paddle`的参数动静态相互转换，同时还能将`Pytorch`的参数转到`Paddle`的动态图。如果大家有兴趣，欢迎一起开发这个工具类，如果没兴趣，往下看看也是可以的！

## 工具列表

- 参数系列
  - [x] `PaddlePaddle`静态图参数转动态图参数
  - [x] `PaddlePaddle`动态图参数转静态图参数
  - [x] `Pytorch`参数转`PaddlePaddle`动态图参数
  
- 日志系列
  - [x] 提供日志Logger，统一输出标准，同时支持日志输出到文件，方便在AIStudio使用

- 进度提醒系列
  - [x] 添加微信消息提醒(Server酱)
  - [x] 添加邮件消息提醒

- 训练辅助系列
  - [x] 命令行参数统一化
  - [ ] 记录命令行参数改动
  - [ ] 联动日志，记录参数改动带来的变化

- 数据读取辅助
  - [ ] 数据增强类
  - [ ] 图像分类数据读取类
  - [ ] 目标检测数据读取类

- 动态图模型OP
  - [ ] 卷积+归一化层

- 动态图分布式训练工具
  - [ ] 分布式训练类


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

功能列表:  
- [PaddleTools](#paddletools)
  - [工具列表](#工具列表)
  - [安装](#安装)
  - [使用](#使用)
    - [参数转换](#参数转换)
    - [日志输出](#日志输出)
    - [消息提醒](#消息提醒)
  - [测试](#测试)
  - [建议与意见](#建议与意见)

### 参数转换

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

### 日志输出

`PaddleTools`提供了封装好的彩色日志输出，无需再对`logger`做进一步的配置。用过`logging`的同学应该会比较熟悉。


```python
from paddletools import logger

# 五种标准输出，默认输出级别是info
logger.debug("test")
logger.info("test")
logger.warning("test")
logger.error("test")
logger.critical("test")

# 新增的两种输出级别，级别在info纸上，warning之下，用于训练和验证
logger.train("test")
logger.eval("test")
```

输出界面长这样：  

![](http://img.lkklein.xyz/logger_output.png)

**另外，logger在显示输出的同时，还支持将输出保存到文件，方便之后查看日志。这一点对AIStudio用户来说非常重要！！！**  
只需要在程序入口设置添加下面这一行就可以了。

```python
from paddletools import logger

# filename是需要存储日志的文件地址，including_all表示是否将引用的其他包的输出也存储到文件，默认为全部存储
logger.log_to_file(filename="path/to/logfile", including_all=True)
```

### 消息提醒

目前提供微信和邮件两种消息提醒方式。消息提醒可以用于推送模型训练的进度，也可以用于程序异常的警告。

- 微信消息推送

微信消息使用[`Server酱`](http://sc.ftqq.com/3.version)进行推送，使用方式也非常简单，只需要使用GitHub账号登录，然后使用微信扫码绑定即可获取一个`secret`，具体见[官网](http://sc.ftqq.com/3.version)。获取`secret`之后即可开始使用。  
**不过，请注意，每人每天发送上限500条，相同内容5分钟内不能重复发送，不同内容一分钟只能发送30条。请注意控制推送的数量！！！**

```python
from paddletools.reminder.wechat import WeChatReminder

reminder = WeChatReminder("your-secret")
reminder.send(title="我是标题", content="我是正文")
```

- 邮件消息推送

邮件目前支持`163邮箱`、`126邮箱`、`QQ邮箱`、`Gmail邮箱`四种，后续会慢慢进行扩展的。在使用邮件发送消息之前，请先确认发送邮件的邮箱已经开启了`SMTP服务`，具体开启方式请自行百度，关键词可参考`163邮箱开启SMTP`。

```python
from paddletools.reminder.email import EmailReminder

# 这里需提供发送邮件的邮箱，接收邮件的邮箱，以及发送邮件的邮箱密码
# 部分邮箱的密码为开启SMTP服务时给的授权码，而不是登录密码，如163、QQ，请注意区分
reminder = EmailReminder(send_mail="xxx#163.com", receive_mail="xxx@qq.com", password="xxxxx")
reminder.send(title="我是标题", content="我是正文")
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

