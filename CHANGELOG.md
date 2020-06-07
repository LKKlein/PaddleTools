# Changelog

## [0.0.6] - 2020-05-20

### bug fix  

- `torch`转`paddle`中，部分参数命名不统一，比如`BN`的`mean`和`var`，这里强制转换了这些命名 
- `torch`和`paddle`有部分Layer的参数形状不同，比如`Linear`层，`Paddle`中为[in_dim, out_dim]，`torch`中则相反，需要转换一下
- 静态图权重解码时的bug

### new feature  

- 添加了命令行参数工具`PDConfig`，便捷的在训练时添加命令行参数
- `demo.py`中添加了torch到paddle动态图参数转换的精度测试

## [0.0.5] - 2020-05-10

### finement  

- 调整requirements，隐藏一些大的package，用户自由选择安装
- 调整logger多行输出的单行最长长度为150

## [0.0.4] - 2020-04-29

### new feature  

- 添加了日志输出功能，提供统一的彩色Logger，统一输出标准，同时支持将日志输出到文件，方便在AIStudio使用
- 新增微信消息推送的功能，方便在微信中随时观察训练进度等
- 新增邮件消息发送的功能

## [0.0.3] - 2020-04-27

### new feature  

- 添加动态图转静态图参数功能，同时添加命令行转换工具

## [0.0.2] - 2020-04-24

### new feature  

- 添加`torch`的`state_dict`参数文件转`Paddle`的动态图参数文件，同时添加了命令行转换工具

## [0.0.1] - 2020-04-23

### new feature  

- 创建`PaddleTools`项目，添加一些日常可能用到的Paddle工具
- 添加静态图参数转动态图参数的功能，目前只支持`fp32`的参数，以及一个文件夹下多个独立参数文件的静态图参数
- 添加静态图参数转动态图参数的命令行工具
