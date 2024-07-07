# Video Demo

[Read this in English.](./README.md)

该文件夹下为运行 CogVLM2-Video 模型的示例代码。

## 安装

在执行代码之前，请您确保已经正确安装了 `basic_demo`中的依赖以及当前文件夹下的额外依赖。

```shell
pip install -r requirements.txt
```

## CLI 调用模型

运行本代码以开始在命令行中对话。请注意，运行该代码，模型必须在一张GPU上载入

```shell
CUDA_VISIBLE_DEVICES=0 python cli_demo.py
```
w