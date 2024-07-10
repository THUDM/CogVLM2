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
## Restful API

运行以下代码以启动一个 Restful API 服务器：

```shell
python api_demo.py
```

这将会在5000端口启动一个 Restful API。运行以下代码以向服务器发送请求：
```shell
python test_api.py
```

## Gradio 演示

在启动 Restful API 服务器后，你可以运行以下代码来启动 Gradio 网页演示：

```shell
python gradio_demo.py
```
然后打开浏览器并访问 `http://0.0.0.0:7868/` 来与模型进行聊天。