

## CogVLM2

欢迎使用 CogVLM2 开源模型，本代码为基础调用方式，欢迎使用，请注意：
+ 模型仅支持中文和英语两种语言
+ 第一次对话，必须上传图片。模型仅支持对一张图片进行对话，上传下一张图片将会覆盖之前的图片信息。
+ 此模型禁止商用，学术界可以免费使用。

Welcome to use the CogVLM2 open source model. This code is the basic calling method. Welcome to use it. Please note:
+ The model only supports two languages: Chinese and English
+ For the first conversation, you must upload a picture. The model only supports dialogue with one picture. Uploading the next picture will overwrite the previous picture information.
+ This model is prohibited for commercial use, free for academic use.

在开始运行之前，确保你已经安装了相关的依赖。

Before starting to run, make sure you have the relevant dependencies installed.

## CLI 调用模型 / CLI  demo

运行本代码以开始在命令行中对话。 / Run this code to start a conversation at the command line.

```shell
python cli_demo.py
```

## Web 端在线调用模型 / Web demo

运行本代码以开始在 WebUI 中对话。 / Run this code to start a conversation at the WebUI.
```shell
chainlit run web_demo.py
```