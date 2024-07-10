# Video Demo

[中文版README](./README_zh.md)

This folder contains sample code for running the CogVLM2-Video model.

## Installation

Before executing the code, please make sure that the dependencies in `basic_demo` and the additional dependencies in the current folder have been correctly installed.

```shell
pip install -r requirements.txt
```

## CLI call model

Run this code to start a conversation in the command line. Please note that to run this code, the model must be loaded on a GPU

```shell
CUDA_VISIBLE_DEVICES=0 python cli_demo.py
```

## Restful API Demo

Run this code to launch a Restful API server:

```shell
python api_demo.py
```

This will start a Restful API on the 5000 port. Run following code to make a request to the server:
```shell
python test_api.py
```

## Gradio Demo

After launch the Restful API server, you can run this code to start a Gradio web demo:

```shell
python gradio_demo.py
```
Then open the browser and visit `http://0.0.0.0:7868/` to chat with the model.