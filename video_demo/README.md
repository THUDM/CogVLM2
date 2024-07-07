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