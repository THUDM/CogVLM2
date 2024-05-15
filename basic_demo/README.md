## Basic Demo

### 最低配置要求

Python: 3.10.12 以上版本

GPU要求如下表格所示

| 模型名称                   | 19B 系列模型 | 备注          |
|------------------------|----------|-------------|
| BF16 / FP16 推理         | 42GB     | 测试对话文本长度为2K | 
| Int4    推理             | 16GB     | 测试对话文本长度为2K | 
| BF16 Lora 微调 (冻结视觉专家部分） | 57GB     | 训练文本的长度为2K  |
| BF16 Lora 微调 (含视觉专家部分） | \> 80GB   | 单卡无法微调      |

## CLI 调用模型 

运行本代码以开始在命令行中对话。

```shell
python cli_demo.py
```

如果您有多张GPU，但是单张GPU无法拉起完整的模型，您可以通过以下代码执行多卡拉起模型。
    
```shell
python cli_demo_multi_gpu.py
```

## Web端 在线调用模型

运行本代码以开始在 WebUI 中对话。

```shell
chainlit run web_demo.py
```
拉起对话后，你将能和模型进行对话，效果如下：

<img src="../resources/web_demo.png" alt="web_demo" width="600" />




