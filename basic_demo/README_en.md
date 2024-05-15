## Basic Demo

### Minimum Requirements

Python: 3.10.12 or above

GPU requirements are as shown in the table below:

| Model Name                  | 19B Series Model | Remarks                      |
|-----------------------------|------------------|------------------------------|
| BF16 / FP16 Inference       | 42GB             | Tested with 2K dialogue text |
| Int4 Inference              | 16GB             | Tested with 2K dialogue text |
| BF16 Lora Tuning (Freeze Vision Expert Part) | 57GB             | Training text length is 2K   |
| BF16 Lora Tuning (With Vision Expert Part)  | \> 80GB          | Single GPU cannot tune       |

## CLI Model Invocation

Run this code to start a conversation in the command line.

```shell
python cli_demo.py
```

If you have multiple GPUs but a single GPU cannot load the entire model, you can use the following code to start the model on multiple GPUs.

```shell
python cli_demo_multi_gpu.py
```

## Using Web Demo

Run this code to start a conversation in the WebUI.

```shell
chainlit run web_demo.py
```

After starting the conversation, you will be able to interact with the model, as shown below:

<img src="../resources/web_demo.png" alt="web_demo" width="600" />
