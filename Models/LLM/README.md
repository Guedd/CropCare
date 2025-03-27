
Now, regarding the PLLaMa fine-tuning files, the `finetune.sh` file is used to launch the training process using the [accelerate](https://huggingface.co/docs/accelerate/en/index) library from [HuggingFace](https://huggingface.co). This setup is designed for multi-GPU training, utilizing [Full Sharded Data Parallel (FSDP)](https://huggingface.co/docs/accelerate/en/usage_guides/fsdp) for efficient training across multiple devices. The accelerator file contains the FSDP configuration necessary for this setup.

Additionally, we are applying [QLoRA (Quantized Low-Rank Adapter)](https://huggingface.co/docs/bitsandbytes/main/en/fsdp_qlora) to the model to improve the efficiency of the fine-tuning process while reducing computational costs.

The `pllama_cropcare_v1.py` file represents the first step of fine-tuning on two Hugging Face datasets. We will soon add the second step of fine-tuning on the file-based data.
