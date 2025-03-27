import argparse
import os
import sys
import time
from trl import SFTTrainer, SFTConfig
import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments)
from peft import get_peft_model, LoraConfig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="Xianjun/PLLaMa-7b-instruct",
        help="The pre-trained model from Hugging Face to use as basis: "
        "https://huggingface.co/models"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="The root directory under which model checkpoints are stored.",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=8,
        help="Training batch size"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        help="The number of CPU worker processes to use.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="The number of CPU worker processes to use.",
    )
    parser.add_argument(
        "--resume",
        default=False,
        action="store_true",
        help="If set, continue from a previously interrupted run. "
        "Otherwise, overwrite existing checkpoints.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=400,
        help="The number of training steps.",
    )
    parser.add_argument(
        "--peft",
        default=True,
        action='store_true',
        help="Use PEFT: https://huggingface.co/docs/peft/index"
    )
    parser.add_argument(
        "--4bit",
        default=True,
        dest="bnb_4bit",
        action='store_true',
        help="Use 4bit quantization with bitsandbytes: "
        "https://huggingface.co/docs/bitsandbytes/main/en/index"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="The Max lenth for tokenizing.",
    )
    
    args, _ = parser.parse_known_args()

    ############################################################################
    # Device Configuration
    ############################################################################
    # Read the environment variables provided by torchrun
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])

    # Then we determine the device on which to train the model.
    if rank == 0:
        print("Using PyTorch version:", torch.__version__)
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        print(f"Using GPU {local_rank}, device name: {torch.cuda.get_device_name(device)}")
    else:
        print(f"No GPU found, using CPU instead. (Rank: {local_rank})")
        device = torch.device("cpu")

    if rank == 0 and args.batch_size % world_size != 0:
        print(f"ERROR: batch_size={args.batch_size} has to be a multiple of "
              f"the number of GPUs={world_size}!")
        sys.exit(1)
    
    ############################################################################
    # Tokenizer Loading
    ############################################################################
    def tokenizer_load(model=args.model):
        print('Tokenizer Loading')
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token
        # special_tokens = tokenizer.special_tokens_map
        print('Tokenizer Loaded')
        return tokenizer

    ############################################################################
    # Model Loading
    ############################################################################
    # Load the actual base model from Hugging Face
    def model_loading(model=args.model):
        quantization_config = None
        if args.bnb_4bit:
            from transformers import BitsAndBytesConfig
            print('Quantization Config')
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_storage=torch.bfloat16,
            )
            quantization_config = bnb_config

        model = AutoModelForCausalLM.from_pretrained(
            model,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map=device,
            # attn_implementation="flash_attention_2",
            )

        if args.peft:
            print('PEFT Config')
            peft_config = LoraConfig(
                lora_alpha=32,
                lora_dropout=0.05,
                r=64,
                bias="none",
                target_modules="all-linear",
                task_type="CAUSAL_LM",
                # modules_to_save = ["lm_head", "embed_tokens"] # add if you want to use the Llama 3 instruct template
                )
        # we commented this step to use SFTTrainer
        # model = get_peft_model(model, peft_config)
        print("Using PEFT")
        model.print_trainable_parameters()
        return model, peft_config

    ############################################################################
    # Load the meithnav/agriculture data set
    ############################################################################
    def tokenizing_meithnav(x, tokenizer):
        texts = []
        print('Tokenization Intialized') 
        for instruction, response in zip(x["instruction"], x["response"]):
            # Use Llama's instruction template
            prompt = f"""<s>[INST]  
                        <<SYS>>
                        You are CropCare AI, an agricultural assistant. Provide helpful, 
                        accurate responses about farming practices, your rool is 
                        to answer some questions, and to explain information to the user.\n
                        <</SYS>>
                        User Instruction and Question: {instruction} [/INST] \n
                        CropCare Answeer: {response} {tokenizer.eos_token}"""
            
            texts.append(prompt)

        print('Text Tokenization done')     
        return tokenizer(
            texts,
            max_length=args.max_length,
            truncation=True,
            add_special_tokens=True,
            return_overflowing_tokens=True,
            return_length=False,
        )
    
    def meithnav(tokenizer):
        print('Data Loading - meithnav/agriculture')
        data_meithnav = load_dataset('meithnav/agriculture',
                                     split='train',
                                     keep_in_memory=True
                                     )
        print(f"The meithnav/agriculture dataset contains {len(data_meithnav)} rows")

        selected_data_meithnav = data_meithnav.filter(lambda x: x['subdomain'] == 'Farm Environmental Protection')
        selected_data_meithnav = selected_data_meithnav.select_columns(['instruction', 'response'])
        print(f"We have selected the instruction and response columns from the dataset, it contains {len(selected_data_meithnav)} rows")

        selected_data_meithnav_tok = selected_data_meithnav.map(
        lambda x: tokenizing_meithnav(x, tokenizer=tokenizer), # def tokenize
        remove_columns=["instruction", "response"],
        batched=True,
        batch_size=args.batch_size // world_size,
        num_proc=args.num_workers,
        )
        print('Data Tokenized - meithnav/agriculture')

        return selected_data_meithnav_tok 
    
    ############################################################################
    # Load the KisanVaani/agriculture-qa-english-only data set
    ############################################################################
    def tokenizing_kisan_vaani(x, tokenizer):
        texts = []
        print('Tokenization Initialized') 
        for question, answers in zip(x["question"], x["answers"]):
            # Use Llama's instruction template
            prompt = f"""<s>[INST]  
                        <<SYS>>
                        You are CropCare AI, an agricultural assistant. 
                        Provide helpful, accurate responses about 
                        farming practices, your role is to answer some 
                        questions and instruction, 
                        and to explain information to the user.\n
                        <</SYS>>
                        User Instruction and Question: {question} [/INST] \n
                        CropCare Answer: {answers} {tokenizer.eos_token}"""
            
            texts.append(prompt)

        print('Text Tokenization done')     
        return tokenizer(
            texts,
            max_length=args.max_length,
            truncation=True,
            add_special_tokens=True,
            return_overflowing_tokens=True,
            return_length=False,
        )
    
    def kisan_vaani(tokenizer):
        print('Data Loading - KisanVaani/agriculture-qa-english-only')
        kisan_vaani_data = load_dataset('KisanVaani/agriculture-qa-english-only',
                                     split='train',
                                     keep_in_memory=True
                                     )
        kisan_vaani_tok = kisan_vaani_data.map(
            lambda x: tokenizing_kisan_vaani(x, tokenizer=tokenizer), # def tokenize
            remove_columns=["question", "answers"],
            batched=True,
            batch_size=args.batch_size // world_size,
            num_proc=args.num_workers,
        )
        print('Data Tokenized - KisanVaani/agriculture-qa-english-only')

        return kisan_vaani_tok

    ############################################################################
    # Load the Mahesh2841/Agriculture data set
    ############################################################################
    def tokenizing_Mahesh(x, tokenizer):
        texts = []
        print('Tokenization Intialized') 
        for question, answers, instruction in zip(x["instruction"], x["question"], x["answers"]):
            # Use Llama's instruction template
            prompt = f"""<s>[INST]  
                        <<SYS>>
                        You are CropCare AI, an agricultural assistant. 
                        Provide helpful, accurate responses about 
                        farming practices, your rool is to answer some 
                        questions and instruction, 
                        and to explain information to the user.\n
                        <</SYS>>
                        User Instruction: {instruction}\n
                        User Question: {question} [/INST] \n
                        CropCare Answeer: {answers} {tokenizer.eos_token}"""
            
            texts.append(prompt)

        print('Text Tokenization done')     
        return tokenizer(
            texts,
            max_length=args.max_length,
            truncation=True,
            add_special_tokens=True,
            return_overflowing_tokens=True,
            return_length=False,
        )
    
    def Mahesh(tokenizer):
        print('Data Loading - Mahesh2841/Agriculture')
        data_Mahesh = load_dataset('Mahesh2841/Agriculture',
                                     split='train',
                                     keep_in_memory=True
                                     )
        data_Mahesh_tok = data_Mahesh.map(
            lambda x: tokenizing_Mahesh(x, tokenizer=tokenizer), # def tokenize
            remove_columns=["instruction", "question", "answers"],
            batched=True,
            batch_size=args.batch_size // world_size,
            num_proc=args.num_workers,
        )
        print('Data Tokenized - Mahesh2841/Agriculture')

        return data_Mahesh_tok           

    ############################################################################
    # Model Training
    ############################################################################
    def training(train_model, tokenizer, dataset_tok, output_dir, save_path=None, save=False):
        train_validate_splits = dataset_tok.train_test_split(seed=42, keep_in_memory=True)
        print('Train data splited into Train and Test/Validation')
        train_dataset_tok = train_validate_splits["train"]
        validate_dataset_tok = train_validate_splits["test"]

        collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, return_tensors="pt")

        training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=not args.resume,
        num_train_epochs=args.epochs,
        # save_strategy="no",  # good for testing
        save_strategy="epoch",   # use these if you actually want to save the model
        #save_steps=400,
        save_total_limit=4,
        eval_strategy="epoch",
        # eval_steps=200,  # compute validation loss every 200 steps
        learning_rate=2e-5,
        weight_decay=0.01,
        bf16=True,  # use 16-bit floating point precision
        # fp16=False,
        # divide the total training batch size by the number of GCDs for the per-device batch size
        per_device_train_batch_size=args.batch_size // world_size,
        per_device_eval_batch_size=args.batch_size,
        max_steps=args.max_steps,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        report_to=["wandb"],  # log statistics for tensorboard
        ddp_find_unused_parameters=False,
        )

        trainer = SFTTrainer(
        model=train_model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=collator,
        train_dataset=train_dataset_tok,
        eval_dataset=validate_dataset_tok,
        peft_config=peft_config,
        )

        trainer.train(resume_from_checkpoint=args.resume)

        if trainer.is_fsdp_enabled:
            trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
        
        if save:
            # Save the model using model.save_pretrained
            model_save_dir = os.path.join(os.path.join(save_path, 'model'))
            os.makedirs(model_save_dir, exist_ok=True)  # Ensure the directory exists
            train_model.save_pretrained(model_save_dir)
            print("Model Trainer - Model Saved")

            # Save the tokenizer using tokenizer.save_pretrained
            tokenizer_save_dir = os.path.join(os.path.join(save_path, 'tokenizer'))
            os.makedirs(tokenizer_save_dir, exist_ok=True)  # Ensure the directory exists
            tokenizer.save_pretrained(tokenizer_save_dir)
            print("Model Trainer - Tokenizer Saved")

            if rank == 0:
                print()
                print("Training done, you can find the final model (and checkpoints) in", output_dir)
            return train_model
        return train_model
    
    ############################################################################
    # Main part of the script
    ############################################################################
    # We also ensure that output paths exist
    model_name = args.model.replace('/', '_')

    # this is where trained model and checkpoints will go
    output_dir = os.path.join(args.output_path, model_name)

    if rank == 0:
        print("Loading model and tokenizer")
    start = time.time()
    model_tokenizer = tokenizer_load()
    model, peft_config = model_loading()
    stop = time.time()
    if rank == 0:
        print(f"Loading model and tokenizer took: {stop-start:.2f} seconds")
        print(f"Model loaded on {device} device") 

    # Train on meithnav/agriculture dataset
    meithnav_data = meithnav(tokenizer=model_tokenizer)
    # meithnav_save_path = os.path.join(output_dir, '_meithnav_agriculture dataset')
    meithnav_model = training(train_model=model,
                              tokenizer=model_tokenizer,
                              peft_config=peft_config,
                              dataset_tok=meithnav_data,
                              output_dir=output_dir,
                              # save_path=meithnav_save_path
                              
                              )
    
    # Load the saved meithnav model and train on KisanVaani/agriculture-qa-english-only dataset

    # meithnav_model_load = model_loading(os.path.join(meithnav_save_path, 'model'))
    # meithnav_tokenizer_load = tokenizer_load(os.path.join(meithnav_save_path, 'tokenizer'))

    kisan_vaani_data = kisan_vaani(tokenizer=model_tokenizer)
    # kisan_vaani_save_path = os.path.join(output_dir, "_KisanVaani_agriculture_qa_english_only")
    
    kisan_vaani_model = training(
        train_model=meithnav_model,
        tokenizer=model_tokenizer,
        peft_config=peft_config,
        dataset_tok=kisan_vaani_data,
        output_dir=output_dir,
        # save_path=kisan_vaani_save_path
    )

    # Load the saved kisan_vaani model and train on Mahesh2841/Agriculture dataset
    # kisan_vaani_model_load = model_loading(os.path.join(kisan_vaani_save_path, 'model'))
    # kisan_vaani_tokenizer_load = tokenizer_load(os.path.join(kisan_vaani_save_path, 'tokenizer'))

    Mahesh_data = Mahesh(tokenizer=model_tokenizer)
    save_path = os.path.join(output_dir, "_QA_Chat")
    
    Mahesh_model = training(
        train_model=kisan_vaani_model,
        tokenizer=model_tokenizer,
        peft_config=peft_config,
        dataset_tok=Mahesh_data,
        output_dir=output_dir,
        save_path=save_path,
        save=True
    )
