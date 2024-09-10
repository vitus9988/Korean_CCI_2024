import os
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from src.data import CustomDataset, DataCollatorForSupervisedDataset
from datasets import Dataset
import itertools

def loop_train(ad: int, ep:int, lora_r: int, lora_al:int, lora_dr: float):
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"

    base_model = "google/gemma-2-9b-it"
    new_model = f"../adapters/gemma-2-9b-{ad}t_{ep}ep_{lora_r}r_{lora_al}al_{lora_dr}_dr"
    torch_dtype=torch.bfloat16

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        device_map="auto"
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dataset = CustomDataset("resource/data/train.json", tokenizer)
    valid_dataset = CustomDataset("resource/data/dev.json", tokenizer)

    train_dataset = Dataset.from_dict({
        'input_ids': train_dataset.inp,
        "labels": train_dataset.label,
        })
    valid_dataset = Dataset.from_dict({
        'input_ids': valid_dataset.inp,
        "labels": valid_dataset.label,
        })
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    peft_params = LoraConfig(
        r=lora_r, 
        lora_alpha=lora_al,
        lora_dropout=lora_dr,
        task_type="CAUSAL_LM",
        bias="none",
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]
        )

    training_args = SFTConfig(
        output_dir="resource/results",
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=64,
        learning_rate=2e-4,
        weight_decay=0.01,
        num_train_epochs=ep,
        max_steps=-1,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        log_level="info",
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        packing=True,
        seed=42,
        max_seq_length=2048,
        optim="paged_adamw_8bit",
    )

    trainer = SFTTrainer(
        model=model,
        peft_config=peft_params,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
    )
    
    trainer.train()
    trainer.save_model(new_model)

if __name__ == "__main__":
    ep_list = [3, 4, 5]
    lora_r_list = [4, 8, 16, 32, 64]
    lora_al_list = [8, 16, 32, 64, 128]
    lora_dr_list = [0.01, 0.03, 0.05]
    
    all_combinations = list(itertools.product(ep_list, lora_r_list, lora_al_list, lora_dr_list))
    unique_combinations = [comb for comb in all_combinations if len(set(comb)) == 4]
    for i, dataset in enumerate(tqdm(unique_combinations, desc="Processing"), 1):
        tqdm.write(f"현재 학습 진행중인 파라미터: ep={dataset[0]}, lora_r={dataset[1]}, lora_al={dataset[2]}, lora_dr={dataset[3]}")
        loop_train(ad=i, ep=dataset[0], lora_r=dataset[1], lora_al=dataset[2], lora_dr=dataset[3])
