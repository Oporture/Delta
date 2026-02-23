import os
import glob
import re
import torch
from colorama import init, Fore, Style
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset

# Initialize colorama
init(autoreset=True)

# --- Configuration ---
# Use an Unsloth-optimized base model for best speed/memory
MODEL_NAME = "ibm-granite/granite-4.0-350m" 

MAX_SEQ_LENGTH = 2048
LORA_RANK = 16
EPOCHS = 8
BATCH_SIZE = 1
LR = 1e-3

EXAMPLES_DIR = "examples"
OUTPUT_DIR = "delta-model-finedtuned"

def prepare_dataset():
    """Parses .md files from examples/ into a Hugging Face Dataset."""
    print(f"{Fore.CYAN}Preparing data...")
    
    data = []
    md_files = glob.glob(os.path.join(EXAMPLES_DIR, "*.md"))
    
    for file_path in md_files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            matches = re.findall(r"<(user|assistant)>(.*?)</\1>", content, re.DOTALL)
            
            if matches:
                full_text = ""
                for role, text in matches:
                    full_text += f"<{role}>{text.strip()}</{role}>\n"
                
                # Unsloth likes standard EOS tokens for better training
                data.append({"text": full_text})

    if not data:
        return None
        
    return Dataset.from_list(data)

def main():
    print(f"{Fore.MAGENTA}{Style.BRIGHT}--- Unsloth Fine-Tuning Pipeline (CUDA) ---")
    
    dataset = prepare_dataset()
    if not dataset:
        print(f"{Fore.RED}No samples found in {EXAMPLES_DIR}! Add some with write.py first.")
        return

    print(f"{Fore.CYAN}Loading model and tokenizer with Unsloth: {MODEL_NAME}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        load_in_4bit = True,
        trust_remote_code = True,
    )

    print(f"{Fore.CYAN}Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = LORA_RANK,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 32,
        lora_dropout = 0, # Optimized for speed
        bias = "none",
        use_gradient_checkpointing = "unsloth", # 4x less memory usage
        random_state = 3407,
    )

    training_args = TrainingArguments(
        per_device_train_batch_size = BATCH_SIZE,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = EPOCHS,
        learning_rate = LR,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = OUTPUT_DIR,
        report_to = "none",
    )

    print(f"\n{Fore.MAGENTA}{Style.BRIGHT}Training Configuration:")
    print(f"{Fore.WHITE}• Model:      {MODEL_NAME}")
    print(f"{Fore.WHITE}• Samples:    {len(dataset)}")
    print(f"{Fore.WHITE}• Device:     CUDA (Unsloth Optimized)")
    print(f"{Fore.WHITE}• Max Seq:    {MAX_SEQ_LENGTH}")
    print(f"{Fore.WHITE}• Precision:  4-bit (bitsandbytes)\n")

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        dataset_num_proc = 2,
        args = training_args,
    )

    print(f"{Fore.YELLOW}Launching Unsloth fast training...")
    trainer.train()

    print(f"\n{Fore.GREEN}{Style.BRIGHT}Training Complete!")
    
    # Save the LoRA adapters
    lora_path = os.path.join(OUTPUT_DIR, "lora_model")
    print(f"{Fore.CYAN}Saving adapters to: {lora_path}")
    model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)

    print(f"\n{Fore.CYAN}Exporting to GGUF format for Ollama...")
    model.save_pretrained_gguf("delta-model", tokenizer, quantization_method = "q4_k_m")

    print(f"\n{Fore.GREEN}{Style.BRIGHT}Pipeline Complete!")
    print(f"{Fore.WHITE}Final GGUF saved as: delta-model.Q4_K_M.gguf")

if __name__ == "__main__":
    main()
