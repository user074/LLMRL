from transformers import LlamaForSequenceClassification, AutoTokenizer, LlamaForCausalLM
import torch
import wandb
import os
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset

wandb.login()
os.environ['WANDB_PROJECT'] = 'LLMRL'

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b", # "unsloth/tinyllama" for 16bit loading
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

#Use LoRA to reduce memory usage:
# model = FastLanguageModel.get_peft_model(
#     model,
#     r = 256, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
#     target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
#                       "gate_proj", "up_proj", "down_proj",],
#     lora_alpha = 128,
#     lora_dropout = 0, # Currently only supports dropout = 0
#     bias = "none",    # Currently only supports bias = "none"
#     use_gradient_checkpointing = "unsloth", # @@@ IF YOU GET OUT OF MEMORY - set to True @@@
#     random_state = 3407,
#     use_rslora = False,  # We support rank stabilized LoRA
#     loftq_config = None, # And LoftQ
# )


# prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

# You are a helpful assistant to solve math problems step by step <|eot_id|><|start_header_id|>user<|end_header_id|>

# {}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

# {}"""

# def formatting_prompts_func(examples):
#     texts = []
    
#     for instruction, responses, next_response, rating in zip(examples['instruction'], examples['responses'], examples['next_response'], examples['rating']):
#         # Combine all responses and the next response into a single string with newline separation
#         combined_responses = " + \n".join(responses) + " + \n" + next_response
#         if rating == -1:
#             combined_responses = combined_responses + " - \n"
#         else:
#             combined_responses = combined_responses + " + \n"
        
#         # Format the text with the prompt template
#         text = prompt.format(instruction, combined_responses) 
#         texts.append(text)

    
#     return {"text": texts,}



# # Load and preprocess the dataset
# dataset = load_dataset("Birchlabs/openai-prm800k-stepwise-critic", split='train')
# dataset = dataset.filter(lambda x: x['rating'] is not None)  # Filter entries without ratings

# #filter out the examples that has 'next_response' in the responses of the solution
# dataset = dataset.filter(lambda x: not(x['rating'] == 1 and x['is_solution'] == False))

# #convert ratings of 0 to 1 so we have only binary labels
# dataset = dataset.map(lambda x: {'rating': 1 if x['rating'] == 0 else x['rating']})

# dataset = dataset.map(formatting_prompts_func, batched=True)  # Apply the preprocessing function

dataset = load_dataset("peiyi9979/Math-Shepherd", split='train')

def tokenize_function(examples):
    return tokenizer(examples["input"], padding="max_length", truncation=True,  max_length=512)

def tokenize_labels_function(examples):
    tokenized_labels = tokenizer(examples["label"], padding="max_length", truncation=True,  max_length=512)
    return {"labels": tokenized_labels["input_ids"]}

dataset = dataset.map(tokenize_function, batched=True)
dataset = dataset.map(tokenize_labels_function, batched=True)
dataset= dataset.remove_columns(['input', 'label', 'task'])

from trl import SFTTrainer
from transformers import TrainingArguments, Trainer
from transformers.utils import logging
logging.set_verbosity_info()

trainer = Trainer(
    model = model,
    train_dataset = dataset,
    args = TrainingArguments(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4,
        warmup_ratio = 0.1,
        num_train_epochs = 1,
        learning_rate = 4e-6,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 100,
        save_steps= 5000,
        save_total_limit=2,
        optim = "adamw_8bit",
        weight_decay = 0.1,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = "checkpoints/llama3-8b-critic-fullfinetune",
        report_to= "wandb"
    ),
)

trainer_stats = trainer.train()

model.save_pretrained("checkpoints/llama3-8b-critic-fullfinetune") # Local saving
