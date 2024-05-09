from transformers import LlamaForSequenceClassification, AutoTokenizer, LlamaForCausalLM
import torch
import wandb
import os

wandb.login()
os.environ['WANDB_PROJECT'] = 'LLMRL'

from unsloth.models.loader import FastLanguageModel
import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/tinyllama", # "unsloth/tinyllama" for 16bit loading
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

#well for llama3 'pre-trained models usually do not stop completions naturally.'
#https://github.com/meta-llama/llama3/blob/main/example_text_completion.py

#Use LoRA to reduce memory usage:
# model = FastLanguageModel.get_peft_model(
#     model,
#     r = 32, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
#     target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
#                       "gate_proj", "up_proj", "down_proj",],
#     lora_alpha = 32,
#     lora_dropout = 0, # Currently only supports dropout = 0
#     bias = "none",    # Currently only supports bias = "none"
#     use_gradient_checkpointing = False, # @@@ IF YOU GET OUT OF MEMORY - set to True @@@
#     random_state = 3407,
#     use_rslora = False,  # We support rank stabilized LoRA
#     loftq_config = None, # And LoftQ
# )
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
prompt = """{} {}"""

EOS_TOKEN = tokenizer.eos_token
# def formatting_prompts_func(example):
#     questions = example['question']
#     answers = example['answer']
#     output_texts = []
#     for questions, answers in zip(questions, answers):
#         output_text = prompt.format(questions, answers)+ EOS_TOKEN
#         output_texts.append(output_text)
#     return { "text" : output_texts, }
# pass
# from datasets import load_dataset
# dataset = load_dataset("gsm8k", 'main', split='train')
# dataset = dataset.map(formatting_prompts_func, batched = True,)
# print(dataset[0])

EOS_TOKEN = tokenizer.eos_token
def formatting_prompts_func(example):
    questions = example['query']
    answers = example['response']
    output_texts = []
    for questions, answers in zip(questions, answers):
        output_text = prompt.format(questions, answers)+ EOS_TOKEN
        output_texts.append(output_text)
    return { "text" : output_texts, }
pass
from datasets import load_dataset
# dataset = load_dataset("gsm8k", 'main', split='train')
dataset = load_dataset("meta-math/MetaMathQA", split='train')
dataset = dataset.map(formatting_prompts_func, batched = True,)
print(dataset[0])

from trl import SFTTrainer
from transformers import TrainingArguments
from transformers.utils import logging
logging.set_verbosity_info()

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, 
    args = TrainingArguments(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 2,
        warmup_ratio = 0.1,
        num_train_epochs = 2,
        learning_rate = 2e-5,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        save_strategy = "epoch",
        optim = "adamw_8bit",
        weight_decay = 0.1,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = "checkpoints/tinyllama-MetaMath",
        report_to= "wandb",

    ),
)
trainer_stats = trainer.train()

model.save_pretrained("checkpoints/tinyllama-MetaMath") # Local saving
tokenizer.save_pretrained("checkpoints/tinyllama-MetaMath") # Local saving

# EOS_TOKEN = tokenizer.eos_token  # End-of-Sequence token
# prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

# You are a helpful assistant to solve math problems step by step <|eot_id|><|start_header_id|>user<|end_header_id|>

# {}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

# {}"""

# def formatting_prompts_func(examples):
#     texts = []
#     labels = []
    
#     for instruction, responses, next_response, rating in zip(examples['instruction'], examples['responses'], examples['next_response'], examples['rating']):
#         # Combine all responses and the next response into a single string with newline separation
#         combined_responses = "\n".join(responses) + "\n" + next_response
        
#         # Format the text with the prompt template
#         text = prompt.format(instruction, combined_responses) 
#         texts.append(text)
#         labels.append(rating + 1)  # Convert ratings to labels by adding 1

#     # Tokenize all texts at once using the tokenizer
#     model_inputs = tokenizer(texts, padding="max_length", truncation=True, max_length=512)

#     # Add labels to the model inputs
#     model_inputs['labels'] = labels
    
#     return model_inputs

# from datasets import load_dataset

# # Load and preprocess the dataset
# dataset = load_dataset("Birchlabs/openai-prm800k-stepwise-critic", split='train')
# dataset = dataset.filter(lambda x: x['rating'] is not None)  # Filter entries without ratings
# dataset = dataset.map(formatting_prompts_func, batched=True)  # Apply the preprocessing function

# test_dataset = load_dataset("Birchlabs/openai-prm800k-stepwise-critic", split='test')
# test_dataset = test_dataset.filter(lambda x: x['rating'] is not None)  # Filter entries without ratings
# test_dataset = test_dataset.map(formatting_prompts_func, batched=True) 

# from transformers import TrainerCallback

# class PrintLossCallback(TrainerCallback):
#     """Print the training loss at each logging step."""
#     def on_log(self, args, state, control, logs=None, **kwargs):
#         if 'loss' in logs:
#             print(f"Step {state.global_step}: Loss {logs['loss']:.4f}")
            
# from transformers import TrainingArguments, Trainer
# from datasets import load_dataset
# from transformers import TrainingArguments, Trainer
# import numpy as np
# import evaluate
# import tqdm


# # Define the compute_metrics function
# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     predictions = np.argmax(predictions, axis=1)
#     accuracy = evaluate.load("accuracy")
#     return accuracy.compute(predictions=predictions, references=labels)



# # Define TrainingArguments
# training_args = TrainingArguments(
#     output_dir='./checkpoints/llama3-critic-1epoch',          # Output directory
#     num_train_epochs=1,              # Total number of training epochs
#     bf16=True,                       # Use bfloat16 for training
#     lr_scheduler_type='cosine',      # Learning rate scheduler type
#     per_device_train_batch_size=4,   # Batch size per device during training
#     gradient_accumulation_steps=4,       # Number of update steps to accumulate before performing a backward/update pass
#     warmup_ratio=0.03,                # Number of warmup steps for learning rate scheduler
#     learning_rate=2e-5,                   # Learning rate for the optimizer
#     weight_decay=0.01,               # Strength of weight decay
#     logging_dir='./logs',            # Directory for storing logs
#     logging_strategy='steps',
#     logging_steps=10,
#     evaluation_strategy="no",     # Evaluate each `logging_steps`
#     save_strategy="steps",           # Save a checkpoint at the end of each epoch
#     save_steps=10000,                   # Number of updates steps before saving
#     save_total_limit=2,              # Maximum number of checkpoints to keep
#     # load_best_model_at_end=True,     # Load the best model at the end of training based on metric (if evaluation is enabled)
#     report_to="wandb",               # Enable logging to W&B
# )

# # Initialize Trainer
# trainer = Trainer(
#     model=model,                    # The instantiated 🤗 Transformers model to be trained
#     args=training_args,             # Training arguments, defined above
#     train_dataset=dataset,          # Training dataset
#     # eval_dataset=test_dataset,           # Evaluation dataset
#     # compute_metrics=compute_metrics, # The callback that computes metrics of interest
#     # callbacks=[PrintLossCallback()]
# )

# # Train the model
# trainer.train()

# model.save_pretrained_merged("checkpoints/llama3-critic-1epoch", tokenizer, save_method = "merged_16bit",)