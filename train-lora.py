from transformers import LlamaForSequenceClassification, AutoTokenizer, LlamaForCausalLM
import torch
model = LlamaForSequenceClassification.from_pretrained('checkpoints/tinyLlama-critic')
# model = LlamaForCausalLM.from_pretrained("checkpoints/tinyLlama-GSM8K-10epochs")

tokenizer = AutoTokenizer.from_pretrained("checkpoints/tinyLlama-critic", padding_side='right', use_fast          = False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#to cuda
model.to(device)

#LoRA config
from peft import (
        get_peft_model, 
        prepare_model_for_kbit_training, 
        LoraConfig,
        TaskType
    )

config = LoraConfig(
r= 32,
lora_alpha          = 32,
target_modules      = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
bias                = 'none',
task_type           = TaskType.SEQ_CLS,
modules_to_save     = ["score"],
)

model = get_peft_model(model, config)
model.print_trainable_parameters()

from transformers import LlamaForSequenceClassification, AutoTokenizer, LlamaForCausalLM
import torch
import wandb
import os

wandb.login()
os.environ['WANDB_PROJECT'] = 'LLMRL'

EOS_TOKEN = tokenizer.eos_token  # End-of-Sequence token
prompt = """
### Input:
{}

### Response:
{}"""

def formatting_prompts_func(examples):
    texts = []
    labels = []
    
    for instruction, responses, next_response, rating in zip(examples['instruction'], examples['responses'], examples['next_response'], examples['rating']):
        # Combine all responses and the next response into a single string with newline separation
        combined_responses = "\n".join(responses) + "\n" + next_response
        
        # Format the text with the prompt template
        text = prompt.format(instruction, combined_responses) + EOS_TOKEN
        texts.append(text)
        labels.append(rating + 1)  # Convert ratings to labels by adding 1

    # Tokenize all texts at once using the tokenizer
    model_inputs = tokenizer(texts, padding="max_length", truncation=True, max_length=512)

    # Add labels to the model inputs
    model_inputs['labels'] = labels
    
    return model_inputs

from datasets import load_dataset

# Load and preprocess the dataset
dataset = load_dataset("Birchlabs/openai-prm800k-stepwise-critic", split='train')
dataset = dataset.filter(lambda x: x['rating'] is not None)  # Filter entries without ratings
dataset = dataset.map(formatting_prompts_func, batched=True)  # Apply the preprocessing function

from transformers import TrainerCallback

class PrintLossCallback(TrainerCallback):
    """Print the training loss at each logging step."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if 'loss' in logs:
            print(f"Step {state.global_step}: Loss {logs['loss']:.4f}")
            
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate
import tqdm


# Define the compute_metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = evaluate.load("accuracy")
    return accuracy.compute(predictions=predictions, references=labels)



# Define TrainingArguments
training_args = TrainingArguments(
    output_dir='./checkpoints/tinyLlama-critic-lora-2epoch',          # Output directory
    num_train_epochs=1,              # Total number of training epochs
    per_device_train_batch_size=4,   # Batch size per device during training
    gradient_accumulation_steps=4,       # Number of update steps to accumulate before performing a backward/update pass
    warmup_ratio=0.03,                # Number of warmup steps for learning rate scheduler
    learning_rate=4e-6,                   # Learning rate for the optimizer
    weight_decay=0.01,               # Strength of weight decay
    logging_dir='./logs',            # Directory for storing logs
    logging_strategy='steps',
    logging_steps=10,
    evaluation_strategy="no",     # Evaluate each `logging_steps`
    save_strategy="steps",           # Save a checkpoint at the end of each epoch
    save_steps=10000,                   # Number of updates steps before saving
    save_total_limit=1,              # Maximum number of checkpoints to keep
    # load_best_model_at_end=True,     # Load the best model at the end of training based on metric (if evaluation is enabled)
    report_to="wandb",               # Enable logging to W&B
)

# Initialize Trainer
trainer = Trainer(
    model=model,                    # The instantiated 🤗 Transformers model to be trained
    args=training_args,             # Training arguments, defined above
    train_dataset=dataset,          # Training dataset
    # eval_dataset=dataset,           # Evaluation dataset
    # compute_metrics=compute_metrics, # The callback that computes metrics of interest
    # callbacks=[PrintLossCallback()]
)

# Train the model
trainer.train()


# Assuming the Trainer has completed training
model_path = "./checkpoints/tinyLlama-critic-lora-2epoch"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)