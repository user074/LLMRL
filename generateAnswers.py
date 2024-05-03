from transformers import AutoTokenizer
from unsloth import FastLanguageModel
from transformers import LlamaForSequenceClassification, AutoTokenizer, LlamaForCausalLM

import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #to cuda

generator, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "checkpoints/dpo-tinyllama-5-2", # "unsloth/tinyllama" for 16bit loading
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = False,
)
FastLanguageModel.for_inference(generator) # Enable native 2x faster inference
tokenizer.padding_side = "left" # Padding side for faster inference


EOS_TOKEN = tokenizer.eos_token  # End-of-Sequence token
prompt = """
### Input:
{}

### Response:
{}"""

def formatting_prompts_func(examples):
    texts = []
    final_answer = []
    
    for instruction, answer in zip(examples['question'], examples['answer']):
        # Combine all responses and the next response into a single string with newline separation
        extracted_answer = answer.split('### ')[1]
        final_answer.append(extracted_answer)
        
        # Format the text with the prompt template
        text = prompt.format(instruction, '')
        texts.append(text)

    # Tokenize all texts at once using the tokenizer
    # model_inputs = tokenizer(texts, padding="max_length", truncation=True, max_length=512)
    
    return {'input_text': texts, 'final_answer': final_answer}

from datasets import load_dataset

# Load and preprocess the dataset
dataset = load_dataset("gsm8k", 'main', split='test')
dataset = dataset.map(formatting_prompts_func, batched=True)  # Apply the preprocessing function



# generator = LlamaForCausalLM.from_pretrained("checkpoints/dpo-tinyllama-5-2")
# tokenizer = AutoTokenizer.from_pretrained("checkpoints/dpo-tinyllama-5-2", padding_side='right', use_fast = False)


# generator.to(device)

#set generator to inference mode
# generator.eval()
input_text = dataset['input_text']
import os
import json
import re
from tqdm import tqdm


def generate_answers(input_text, generator, tokenizer, output_dir, n_answers=100, batch_size=128):
    os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist
    
    # Get the existing JSON file numbers in the directory
    existing_files = os.listdir(output_dir)
    file_numbers = []
    for file_name in existing_files:
        match = re.match(r"generated_answers_(\d+)\.json", file_name)
        if match:
            file_numbers.append(int(match.group(1)))
    
    # Start generating new answers from the next available number
    start_number = max(file_numbers, default=0) + 1
    
    for n in tqdm(range(start_number, start_number + n_answers), desc="Generating Answer File", position=0):
        all_answers = []
        for i in tqdm(range(0, len(input_text), batch_size), desc="Generating Batch in the File", position=1, leave=True):
            batch_inputs = input_text[i:i+batch_size]
            batch_inputs = tokenizer(batch_inputs, return_tensors='pt', padding="max_length", truncation=True, max_length=512).to(device)
            outputs = generator.generate(
                **batch_inputs,
                max_new_tokens=256,
                use_cache=True,
                do_sample=True,
                temperature=0.5,
                top_k=40
            )
            answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_answers.extend(answers)
        
        # Generate a unique file name for each set of n_answers
        file_name = f"generated_answers_{n}.json"
        file_path = os.path.join(output_dir, file_name)
        
        # Save the answers to a JSON file for each set of n_answers in the specified directory
        with open(file_path, 'w') as file:
            json.dump(all_answers, file)
        
        print(f"Generated {len(all_answers)} answers for set {n}.")
    
    return all_answers

output_directory = "generated_answers_dpo"

with torch.no_grad():
    answers = generate_answers(input_text, generator, tokenizer, output_directory, n_answers=100)