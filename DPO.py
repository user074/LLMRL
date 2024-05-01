import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from unsloth import FastLanguageModel
LORA_RANK = 64
LORA_ALPHA = 128



model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "checkpoints/tinyLlama-GSM8K-10epochs", # "unsloth/tinyllama" for 16bit loading
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = False,
)

model = FastLanguageModel.get_peft_model(
    model,
    target_modules=[
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",  # attention (self_attn)
        "gate_proj",
        "down_proj",
        "up_proj",  # FFN (mlp)
    ],
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=False,
)

critic, critic_tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/home/jianingqi/LLMRL/checkpoints/llama3-8b-critic-lora-4-29", # "unsloth/tinyllama" for 16bit loading
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = False,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)
FastLanguageModel.for_inference(critic) # Enable native 2x faster inference
critic_tokenizer.padding_side = "left" # Padding side for faster inference

from tqdm import tqdm
from transformers.utils import logging
logging.set_verbosity_error()

def generate_answers(input_text, generator, tokenizer, n_answers=2, batch_size=128):    
    all_answers_list = []
    for n in tqdm(range(0, n_answers), desc=" Answer Set", position=0):
        all_answers = []
        for i in tqdm(range(0, len(input_text), batch_size), desc="Answers in Answer Set", position=1, leave=True):
            batch_inputs = input_text[i:i+batch_size]
            batch_inputs = tokenizer(batch_inputs, return_tensors='pt', padding="max_length", truncation=True, max_length=256).to(device)
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
        
        print(f"Generated {len(all_answers)} answers for set {n}.")
        all_answers_list.append(all_answers)
    
    return all_answers_list

def compute_probabilities(all_answers, critic_tokenizer, critic, batch_size=32, is_llama = True):
    answers_prob = [[] for _ in range(len(all_answers[0]))]
    
    good_token = ' +'
    bad_token = '-'
    step_tag = ' ки'

    candidate_tokens = critic_tokenizer.encode(f"{good_token} {bad_token}")[1:] # [648, 387]
    step_tag_id = critic_tokenizer.encode(f"{step_tag}")[-1] # 12902
    # print(candidate_tokens)
    # print(step_tag_id)

    with torch.no_grad():
        for answers in tqdm(all_answers, desc="Processing rewards", position=0):
            results = []
            response_counts = []
            for answer in answers:
                if '### Response:' in answer:
                    result = answer.split('### Response:')[0]
                    responses = answer.split('### Response:\n')[1].split('\n')
                    num_responses = len(responses)
                    response_counts.append(num_responses)
                elif '?' in answer:
                    # print(answer)
                    result = answer.split('?')[0] + '?'
                    responses = answer.split('?')[1].split('\n')
                    num_responses = len(responses)
                    response_counts.append(num_responses)
                elif '####' in answer:
                    result = answer.split('####')[0]
                    responses = answer.split('####')[1].split('\n')
                    responses[0] = '####' + responses[0]
                    num_responses = len(responses)
                    response_counts.append(num_responses)
                else:
                    result = answer
                    responses = ['']
                    num_responses = len(responses)
                    response_counts.append(num_responses)
                    
                     
                for response in responses:
                    result += response + " ки \n"
                results.append(result)
                                
            correct_probabilities = []
            for i in tqdm(range(0, len(results), batch_size), desc="Processing batch",position=1,  leave=True):
                batch_results = results[i:i+batch_size]
                
                inputs = critic_tokenizer(batch_results, padding="max_length", truncation=True, max_length=512, return_tensors="pt").to("cuda")
                logits = critic(**inputs).logits[:,:,candidate_tokens]
                scores = logits.softmax(dim=-1)[:,:,0] 
                step_scores = scores[inputs['input_ids'] == step_tag_id]
                correct_probabilities.extend(step_scores.tolist())
            
            # response_counts = []
            # for answer in answers:
            #     num_responses = len(answer.split('### Response:\n')[1].split('\n'))
            #     response_counts.append(num_responses)
            
            probability_index = 0
            for i, count in enumerate(response_counts):
                answer_probs = correct_probabilities[probability_index:probability_index+count]
                if answer_probs:
                    # answer_prob = min(answer_probs)
                    answer_prob = torch.tensor(answer_probs).prod().item()
                    answers_prob[i].append(answer_prob)
                else:
                    print('len of prob')
                    print(len(correct_probabilities))
                    print('len of responses')
                    print(sum(response_counts))
                    print('There is a length mismatch')
                    print('-----', i)
                    print(answers[i])
                    answers_prob[i].append(0.0)
                probability_index += count
    
    return answers_prob

def select_high_low_probability_answers(all_answers, answers_prob):
    highest_probability_answers = []
    lowest_probability_answers = []
    extracted_answers = [[] for _ in range(len(all_answers[0]))]
    for answers in all_answers:
        for i, answer in enumerate(answers):
            extracted_answers[i].append(answer)
                
    for i, question_answers in enumerate(extracted_answers):
        question_probs = answers_prob[i]
        if question_probs:
            max_prob_index = question_probs.index(max(question_probs))
            highest_probability_answer = question_answers[max_prob_index]
            min_prob_index = question_probs.index(min(question_probs))
            lowest_probability_answer = question_answers[min_prob_index]
        else:
            highest_probability_answer = ""
            lowest_probability_answer = ""
        highest_probability_answers.append(highest_probability_answer)
        lowest_probability_answers.append(lowest_probability_answer)
    return highest_probability_answers, lowest_probability_answers

def rollout_to_DPO_dataset(dataset, model, tokenizer, critic_tokenizer, critic, device = "cuda"):
    model.to(device)
    print('Rolling Out from model')
    with torch.no_grad():
        answers = generate_answers(dataset['prompt'], model, tokenizer, n_answers=2)
    print('Roll out completed')
    print('Starting to compute rewards')
    answers_prob = compute_probabilities(answers, critic_tokenizer, critic)
    highest_probability_answers, lowest_probability_answers = select_high_low_probability_answers(answers, answers_prob)

    # Add the "chosen" column
    epoch_dataset = dataset
    epoch_dataset = epoch_dataset.add_column("chosen", highest_probability_answers)
    # Add the "rejected" column
    epoch_dataset = epoch_dataset.add_column("rejected", lowest_probability_answers)

    # Compute rewards based on answer probabilities
    rewards = []
    for probs in answers_prob:
        if probs:
            max_prob = max(probs)
            min_prob = min(probs)
            rewards.append([max_prob, min_prob])
        else:
            rewards.append([0.0, 0.0])

    return epoch_dataset, rewards

# One must patch the DPO Trainer first!
from unsloth import PatchDPOTrainer
PatchDPOTrainer()

prompt = """
### Input:
{}

### Response:
"""
from datasets import load_dataset

dataset = load_dataset("gsm8k", 'main', split='train')
dataset = dataset.rename_column('question', 'prompt')

dataset = dataset.remove_columns('answer')

from transformers import TrainingArguments, get_scheduler
from trl import DPOTrainer
from torch.optim import AdamW

epochs = 10
base_lr = 4e-6
total_steps = len(dataset) * epochs

optimizer = AdamW(model.parameters(), lr=base_lr)

import wandb
import os
wandb.login()
os.environ['WANDB_PROJECT'] = 'LLMRL'

training_args= TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_ratio=0.1,
            num_train_epochs=1,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.0,
            lr_scheduler_type="constant",  # Set the scheduler type to "constant"
            seed=42,
            output_dir="checkpoints/dpo-tinyllama-5-1",
        )

for epoch in tqdm(range(epochs)):
    epoch_dataset, rewards = rollout_to_DPO_dataset(dataset, model, tokenizer, critic_tokenizer, critic)
    model.train()
    
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        beta=0.1,
        train_dataset=epoch_dataset,
        tokenizer=tokenizer,
        max_length=512,
        max_prompt_length=256,
    )
    
    # Create the learning rate scheduler
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=int(total_steps * 0.1),  # 10% of total steps for warmup
        num_training_steps=total_steps,
    )
    
    # Train the model for one epoch
    dpo_trainer.train()
    
    # Update the learning rate for the next epoch
    lr_scheduler.step()
    
model.save_pretrained("checkpoints/dpo-tinyllama-5-1") # Local saving
tokenizer.save_pretrained("checkpoints/dpo-tinyllama-5-1")