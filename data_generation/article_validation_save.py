import pandas as pd
import os
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,AutoModelForSequenceClassification,pipeline
import torch
import gc
import transformers
import shutil

import time

start_time = time.time()

#LOGGER = open('valid_save_log_mlb.txt','w')

model = '/scratch/../models/Qwen2.5-32B-Instruct'


MODEL_NAME = model

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    local_files_only=True,
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

#tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

pipeline = transformers.pipeline(
      "text-generation",
      model=model,
      model_kwargs={"torch_dtype": torch.bfloat16},
      device_map='auto',
      tokenizer=tokenizer,
)

def createUserPrompt(match_name):
    prompt = f""" You are the validator for sports articles and must verify if the article 
    is about the given match information, ensuring it is not confused with other matches between
    the same teams. Use the match information to accurately confirm the article's relevance. 
    Only validate articles written in English; articles in other languages are irrelevant. 
    An article is considered relevant only if it contains valid content discussing the match. 
    Articles containing only a URL or lacking substantive text should be marked as irrelevant, 
    even if the URL is associated with the article. Return only one-word responses: 'relevant' 
    if the article has valid content about the given match, 'irrelevant' otherwise. 
    Do not return anything other than the one-word answer."""

    return prompt

def createPrompt(match_name,content):
    prompt = f""""
    Task: Validate whether the content of the provided article is relevant to the given match information.

    Match Information: {match_name}
    Article : {content}
    """

    return prompt

def get_llm_output(content,match_name):
  
  messages = [
      {"role": "system", "content": createUserPrompt(match_name)},
      {"role": "user", "content":createPrompt(match_name,content) }
  ]


# terminators = [
#      pipeline.tokenizer.eos_token_id,
#     pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
#  ]


  with torch.no_grad():
    outputs = pipeline(
        messages,
        max_new_tokens=64,
        # eos_token_id=terminators,
        do_sample=True,
        temperature=0.1,
        top_p=0.95,
    )
  # print('---- ',outputs[0]['generated_text'][-1]['content'])
  return outputs[0]["generated_text"][-1]['content']

def process_files_in_order(folder_path): 
    before_files = [] 
    after_files = [] 
    # Collect files 
    for file_name in os.listdir(folder_path): 
        if file_name.startswith('before-'): 
            before_files.append(file_name) 
        elif file_name.startswith('after-'): 
            after_files.append(file_name)

    return before_files,after_files

SPORTS = ['Soccer','Odi','NBA','MLB', 'T20i']
SPORT_INDEX = 3
path = f'../Dataset/Scrapped_Trimmed/{SPORTS[SPORT_INDEX]}'
out_path = f'../Dataset/ValidatedData_Trimmed/{SPORTS[SPORT_INDEX]}'
LOGGER = open(f'../Logs/valid_save_log_{SPORTS[SPORT_INDEX]}.txt','a')
TH = 2
maxTH = 10


for idx,match_name in enumerate( os.listdir(path)):
    
    match_folder = os.path.join(path,match_name)
    print(idx,match_name,file=LOGGER)
    LOGGER.flush()

    before_files,after_files = process_files_in_order(match_folder)
    
    if len(before_files) < TH or len(after_files) < TH:
        print('     Not enough files continue to next',file=LOGGER)
        LOGGER.flush()
        continue

    precnt = 0
    os.makedirs(f'{out_path}/{match_name}',exist_ok=True)
    
    # Loop processing and saving pre-game articles

    if precnt < TH:
        print("     valid pre game count less than TH, deleting directory and continue to next",file=LOGGER)
        LOGGER.flush()
        shutil.rmtree(f'{out_path}/{match_name}')
        continue
    
    postcnt = 0

    # Loop processing post-game articles

    if postcnt < TH:
        print("     valid post game count less than TH, deleting directory",file=LOGGER)
        LOGGER.flush()
        shutil.rmtree(f'{out_path}/{match_name}')

     
end_time = time.time()

print(f"Execution time: {end_time - start_time:.6f} seconds\n",file=LOGGER)
LOGGER.flush()
LOGGER.close()

