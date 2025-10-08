import os
import pandas as pd
import json
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,AutoModelForSequenceClassification
import torch
import gc
import transformers
import re
import ast

import insightsPrompt as prompts

import time

start_time = time.time()
# from logger_setup import logger

# from huggingface_hub import HfFolder # Save your token

#model = '/scratch/../models/Qwen2.5-14B-Instruct'
#model = '/scratch/../models/Llama-3.3-70B-Instruct'
#model = '/scratch/../models/Mixtral-8x7B-Instruct-v0.1'
model = '/scratch/../models/DeepSeek-R1-Distill-Llama-70B'

model_name = model.split('/')[-1]
MODEL_NAME = model

bnb_config = BitsAndBytesConfig(
     load_in_4bit=True,
     bnb_4bit_use_double_quant=True,   
     bnb_4bit_quant_type="nf4",
     bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    local_files_only=True,
    quantization_config=bnb_config
)

#tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# model = torch.nn.DataParallel(model)
pipeline = transformers.pipeline(
      "text-generation",
      model=model,
      model_kwargs={"torch_dtype": torch.bfloat16},
      device_map='auto',
      tokenizer=tokenizer,
)

# Replace 'your-api-key' with your actual OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# print(OPENAI_API_KEY)

def createSystemPrompt(match_name,sportId):
    if sportId==2:
         return prompts.mlbPrompt(match_name)
    elif sportId==0:
        return prompts.soccerPrompt(match_name)
    elif sportId==1:
        return prompts.nbaPrompt(match_name)
    return prompts.cricketPrompt(match_name)

def get_llm_output(content,match_name,sportId):
  
  messages = [
      {"role": "system", "content": f"{createSystemPrompt(match_name,sportId)}"},
      {"role": "user", "content":f"{content}" }
  ]


  # terminators = [
  #     pipeline.tokenizer.eos_token_id,
  #     pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
  # ]


  with torch.no_grad():
    outputs =pipeline(messages,
        max_new_tokens=1000,
        # eos_token_id=terminators,
        do_sample=True,
        temperature=0.2,
        top_p=0.8,
    )

  return outputs[0]["generated_text"][-1]['content']

SPORTS = ['Soccer','NBA','MLB', 'T20i','Odi']
SPORT_INDEX = 4

input_dir = f"../Dataset/ValidatedData_Trimmed_200/{SPORTS[SPORT_INDEX]}"
output_dir = f"../Dataset/Insights_Trimmed/{model_name}/{SPORTS[SPORT_INDEX]}"



def extract_json(response):
    try:
        # Regex to match JSON object
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        
        if json_match:
            json_data = json_match.group(0)
            json_data = re.sub(r",\s*([\]}])", r"\1", json_data)
            python_dict = ast.literal_eval(json_data)

            return python_dict, None
        else:
            return None, "No JSON object found in the response."
    except Exception as e:
        return None, f"JSON decode error: {e}"
    

total = 0
irrelevant = 0
jsonerror = 0


logger = open(f'../Logs/insights_log_{model_name}_{SPORTS[SPORT_INDEX]}.txt', 'a')



# Loop processing each articles and storing thiei insights
             
        
print(f"Total Files: {total}", file=logger)
print(f"Irrelevant Files: {irrelevant}", file=logger)
print(f"json error Files: {jsonerror}", file=logger)
# print(f"Total Files: {total}")
# print(f"Irrelevant Files: {irrelevant}")
# print(f"json error Files: {jsonerror}")
print("completed")
end_time = time.time()

print(f"Execution time: {end_time - start_time:.6f} seconds\n",file=logger)
logger.close()
