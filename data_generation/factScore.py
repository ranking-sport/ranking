import os
import json
import pandas as pd
from openai import OpenAI
import numpy as np

OPENAI_API_KEY ="API KEY"

def get_score(facts, generations, gamma):
  penalty = 1.0 if len(generations)>gamma else np.exp(1-gamma/len(facts))
  scores = []
  abstain_count = 0
  for generated_response in generations:
    if generated_response == True:
      scores.append(1*penalty)
    elif generated_response == False:
      scores.append(0)
    else:
      scores.append(0)
      abstain_count += 1

  average_score = sum(scores)/len(generations)
  respond_ratio = (len(facts) - abstain_count)/len(facts)

  return scores, average_score, respond_ratio


def createSystemPrompt():
  prompt = f"""
  You are an expert on hallucination detection and you are given a list of 
  statements and a knowledge source, determine for each statement whether it is 
  factually and meaningfully correct and is part of the provided knowledge source.
  Return a list of one-word responses: True or False

  Output formt:
  [True, False, True]
  """
  return prompt

def createUserPrompt(fact, knowledge_source):
  prompt = f"""
  Task: Determine for each fact whether it is True or False based on the knowledge source.

  Statements: {fact}
  Knowledge Source: {knowledge_source}
  """
  return prompt


def get_gpt_response(fact,knowledge_source):
    client = OpenAI(api_key=OPENAI_API_KEY)
    completion = client.chat.completions.create(
      model="gpt-4o",
      messages = [
      {"role": "system", "content": createSystemPrompt()},
      {"role": "user", "content":createUserPrompt(fact,knowledge_source) }
    ]
    )

    return completion.choices[0].message.content


SPORTS = ['Odi','T20i','Soccer','NBA','MLB']
id=4
MODELS = ['Llama-3.3-70B-Instruct','Mixtral-8x7B-Instruct-v0.1','Qwen2.5-72B-Instruct',]
mid =2

output_dir = f"../Dataset/HallucinationSet_Trimmed/{MODELS[mid]}/{SPORTS[id]}"

def extract_list_from_string(input_string):
    # Find the start and end of the list in the input string
    start = input_string.find('[')
    end = input_string.find(']', start)

    # Extract the list part from the input string
    list_string = input_string[start:end+1]

    # Convert the list string to a Python list
    extracted_list = eval(list_string)

    return extracted_list



for idx,match_name in enumerate(os.listdir(output_dir)):

    print(idx,match_name)

    knowledge_dir = os.path.join(output_dir,match_name,'articles')
    insight_dir = os.path.join(output_dir,match_name,'insights')

    files = [file.split('.')[0] for file in os.listdir(knowledge_dir)]
    scores_list = []
    for file in files:

        knowledge_source = ""
        with open(f"{knowledge_dir}/{file}.txt",encoding='utf-8') as f:
            knowledge_source = f.read()

        insight_source = ""
        try:
            with open(f"{insight_dir}/{file}.json",encoding='utf-8') as f:
                insight_source = json.load(f)

            facts =[]
            print(file)
        
            for key in insight_source:

                if key == "Relevancy":
                    continue
                insights = insight_source[key]

                for fact in insights:
                    facts.append(fact)
        except Exception as e:
            print("json error")

        if len(facts) == 0:
            continue
        
        try:
            generations =get_gpt_response(facts,knowledge_source)
            generations = extract_list_from_string(generations)

            print(generations)
            # print(len(facts),len(generations))
            # print(facts)

            scores, average_score, respond_ratio = get_score(facts, generations, 1)
            print(scores,average_score)
            scores_list.append({"file":file,"fact-score":average_score,"insights":len(generations)})
        except Exception as e:
            print(e)
    df = pd.DataFrame(scores_list)

    df.to_csv(f"{output_dir}/{match_name}/fact_scores.csv", index=False)