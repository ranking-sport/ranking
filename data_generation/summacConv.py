import pandas as pd
import os
import json
from summac.model_summac import SummaCZS, SummaCConv

model_zs = SummaCZS(granularity="sentence", model_name="vitc", device="cuda") # If you have a GPU: switch to: device="cuda"
model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device="cuda", start_file="default", agg="mean")



SPORTS = ['Odi','T20i','Soccer','NBA','MLB']
id= 4
#output_dir = f"/home/../../Dataset/HallucinationSet_Trimmed/Llama-3.3-70B-Instruct/{SPORTS[id]}"
#output_dir = f"/home/../../Dataset/HallucinationSet_Trimmed/Mixtral-8x7B-Instruct-v0.1/{SPORTS[id]}"
output_dir = f"/home/../../Dataset/HallucinationSet_Trimmed/GPT4o/{SPORTS[id]}"

for idx,match_name in enumerate(os.listdir(output_dir)):

    print(match_name)

    knowledge_dir = os.path.join(output_dir,match_name,'articles')
    insight_dir = os.path.join(output_dir,match_name,'insights')

    files = [file.split('.')[0] for file in os.listdir(knowledge_dir)]
    scores_list = []
    files.sort()
    for file in files:
        
        knowledge_source = ""
        with open(f"{knowledge_dir}/{file}.txt") as f:
            knowledge_source = f.read()

        insight_source = ""

        with open(f"{insight_dir}/{file}.json") as f:
            insight_source = json.load(f)

        facts =[]
        print(file)
        try:
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
        generations = []

        #score_zs2 = model_zs.score([knowledge_source], facts)
        score_conv2 = model_conv.score([knowledge_source], facts)
        print("[Summary 2] SummacConv score: %.3f" % (score_conv2["scores"][0]))

        scores_list.append({"file":file,"summac-score":score_conv2["scores"][0]})

    df = pd.DataFrame(scores_list)

    df.to_csv(f"{output_dir}/{match_name}/summac_scores.csv", index=False)

