import re
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
#from trl import AutoModelForCausalLMWithValueHead
import numpy as np

# === Config ===
LLAMA_MODEL_PATH = "/scratch/nitishk_iitp/Model/Llama-3B-r/ppo_epoch4"
device = "cuda" if torch.cuda.is_available() else "cpu"

# === Load Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_PATH, padding_side="left", local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# === Load Config ===
config = AutoConfig.from_pretrained(LLAMA_MODEL_PATH, local_files_only=True)
config.rope_scaling = {"type": "dynamic", "factor": 8.0}

# === Load Model ===
model = AutoModelForCausalLM.from_pretrained(
    LLAMA_MODEL_PATH,
    config=config,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    local_files_only=True,
    #trust_remote_code=True,
)
model.eval()


# === Data points ===
data_points = [
    {"candidates": ["Eoin Morgan commented on England's underperformance, suggesting something is unsettled within the team.", "Chris Woakes expressed disappointment and mentioned the lack of confidence in the team.", "Jos Buttler acknowledged the team's poor performance and the need to play for Champions Trophy qualification.", "Jasprit Bumrah mentioned the team's enjoyment of the tournament and their practice in evening conditions.", "Rohit Sharma, adjudged the player of the match, emphasized the importance of his partnership with KL Rahul and taking the game deep.", "Rohit Sharma felt India were 30-40 runs short but was pleased with the overall performance.", "India's victory will galvanize the team as they head into the business end of the tournament.", "England's captain Jos Buttler will have much to contemplate as their chances of progression look even more scant.", "Rohit Sharma was named Player of the Match for his 87 runs.", "India earned 2 points from the match.", "Rohit Sharma expressed pride in the team's character and experience, noting the challenge of batting first and the effectiveness of their bowling attack.", "Jos Buttler expressed disappointment in England's performance, particularly their batting, despite a good start by the bowlers.", "India captain Rohit Sharma: 'This was a game that showed a lot of character in our squad. When times were tough, our experienced players stood up at the right time and fought for us.'", "England captain Jos Buttler: 'Very disappointing. At the halfway stage chasing 230 we fancied ourselves. But it's the same old story.'", "Player of the match Rohit Sharma: 'Looking at where we were after the first 10 overs of our batting, it was important to put on a partnership like myself and KL Rahul did. It was a challenging pitch to start with but it got easier the longer you spent in the middle. We are very happy with the performance.'"], "ranking": [12, 11, 10, 15, 8, 9, 5, 6, 14, 13, 4, 3, 2, 1, 7]},
{"candidates": ["India maintained their 100% record at the Cricket World Cup.", "England's World Cup defense is in a dire state, needing to win all remaining games and hoping for other results to go their way.", "The stadium was emptying behind the commentators as the match concluded.", "India are not mathematically into the semi-finals, nor are England mathematically out of the tournament.", "India remain the sole undefeated team in the tournament with this win.", "India displaced South Africa at the top of the standings with 12 points in six matches.", "England's next match is against Australia on November 4, which is a must-win for them.", "England's bowlers and fielders showed commitment and quality, with Chris Woakes delivering a seven-over opening spell that went for just 23 runs.", "Virat Kohli was dismissed for a nine-ball duck by David Willey.", "Rohit Sharma scored 87 runs off 101 balls, stabilizing India's innings.", "The match was held at Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow.", "The match was part of the ICC Cricket World Cup 2023.", "India stay unbeaten with six wins in as many games and regain the top spot.", "England remain at the bottom of the points table with their fifth defeat in six games.", "India's pacers Jasprit Bumrah and Mohammed Shami ripped through the England top-order in Lucknow.", "India's win leaves England rooted to the bottom of the standings.", "India's unchanged team took off at the back of Rohit Sharma's aggressive charge.", "England's spin-bowling all-rounders tried to rebuild but were undone by India's bowlers.", "England's last four innings have ended in scores of 215, 170, 156 and 129. Against Sri Lanka, they went from 45-0 to 156 all out. Today, it is 30-0 to 129."], "ranking": [5, 2, 18, 15, 1, 4, 12, 8, 11, 3, 19, 16, 6, 14, 7, 9, 10, 13, 17]},
{"candidates": ["Sherfane Rutherford scored 80 runs off 82 balls with 7 fours and 4 sixes.", "Wanindu Hasaranga took 4 wickets for 40 runs in 8 overs.", "Maheesh Theekshana was named Player of the Match.", "Sri Lanka beat West Indies by 5 wickets.", "Charith Asalanka scored an unbeaten 62 off 61 balls.", "Sri Lanka win by 5 wickets", "Roston Chase to Kamindu Mendis. Off break length ball, outside off stump on the back foot driving, well timed to deep backward point for 2 runs, fielded by Carty.", "Roston Chase to Charith Asalanka. Off break half volley, outside off stump on the front foot driving, well timed to deep cover for 1 run, fielded by Carty.", "Alick Athanaze to Kamindu Mendis. Off break length ball, outside off stump on the front foot defending, to short extra cover for no runs.", "Alick Athanaze to Kamindu Mendis. Off break length ball, outside off stump on the front foot pushing, to short extra cover for no runs, fielded by Walsh.", "Alick Athanaze to Charith Asalanka. Off break half volley, outside off stump on the front foot driving, to long off for 1 run, fielded by Rutherford.", "FOUR! Alick Athanaze to Charith Asalanka. Off break length ball, outside off stump on the back foot pulling, well timed in the air under control past deep mid wicket for 4 runs.", "Alick Athanaze to Kamindu Mendis. Off break back of a length, outside off stump on the back foot pulling, mis-timed in the air uncontrolled to deep mid wicket for 1 run, fielded by Carty.", "Alick Athanaze to Kamindu Mendis. Off break half volley, outside off stump on the front foot driving, to extra cover for no runs, fielded by Walsh.", "Roston Chase to Kamindu Mendis. Off break length ball, outside off stump on the back foot cutting, well timed to deep cover for 1 run, fielded by Carty."], "ranking": [5, 4, 3, 2, 6, 1, 8, 9, 11, 10, 7, 12, 13, 14, 15]},
{"candidates": ["The New York Knicks will take on the Milwaukee Bucks in NBA action at Madison Square Garden on Saturday, starting at 11:30am AEDT.", "Led by star players Jalen Brunson, Karl-Anthony Towns and Mikal Bridges, the Knicks are aiming to beat a Bucks team that includes Giannis Antetokounmpo, Damian Lillard and Bobby Portis.", "Stats Insider's predictive analytics model currently gives the Knicks a 68% chance of beating the Bucks at Madison Square Garden.", "The Knicks are listed as 6.5-point favourites against the Bucks, with odds of $1.91 available at Bet365.", "According to Stats Insider's model, the Bucks (+6.5) are predicted to cover the line 54% of the time, while the 223.5-point over/under is expected to go over 51% of the time.", "Stats Insider's predicted final score for Knicks vs Bucks at Madison Square Garden on Saturday is the Knicks winning 114-109.", "Jalen Brunson is expected to lead the Knicks with 36 points, 4 rebounds and 8 assists, while Giannis Antetokounmpo is projected to finish with 31 points, 11 rebounds and 9 assists for the Bucks.", "Milwaukee Bucks are looking to break their four-game road losing streak when they face the New York Knicks.", "The Knicks are favored to win with a -7.5 point spread according to BETMGM Sportsbook.", "The over/under for the game is set at 225.5 points.", "New York Knicks had a strong previous season with a 50-32 overall record and 35-17 in Eastern Conference games.", "Milwaukee Bucks finished the previous season with a 49-33 overall record and 34-18 in Eastern Conference action.", "Key injuries for the Knicks include Cameron Payne (day to day with a hamstring injury), Precious Achiuwa (out with a hamstring injury), and Mitchell Robinson (out with an ankle injury).", "Key injuries for the Bucks include Khris Middleton (out with an ankle injury) and Giannis Antetokounmpo (day to day with an adductor injury)."], "ranking": [12, 5, 8, 10, 9, 4, 3, 2, 11, 7, 6, 1, 13, 14]},
{"candidates": ["Memphis Grizzlies secured a 124-111 victory over Orlando Magic.", "Yuki Kawamura made his appearance in the last two minutes of the game but missed both 3-point attempts and had a turnover.", "Ja Morant played 25 minutes despite being listed as questionable with right thigh soreness.", "Morant scored 16 points and made 10 assists in the game.", "Ja Morant was twisting in the air and catching alley-oops.", "Memphis was blocking balls and intercepting passes.", "Jay Huff threw down multiple reverse dunks.", "The Grizzlies finished with 38 assists, their most since April 2023.", "Five players were tied with a team-high 11 points at the end of the third quarter.", "Jaren Jackson Jr. returned from a hamstring injury and finished with 13 points and four rebounds.", "Santi Aldama scored 22 points, leading the team in scoring for the second time in three games.", "Scotty Pippen Jr. finished with 11 points and 12 assists.", "Ja Morant finished with a double-double, scoring 16 points with 10 assists.", "Morant's energy was infectious in the Grizzlies' 124-111 home-opening win over the Orlando Magic.", "Jay Huff and Scotty Pippen Jr. had career nights thanks to Morant's involvement."], "ranking": [5, 12, 9, 8, 11, 10, 7, 4, 14, 6, 3, 2, 1, 13, 15]},
{
    "candidates": [
        "Virat Kohli scored a match-winning 97 under pressure.",
        "Steve Smith top-scored for Australia with 85 runs.",
        "India chased down 265 with four wickets remaining.",
        "Rohit Sharma contributed a quick 45 at the top.",
        "Pat Cummins took three wickets but was expensive.",
        "Australia collapsed in the middle overs, losing 4 wickets for 20 runs.",
        "Jasprit Bumrah bowled an excellent death over.",
        "Hardik Pandya's cameo of 28 from 12 turned the game.",
        "KL Rahul anchored the chase with a composed 50.",
        "India moved to the top of the table with the win.",
        "The match was played at Eden Gardens, Kolkata.",
        "Australia have now lost 3 matches in the tournament."
    ],
    "ranking": [2, 5, 1, 6, 10, 4, 3, 7, 8, 9, 12, 11]
},

{
    "candidates": [
        "Real Madrid secured a 2-1 comeback win over Bayern Munich.",
        "Karim Benzema scored both goals for Real Madrid.",
        "Manuel Neuer made 7 crucial saves for Bayern.",
        "The match saw a red card for Bayern defender Upamecano.",
        "Luka Modric controlled the midfield brilliantly.",
        "Vinicius Jr. won the penalty that led to the equalizer.",
        "Joshua Kimmich's goal gave Bayern an early lead.",
        "Toni Kroos was subbed off in the 70th minute.",
        "The game was played at Santiago Bernabéu.",
        "Real Madrid now qualify for the final.",
        "Bayern had 56% possession but lacked finishing."
    ],
    "ranking": [1, 2, 6, 7, 4, 3, 5, 10, 11, 8, 9]
},

{
    "candidates": [
        "LeBron James posted 30 points, 12 rebounds, and 8 assists.",
        "Anthony Davis added 25 points and 15 boards.",
        "The Lakers beat the Warriors 118-110 in Game 5.",
        "Stephen Curry had a quiet night with 22 points.",
        "Draymond Green fouled out in the fourth quarter.",
        "Austin Reaves scored 18 off the bench.",
        "The Lakers now lead the series 3-2.",
        "Klay Thompson was held to 5/17 shooting.",
        "Golden State shot just 29% from three-point range.",
        "Darvin Ham praised the team's defensive effort.",
        "The crowd at Crypto.com Arena was electric."
    ],
    "ranking": [1, 3, 2, 6, 8, 5, 4, 7, 9, 10, 11]
},

{
    "candidates": [
        "Pakistan defended 220 successfully in a low-scoring thriller.",
        "Shaheen Afridi took 4 wickets for just 22 runs.",
        "Shadab Khan's late strikes turned the game.",
        "Quinton de Kock scored 70 but lacked support.",
        "South Africa lost 5 wickets in the final 6 overs.",
        "Babar Azam top-scored with 58 runs.",
        "Mohammad Rizwan added a crucial 42-run partnership.",
        "South Africa dropped two catches during crucial phases.",
        "Pakistan broke their 3-match losing streak.",
        "The match was played in Cape Town.",
        "South Africa remain second on the points table."
    ],
    "ranking": [1, 2, 3, 5, 4, 6, 7, 9, 8, 11, 10]
},

{
    "candidates": [
        "Manchester City beat Liverpool 3-2 in a thrilling encounter.",
        "Erling Haaland scored twice and assisted once.",
        "Kevin De Bruyne orchestrated play with 3 key passes.",
        "Mohamed Salah netted Liverpool's second goal.",
        "Phil Foden opened the scoring in the 12th minute.",
        "Virgil van Dijk was solid in defense despite the loss.",
        "Alisson made 5 important saves for Liverpool.",
        "Pep Guardiola praised his team's composure.",
        "Jurgen Klopp admitted they were second-best today.",
        "Manchester City move to second place in the standings.",
        "The match was held at the Etihad Stadium."
    ],
    "ranking": [1, 2, 3, 5, 4, 8, 7, 6, 9, 10, 11]
}
]

# === NDCG@k Functions ===
def dcg_at_k(rel_scores, k):
    return sum((2**rel - 1) / np.log2(idx + 2) for idx, rel in enumerate(rel_scores[:k]))

def ndcg_at_k(pred_indices, gold_ranks, k):
    # Convert gold ranks (lower is better) to relevance scores (higher is better)
    # The maximum relevance score should be based on the number of candidates
    # The rank 1 item gets the highest relevance, rank N gets the lowest.
    # We are using 1-based ranks, so max_rel_val = N, and rank_i gives rel_score = N - (rank_i - 1)
    max_rel_val = len(gold_ranks)
    rel_scores_map = {idx: max_rel_val - (rank - 1) for idx, rank in enumerate(gold_ranks)}

    pred_rels = [rel_scores_map.get(i, 0) for i in pred_indices]

    # Ideal relevance scores: sort the actual relevance values present in the gold_ranks
    ideal_rels = sorted(rel_scores_map.values(), reverse=True)

    dcg = dcg_at_k(pred_rels, k)
    idcg = dcg_at_k(ideal_rels, k)
    return dcg / idcg if idcg > 0 else 0.0

def recall_at_k(pred_indices, gold_top_k_indices, k):
    top_k_preds = set(pred_indices[:k])
    # Ensure k is not zero to avoid division by zero
    if k == 0:
        return 0.0
    return len(top_k_preds.intersection(set(gold_top_k_indices))) / k

def log_metrics(store:list, ndcg2:float, ndcg5:float, ndcg10:float,
                rec2:float, rec5:float, rec10:float)->None:
    """Append one run’s six metrics to `store`."""
    store.append({
        "ndcg2": ndcg2, "ndcg5": ndcg5, "ndcg10": ndcg10,
        "rec2": rec2,    "rec5": rec5,    "rec10": rec10
    })

def top_k(store:list, k:int=5, sort_key:str="ndcg5")->list:
    """Return the k best runs sorted by the desired key (desc)."""
    return sorted(store, key=lambda d: d[sort_key], reverse=True)[:k]

def avg_metrics(runs:list)->dict:
    """Average every metric across the given runs."""
    if not runs:
        return {} # Return empty dict if no runs to average
    keys = runs[0].keys()
    return {k: sum(r[k] for r in runs)/len(runs) for k in keys}

def report_top3(store:list)->None:
    """Print top-5 runs (by ndcg5) plus their average block in specified Markdown format."""
    top3 = top_k(store, 5, "ndcg5")

    # Top 3 Samples Table
    print("\n---")
    print("### Top 5 Samples (Sorted by NDCG@5)")
    print("---")
    print("| Rank | NDCG@2    | NDCG@5    | NDCG@10   | Recall@2  | Recall@5  | Recall@10 |")
    print("|------|-----------|-----------|-----------|-----------|-----------|-----------|")
    for i, r in enumerate(top3, 1):
        print(f"| #{i:<4} | {r['ndcg2']:.4f}    | {r['ndcg5']:.4f}    | {r['ndcg10']:.4f}    | {r['rec2']:.4f}    | {r['rec5']:.4f}    | {r['rec10']:.4f}    |")

    # Averaged Metrics Table
    mean = avg_metrics(top3)
    print("\n---")
    print("### Averaged Metrics Over Top 5 Samples")
    print("---")
    print("| Metric    | Average Value |")
    print("|-----------|---------------|")
    for k in mean:
        # Align keys for consistent formatting
        print(f"| {k.replace('ndcg', 'NDCG@').replace('rec', 'Recall@'):<9} | {mean[k]:<13.4f} |")


# === Inference + Evaluation ===
metrics_log = []

for idx, sample in enumerate(data_points):
    sentences = sample["candidates"]
    gold_ranks = sample["ranking"] # This is list of 1-based ranks, where gold_ranks[i] is rank of sentence i
    n_sentences = len(sentences)

    # Calculate gold_top_k_indices for recall: 0-based indices of truly top-ranked items
    indexed_gold_ranks = list(enumerate(gold_ranks)) # [(0, rank0), (1, rank1), ...]
    sorted_gold_pairs_by_rank = sorted(indexed_gold_ranks, key=lambda x: x[1]) # Sort by rank
    gold_original_indices_sorted_by_rank = [pair[0] for pair in sorted_gold_pairs_by_rank] # Extract original 0-based indices

    # Ensure k does not exceed the number of available candidates for slicing
    gold_top_k_indices_rec2 = gold_original_indices_sorted_by_rank[:min(2, n_sentences)]
    gold_top_k_indices_rec5 = gold_original_indices_sorted_by_rank[:min(5, n_sentences)]
    gold_top_k_indices_rec10 = gold_original_indices_sorted_by_rank[:min(10, n_sentences)]


    prompt = (
        "You are an AI that ranks sports-related sentences based on importance using these criteria:\n"
        "1. Sports Relevance\n"
        "2. Emotional Intensity\n"
        "3. Sarcasm Presence\n"
        "4. Key People Mentions\n"
        "5. Buzzword Usage\n\n"
        f"Rank the following {n_sentences} sentences (0-based indices). Output ONLY numbers in order (best first), separated by spaces:\n\n"
        + "\n".join(f"{i}. {s}" for i, s in enumerate(sentences)) +
        "\n\nRanked indices:"
    )

    tokenized = tokenizer(
        [prompt],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096
    )
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    input_length = input_ids.shape[1]

    with torch.no_grad():
        output = model.pretrained_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=min(100, 15 * n_sentences), # Generate enough tokens for ranking
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.9,
            num_beams=1,
            temperature=0.8
            # Have tried with best config by trying again and again
        )

    generated_ids = output[:, input_length:]
    response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Robust parsing of generated indices
    resp_clean = re.sub(r'[^0-9 ]', '', response_text) # Remove non-numeric, non-space characters
    numbers = re.findall(r'\b\d+\b', resp_clean) # Find all sequences of digits as whole numbers
    indices = []
    for n_str in numbers:
        try:
            idx_n = int(n_str)
            if 0 <= idx_n < n_sentences and idx_n not in indices:
                indices.append(idx_n)
        except ValueError:
            pass # Should not happen with re.sub and re.findall as above, but good practice

    # If the model didn't generate all n_sentences indices, append missing ones
    if len(indices) < n_sentences:
        missing = [i for i in range(n_sentences) if i not in indices]
        indices.extend(missing[:n_sentences - len(indices)]) # Only add enough to reach n_sentences, if needed

    # Compute reward using NDCG@k
    ndcg2 = ndcg_at_k(indices, gold_ranks, k=2)
    ndcg5 = ndcg_at_k(indices, gold_ranks, k=5)
    ndcg10 = ndcg_at_k(indices, gold_ranks, k=10)

    # Use the correctly derived gold_top_k_indices for each k
    rec2 = recall_at_k(indices, gold_top_k_indices_rec2, k=2)
    rec5 = recall_at_k(indices, gold_top_k_indices_rec5, k=5)
    rec10 = recall_at_k(indices, gold_top_k_indices_rec10, k=10)

    log_metrics(metrics_log, ndcg2, ndcg5, ndcg10, rec2,  rec5,  rec10)
    # print("Metrics we got for current sample",metrics_log[-1]) # Commented out for cleaner final output

# === Generate final report ===
report_top3(metrics_log)
