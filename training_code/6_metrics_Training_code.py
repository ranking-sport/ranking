import os
import re
import json
import torch
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from trl import AutoModelForCausalLMWithValueHead, PPOTrainer, PPOConfig
import spacy
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import pandas as pd
import logging
import faiss # Import faiss for semantic scoring
# from scipy.stats import kendalltau # Removed kendalltau
import torch.nn.functional as F # For sigmoid
from transformers import AutoModelForSeq2SeqLM, T5Tokenizer # For T5 sarcasm model

# Configure logging to suppress INFO and WARNING messages, only show ERRORs if any
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Essential Setup & Feature Definitions ---

# Download NLTK data (if not already present)
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except nltk.downloader.DownloadError:
    nltk.download("vader_lexicon", quiet=True)

# Load SpaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# --- Load external data (CSV files) ---
try:
    sports_keywords_df = pd.read_csv("sports_keywords.csv")
    sports_keywords_dict = dict(zip(sports_keywords_df['Word'].str.lower(), sports_keywords_df['AverageScore']))
except FileNotFoundError:
    sports_keywords_dict = {
        "champion": 1.0, "win": 0.9, "record": 0.8, "goal": 0.7, "title": 0.95,
        "trophy": 0.85, "victory": 0.9, "cup": 0.75, "medal": 0.7, "stadium": 0.5,
        "athlete": 0.6, "coach": 0.55, "game": 0.6, "match": 0.6,
        "runs": 0.6, "wickets": 0.7, "innings": 0.5, "test": 0.4,
        "championship": 0.9, "final": 0.8, "world cup": 0.95, "odi": 0.7, "t20": 0.7
    }
except KeyError as e:
    sports_keywords_dict = {
        "champion": 1.0, "win": 0.9, "record": 0.8, "goal": 0.7, "title": 0.95,
        "trophy": 0.85, "victory": 0.9, "cup": 0.75, "medal": 0.7, "stadium": 0.5,
        "athlete": 0.6, "coach": 0.55, "game": 0.6, "match": 0.6,
        "runs": 0.6, "wickets": 0.7, "innings": 0.5, "test": 0.4,
        "championship": 0.9, "final": 0.8, "world cup": 0.95, "odi": 0.7, "t20": 0.7
    }

try:
    famous_people_df = pd.read_csv("processed_persons.csv")
    famous_people_dict = dict(zip(famous_people_df['name'].str.lower(), famous_people_df['hpi']))
except FileNotFoundError:
    famous_people_dict = {
        "virat kohli": 850, "sachin tendulkar": 900, "ms dhoni": 800,
        "shubman gill": 250, "ravindra jadeja": 300, "pat cummins": 400,
        "cristiano ronaldo": 950, "lionel messi": 980, "serena williams": 880,
        "novak djokovic": 870, "stephen curry": 890, "michael jordan": 990,
        "max verstappen": 700, "fernando alonso": 650, "charles leclerc": 500,
        "temba bavuma": 100, "mushfiqur rahim": 120
    }
except KeyError as e:
    famous_people_dict = {
        "virat kohli": 850, "sachin tendulkar": 900, "ms dhoni": 800,
        "shubman gill": 250, "ravindra jadeja": 300, "pat cummins": 400,
        "cristiano ronaldo": 950, "lionel messi": 980, "serena williams": 880,
        "novak djokovic": 870, "stephen curry": 890, "michael jordan": 990,
        "max verstappen": 700, "fernando alonso": 650, "charles leclerc": 500,
        "temba bavuma": 100, "mushfiqur rahim": 120
    }

# --- Global Feature Extraction Models/Tools ---

# For Semantic Score
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
def generate_verb_dictionary():
    sports_verbs = {
        "winning": 1.0, "win": 1.0, "won": 1.0,
        "scoring": 0.9, "score": 0.9, "scored": 0.9,
        "competing": 0.85, "compete": 0.85, "competed": 0.85,
        "training": 0.8, "train": 0.8, "trained": 0.8,
        "dribbling": 0.75, "dribble": 0.75,
        "shooting": 0.78, "shoot": 0.78, "shot": 0.78,
        "defending": 0.7, "defend": 0.7, "defended": 0.7,
        "serve": 0.6, "served": 0.6,
        "beat": 0.9, "defeated": 0.95, "took": 0.7, "taking": 0.7,
        "played": 0.6, "play": 0.6, "runs": 0.6, "ran": 0.6, "hit": 0.65
    }
    other_verbs = {
        "drinking": 0.4, "talking": 0.3, "eating": 0.35, "sleeping": 0.2,
        "reading": 0.25, "writing": 0.3, "listening": 0.28
    }
    verb_dict = {}
    verb_dict.update(sports_verbs)
    verb_dict.update(other_verbs)
    return verb_dict

def get_verb_embeddings(verb_dict, model):
    verbs_list = list(verb_dict.keys())
    embeddings = model.encode(verbs_list, convert_to_numpy=True)
    return verbs_list, embeddings

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

verb_dict = generate_verb_dictionary()
verbs, verb_embeddings = get_verb_embeddings(verb_dict, embedding_model)
faiss_index = create_faiss_index(verb_embeddings)

# For Sentiment Score
sentiment_tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
sentiment_model = AutoModelForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions")

# For TFIDF Score
tfidf_vectorizer = TfidfVectorizer()

# For Sarcasm Score
t5_sarcasm_tokenizer = T5Tokenizer.from_pretrained("mrm8488/t5-base-finetuned-sarcasm-twitter")
t5_sarcasm_model = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/t5-base-finetuned-sarcasm-twitter")
vader_sia = SentimentIntensityAnalyzer()

# -------------------- ScoreNet Model --------------------
class ScoreNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.randn(6) * 0.01)

    def forward(self, x):
        weights = torch.softmax(self.logits, dim=0)
        return (x * weights).sum(dim=-1)

# -------------------- Feature Computation --------------------
def normalize_feature(value, max_val, min_val=0.0):
    if max_val == min_val:
        return 0.0
    return float((value - min_val) / (max_val - min_val))

def calculate_semantic_score(sentence_text, verb_dict, model, index, verbs_list, nlp_model, k=1):
    if not isinstance(sentence_text, str):
        sentence_text = str(sentence_text)
    doc = nlp_model(sentence_text)
    token_scores = []
    for token in doc:
        if token.ent_type_:
            continue
        token_lower = token.text.lower()
        if token_lower in verb_dict:
            token_scores.append(verb_dict[token_lower])
        else:
            token_embedding = model.encode([token_lower], convert_to_numpy=True)
            distances, indices = index.search(token_embedding, k)
            neighbor_scores = []
            for i in range(k):
                neighbor_index = indices[0][i]
                neighbor_verb = verbs_list[neighbor_index]
                neighbor_value = verb_dict[neighbor_verb]
                neighbor_scores.append(neighbor_value)
            avg_neighbor_score = sum(neighbor_scores) / len(neighbor_scores) if neighbor_scores else 0.0
            token_scores.append(avg_neighbor_score)
    if token_scores:
        overall_score = sum(token_scores) / len(token_scores)
    else:
        overall_score = 0.0
    return overall_score

def calculate_sentiment_score(text):
    inputs = sentiment_tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    current_device = sentiment_model.device
    inputs = {k: v.to(current_device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
    scores = torch.nn.functional.sigmoid(outputs.logits)
    intensity_score = torch.max(scores[0]).item()
    return intensity_score

def calculate_tfidf_raw_score(text, vectorizer):
    if not hasattr(vectorizer, 'vocabulary_') or not vectorizer.vocabulary_:
        return 0.0
    tfidf_matrix = vectorizer.transform([text])
    raw_sum = tfidf_matrix.sum()
    return raw_sum

def calculate_buzzword_raw_score(sentence_text):
    score = sum(sports_keywords_dict.get(word, 0) for word in sentence_text.lower().split())
    return score

def calculate_person_raw_score(sentence_text):
    doc = nlp(sentence_text)
    hpi_values = []
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            hpi = famous_people_dict.get(ent.text.lower(), 0)
            hpi_values.append(hpi)
    return float(max(hpi_values)) if hpi_values else 0.0

# Sarcasm Score (modified to always calculate VADER-based score, clamped 0-1)
def detect_sarcasm_t5(text):
    """
    Detects if a sentence is sarcastic using a pre-trained T5 model.
    Returns True if sarcastic, False otherwise.
    This function's output will still be used for verbose logging,
    but the main sarcasm_score will calculate VADER divergence always.
    """
    input_ids = t5_sarcasm_tokenizer.encode(text + '</s>', return_tensors='pt')
    current_device = t5_sarcasm_model.device
    input_ids = input_ids.to(current_device)

    with torch.no_grad():
        output = t5_sarcasm_model.generate(input_ids=input_ids, max_length=3)
        
    dec = [t5_sarcasm_tokenizer.decode(ids) for ids in output]
    label = dec[0].replace("<pad>", "").strip().replace("</s>", "").strip()
    
    # For debugging: print the T5 model's raw classification
    # print(f"  [DEBUG - T5 Sarcasm Detector] Text: '{text[:80]}...', T5 Label: '{label}'")

    return label == "sarcastic" # Returns boolean for T5's classification

def calculate_sarcasm_score(sentence_text: str) -> float:
    """
    Calculates a sarcasm intensity score based on VADER sentiment divergence.
    This version *always* computes the VADER divergence score.
    The T5 model's detection is optionally used for logging/debugging, but not
    to gate the score calculation.
    """
    if not sentence_text.strip():
        return 0.0

    # Always calculate sentence polarity and word polarities using VADER
    sentence_polarity = vader_sia.polarity_scores(sentence_text)['compound']
    words = sentence_text.split()
    
    if not words:
        return 0.0 # Return 0.0 for empty strings after splitting

    word_polarities = [vader_sia.polarity_scores(word)['compound'] for word in words]
    avg_word_polarity = sum(word_polarities) / len(word_polarities)
    
    # The sarcasm value is the absolute difference between average word polarity and sentence polarity
    sarcasm_val = abs(avg_word_polarity - sentence_polarity)
    
    # Clamp the score to be within [0.0, 1.0]
    return min(max(sarcasm_val, 0.0), 1.0)

# The compute_raw_features function (no changes needed here as it calls calculate_sarcasm_score)

def compute_raw_features(pred_sentence):
    sem_score = calculate_semantic_score(pred_sentence, verb_dict, embedding_model, faiss_index, verbs, nlp)
    sent_score = calculate_sentiment_score(pred_sentence)
    sarc_score = calculate_sarcasm_score(pred_sentence)
    tfidf_raw_score = calculate_tfidf_raw_score(pred_sentence, tfidf_vectorizer)
    buzz_raw_score = calculate_buzzword_raw_score(pred_sentence)
    person_raw_score = calculate_person_raw_score(pred_sentence)
    return {
        "Semantic": sem_score,
        "Sentiment": sent_score,
        "TFIDF_Raw": tfidf_raw_score,
        "Buzzword_Raw": buzz_raw_score,
        "Person_Raw": person_raw_score,
        "Sarcasm": sarc_score,
    }

# --- NDCG Calculation Function ---
def calculate_ndcg(true_relevance, predicted_order, k=None):
    """
    Calculates Normalized Discounted Cumulative Gain (NDCG).
    
    Args:
        true_relevance (list or array): List of relevance scores for items in their original order.
        predicted_order (list or array): List of indices representing the predicted ranking order
                                          (e.g., [2, 0, 1] means item at original index 2 is ranked first).
        k (int, optional): The number of top results to consider. If None, considers all results.
    
    Returns:
        float: The NDCG score.
    """
    # FIX: Explicitly check length of true_relevance if it's a numpy array
    if isinstance(true_relevance, np.ndarray):
        if true_relevance.size == 0 or not predicted_order: # Check if numpy array is empty or predicted_order is empty
            return 0.0
    elif not true_relevance or not predicted_order: # For list inputs
        return 0.0

    if k is None:
        k = len(predicted_order)
    
    k = min(k, len(predicted_order))

    # Get relevance scores based on predicted order
    predicted_relevance = [true_relevance[idx] for idx in predicted_order[:k]]

    # Calculate DCG
    dcg = 0.0
    for i, rel in enumerate(predicted_relevance):
        dcg += (2**rel - 1) / np.log2(i + 2) # i + 2 because 0-indexed position means 1-indexed rank + 1

    # Calculate Ideal DCG (IDCG)
    # Sort true relevance scores in descending order
    ideal_relevance = sorted(true_relevance.tolist() if isinstance(true_relevance, np.ndarray) else true_relevance, reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(ideal_relevance):
        idcg += (2**rel - 1) / np.log2(i + 2)

    if idcg == 0:
        return 0.0 # Avoid division by zero if there's no ideal gain

    return dcg / idcg


# -------------------- Reward Function for PPO (Now using NDCG) --------------------
def calculate_ppo_reward_from_ranking(
    predicted_permutation_indices,
    gold_permutation_original_indices, # This is already sorted by gold rank (best to worst)
    all_original_sentences,
    score_model_instance,
    verbose=False,
    ndcg_k_value: int = None # Added k for NDCG
):
    n_sentences = len(all_original_sentences)
    if n_sentences <= 1:
        return 1.0

    score_model_instance.eval()

    # First pass: Compute all raw feature scores for all sentences in the current sample
    raw_features_per_sentence = []
    for s in all_original_sentences:
        raw_features_per_sentence.append(compute_raw_features(s))

    # Collect raw values for TFIDF, Buzzword, Person across the sample to find min/max
    tfidf_raw_values = np.array([f["TFIDF_Raw"] for f in raw_features_per_sentence])
    buzzword_raw_values = np.array([f["Buzzword_Raw"] for f in raw_features_per_sentence])
    person_raw_values = np.array([f["Person_Raw"] for f in raw_features_per_sentence])

    # Calculate min/max for dynamic normalization (per sample)
    min_tfidf, max_tfidf = np.min(tfidf_raw_values), np.max(tfidf_raw_values)
    min_buzzword, max_buzzword = np.min(buzzword_raw_values), np.max(buzzword_raw_values)
    min_person, max_person = np.min(person_raw_values), np.max(person_raw_values)

    all_original_features = []
    for i, s in enumerate(all_original_sentences):
        raw_scores = raw_features_per_sentence[i]

        # Apply dynamic normalization
        sem_score = raw_scores["Semantic"] # Already normalized/clamped
        sent_score = raw_scores["Sentiment"] # Already normalized/clamped
        sarc_score = raw_scores["Sarcasm"] # Already normalized/clamped

        tfidf_score = normalize_feature(raw_scores["TFIDF_Raw"], max_tfidf, min_tfidf)
        buzz_score = normalize_feature(raw_scores["Buzzword_Raw"], max_buzzword, min_buzzword)
        person_score = normalize_feature(raw_scores["Person_Raw"], max_person, min_person)

        feature_tensor = torch.tensor([
            sem_score,
            sent_score,
            tfidf_score,
            buzz_score,
            person_score,
            sarc_score,
        ], dtype=torch.float32)
        all_original_features.append(feature_tensor)

    all_original_features = torch.stack(all_original_features).to(score_model_instance.logits.device)

    with torch.no_grad():
        all_raw_scores_from_scorenet = score_model_instance(all_original_features)

        # --- REWARD CALCULATION USING NDCG ---
        
        # 1. Prepare relevance scores for Gold
        # 'gold_permutation_original_indices' is already the list of original indices
        # sorted from best to worst according to the gold ranks.
        # We need to assign relevance values to these indices.
        
        # Initialize an array for true_relevance_scores_gold based on original index
        true_relevance_scores_gold = np.zeros(n_sentences)
        # Higher rank in gold_permutation_original_indices (closer to beginning) means higher relevance.
        # Assign relevance: highest for rank 0, lowest for rank n-1. Use a positive scale.
        for rank, original_idx in enumerate(gold_permutation_original_indices):
            true_relevance_scores_gold[original_idx] = n_sentences - rank
        
        # ScoreNet's predicted relevance scores (directly from ScoreNet output)
        # These are already higher for better sentences.
        true_relevance_scores_scorenet = all_raw_scores_from_scorenet.cpu().numpy()

        # Calculate NDCG for LLaMA's permutation against Gold
        ndcg_gold = calculate_ndcg(true_relevance_scores_gold, predicted_permutation_indices, k=ndcg_k_value)
        
        # To calculate NDCG for LLaMA's permutation against ScoreNet's scores,
        # we need to consider ScoreNet's scores as the "true relevance" for that comparison.
        # The predicted_permutation_indices is LLaMA's output, and true_relevance_scores_scorenet
        # are ScoreNet's assessment of each sentence's relevance.
        ndcg_scorenet = calculate_ndcg(true_relevance_scores_scorenet, predicted_permutation_indices, k=ndcg_k_value)

        # Handle NaNs
        if np.isnan(ndcg_gold): ndcg_gold = 0.0
        if np.isnan(ndcg_scorenet): ndcg_scorenet = 0.0

        lambda1 = 0.7
        lambda2 = 0.3

        r_raw = lambda1 * ndcg_gold + lambda2 * ndcg_scorenet
        
        # Apply sigmoid to r_raw to get a value between 0 and 1
        r_final = F.sigmoid(torch.tensor(r_raw, dtype=torch.float32)).item()

        final_reward = r_final
            
        return final_reward

# -------------------- Dataset Loader --------------------
def load_jsonl_dataset_from_folder(root_folder):
    dataset = []
    if not os.path.exists(root_folder):
        print(f"Error: Dataset root folder not found at {root_folder}")
        return []

    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".jsonl"):
                file_path = os.path.join(subdir, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            if "candidates" in data and "ranking" in data:
                                if len(data["candidates"]) > 1 and len(data["ranking"]) == len(data["candidates"]):
                                    # Ensure gold_permutation is a list of original indices, best to worst
                                    # data["ranking"] is like [1, 5, 3, 4, 6, 2] where lower is better rank
                                    # We need original indices sorted by these ranks.
                                    # Create pairs (rank_value, original_index) and sort by rank_value
                                    gold_perm_with_indices = sorted(
                                        [(data["ranking"][i], i) for i in range(len(data["ranking"]))],
                                        key=lambda x: x[0]
                                    )
                                    gold_permutation_original_indices = [idx for rank_val, idx in gold_perm_with_indices]

                                    dataset.append({
                                        "sentences": data["candidates"],
                                        "gold_permutation": gold_permutation_original_indices # This is the sorted list of original indices
                                    })
                                else:
                                    logging.warning(f"Skipping sample in {file_path} due to insufficient sentences (<=1) or mismatch between candidates and ranking length.")
                            else:
                                logging.warning(f"Skipping malformed data in {file_path}: missing 'candidates' or 'ranking'.")
                        except json.JSONDecodeError:
                            logging.error(f"Skipping malformed JSON line in {file_path}.")
                            continue
                        except Exception as e:
                            logging.error(f"Skipping line due to error in {file_path}: {e}")
                            continue
    return dataset

# -------------------- LLaMA + PPO Setup --------------------
LLAMA_MODEL_PATH = "/scratch/../models/llama-3.2-3B"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize tokenizer from the local LLaMA model path
tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_PATH, padding_side="left", local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Move sentiment and T5 sarcasm models to the correct device
sentiment_model.to(device)
t5_sarcasm_model.to(device)

# Load model configuration
config = AutoConfig.from_pretrained(LLAMA_MODEL_PATH, local_files_only=True)
if hasattr(config, "rope_scaling"):
    config.rope_scaling = {"type": "dynamic", "factor": 8.0}

# Load policy model (for training)
policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    LLAMA_MODEL_PATH,
    config=config,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    local_files_only=True,
    trust_remote_code=True,
)
# Load reference model (for PPO baseline)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    LLAMA_MODEL_PATH,
    config=config,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    local_files_only=True,
    trust_remote_code=True,
)
ref_model.eval() # Set reference model to evaluation mode

# Configure PPO training parameters
ppo_config = PPOConfig(
    batch_size=1,
    learning_rate=2e-5,
    mini_batch_size=1,
    gradient_accumulation_steps=1,
    seed=42,
    target_kl=0.2,
    cliprange=0.1,
    cliprange_value=0.1,
    max_grad_norm=0.5,
)

# Initialize PPOTrainer
trainer = PPOTrainer(
    config=ppo_config,
    model=policy_model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    dataset=None,
)

# --- Initialize ScoreNet ---
score_model = ScoreNet().to(device)

# -------------------- Load and Prepare Dataset --------------------
dataset_path = "./Dataset"
dataset = load_jsonl_dataset_from_folder(dataset_path)
print(f"Dataset loaded: {len(dataset)} samples ready for training (filtered out N<=1).")

# Fit TF-IDF Vectorizer on all sentences in the corpus
all_corpus_sentences = [s for item in dataset for s in item["sentences"]]
if all_corpus_sentences:
    tfidf_vectorizer.fit(all_corpus_sentences)
else:
    print("No sentences available to fit TF-IDF Vectorizer. TF-IDF scores will be 0.")

# --- Define and Load the PRE-TRAINED ScoreNet weights ---
PRETRAINED_SCORENET_PATH = "/scratch/../Model/Ranking/trained_score_model.pth"

if os.path.exists(PRETRAINED_SCORENET_PATH):
    score_model.load_state_dict(torch.load(PRETRAINED_SCORENET_PATH, map_location=device))
    score_model.eval()
else:
    raise FileNotFoundError(f"Error: Pre-trained ScoreNet weights not found at {PRETRAINED_SCORENET_PATH}. Cannot proceed without a trained ScoreNet.")

# -------------------- PPO Training Loop --------------------
from random import shuffle
num_epochs = 5

for epoch in range(num_epochs):
    epoch_rewards = []
    shuffle(dataset)

    for batch_idx, batch in enumerate(tqdm(dataset, desc=f"Epoch {epoch + 1}")):
        try:
            torch.cuda.empty_cache()

            n = len(batch["sentences"])
            gold_perm_original_indices = batch["gold_permutation"] # This is now the sorted list of original indices
            all_sentences = batch["sentences"]

            # Calculate k for NDCG based on n
            ndcg_k = max(1, n // 2)

            raw_features_per_sentence = []
            for s in all_sentences:
                raw_features_per_sentence.append(compute_raw_features(s))

            tfidf_raw_values = np.array([f["TFIDF_Raw"] for f in raw_features_per_sentence])
            buzzword_raw_values = np.array([f["Buzzword_Raw"] for f in raw_features_per_sentence])
            person_raw_values = np.array([f["Person_Raw"] for f in raw_features_per_sentence])

            min_tfidf, max_tfidf = np.min(tfidf_raw_values), np.max(tfidf_raw_values)
            min_buzzword, max_buzzword = np.min(buzzword_raw_values), np.max(buzzword_raw_values)
            min_person, max_person = np.min(person_raw_values), np.max(person_raw_values)

            sentences_with_scores = []
            for j, s in enumerate(all_sentences):
                raw_scores = raw_features_per_sentence[j]

                sem_score = raw_scores["Semantic"]
                sent_score = raw_scores["Sentiment"]
                sarc_score = raw_scores["Sarcasm"]
                tfidf_score = normalize_feature(raw_scores["TFIDF_Raw"], max_tfidf, min_tfidf)
                buzz_score = normalize_feature(raw_scores["Buzzword_Raw"], max_buzzword, min_buzzword)
                person_score = normalize_feature(raw_scores["Person_Raw"], max_person, min_person)

                feature_tensor = torch.tensor([
                    sem_score, sent_score, tfidf_score, buzz_score, person_score, sarc_score,
                ], dtype=torch.float32).to(device)

                with torch.no_grad():
                    scorenet_predicted_score = score_model(feature_tensor).item()
                
                sentences_with_scores.append((s, scorenet_predicted_score, j))

            # Sort sentences based on ScoreNet's prediction for the prompt
            scorenet_ordered_sentences_info = sorted(sentences_with_scores, key=lambda x: x[1], reverse=True)


            prompt = (
    "You are an advanced AI designed to meticulously rank sports-related sentences.\n"
    "Your ranking must be based on the following specific and weighted criteria:\n"
    "1. Semantic Relevance to Sports: How closely related is the sentence to core sports concepts and actions?\n"
    "2. Emotional Intensity: Does the sentence convey strong emotions (e.g., excitement, disappointment, triumph)?\n"
    "3. Sarcasm Presence: Is there any underlying sarcasm, which might alter the true meaning or tone?\n"
    "4. Mentions of Key People: Does the sentence include references to prominent athletes, coaches, or figures?\n"
    "5. Buzzword Usage: Does the sentence effectively utilize impactful sports-specific terminology and jargon?\n"
    "6. TFIDF Score: How unique and important are the words in the sentence within the context of sports reporting?\n\n"
    "Below are the sentences you need to rank. Each sentence is accompanied by an 'Overall Score', which has been pre-calculated based on these criteria. A higher 'Overall Score' indicates a more impactful and relevant sentence.\n\n"
    "Your task is to produce a ranked list of these sentences, from best to worst, using their original numerical indices. Prioritize sentences with higher 'Overall Scores' and explicitly consider all the criteria listed above in your decision-making process.\n\n"
)
            for rank_idx, (s, score, original_idx) in enumerate(scorenet_ordered_sentences_info):
                prompt += f"Sentence {original_idx} (Overall Score: {score:.4f}): {s}\n"
            
            prompt += f"\nNow, rank these {n} sentences from best to worst (output only the original indices, best to worst):"

            tokenized_prompt = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)
            input_ids = tokenized_prompt["input_ids"].to(device)
            attention_mask = tokenized_prompt["attention_mask"].to(device)

            with torch.no_grad():
                gen_output = trainer.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=2 * n + 10,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.7,
                    return_dict_in_generate=True,
                )

            generated_ids = gen_output.sequences[:, input_ids.shape[1]:]
            decoded_response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

            predicted_permutation = []
            seen_indices = set()
            for num_str in re.findall(r'\b\d+\b', decoded_response):
                try:
                    num = int(num_str)
                    if 0 <= num < n and num not in seen_indices:
                        predicted_permutation.append(num)
                        seen_indices.add(num)
                except ValueError:
                    pass

            if len(predicted_permutation) < n:
                for i_fill in range(n):
                    if i_fill not in seen_indices:
                        predicted_permutation.append(i_fill)
                        seen_indices.add(i_fill)

            if len(predicted_permutation) != n or len(set(predicted_permutation)) != n:
                logging.error(f"Invalid predicted permutation for sample {batch_idx} (len={len(predicted_permutation)}, unique={len(set(predicted_permutation))}, expected={n}). Decoded: '{decoded_response}', Repaired: {predicted_permutation}")
                continue
                
            reward = calculate_ppo_reward_from_ranking(
                predicted_permutation,
                gold_perm_original_indices, # Pass the original indices sorted by gold rank
                all_sentences,
                score_model,
                verbose=False,
                ndcg_k_value=ndcg_k # Pass the calculated k here
            )

            if not np.isfinite(reward):
                logging.error(f"Non-finite reward encountered ({reward}) for sample {batch_idx}. Skipping batch.")
                continue
            if reward < 0:
                reward = 0.0

            # --- Print reward value per datapoint (ONLY THIS REMAINS) ---
            print(f"Batch {batch_idx+1}: Reward = {reward:.4f}")
            # --- End Print ---

            queries = [input_ids[0]]
            responses = [generated_ids[0]]
            rewards_tensor = torch.tensor(reward, dtype=torch.float32).to(device)
            trainer.step(queries, responses, [rewards_tensor])

            epoch_rewards.append(reward)

        except RuntimeError as e:
            if "out of memory" in str(e):
                logging.error(f"CUDA Out of Memory in batch {batch_idx}. Skipping. 🧠")
                torch.cuda.empty_cache()
            else:
                logging.error(f"Runtime Error in batch {batch_idx}: {e} 🐛")
            continue
        except Exception as e:
            logging.error(f"General Error in batch {batch_idx}: {e} 🐞")
            continue

    save_path = f"/scratch/../Model/Ranking-3B/epoch_{epoch + 1}"
    os.makedirs(save_path, exist_ok=True)
    trainer.model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    torch.save(score_model.state_dict(), os.path.join(save_path, "score_model-3B.pth"))
