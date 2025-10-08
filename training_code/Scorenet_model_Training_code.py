import os
import json
import torch
import numpy as np
import pandas as pd
import spacy
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5Tokenizer, AutoModelForSeq2SeqLM
from tqdm.auto import tqdm
import logging
import faiss # Import faiss for semantic scoring
import torch.nn.functional as F # For softmax in ScoreNet and sigmoid if needed elsewhere

# Configure logging for better feedback
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Essential Setup & Feature Definitions ---

# Download NLTK data (if not already present)
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except nltk.downloader.DownloadError:
    logging.info("Downloading NLTK vader_lexicon...")
    nltk.download("vader_lexicon")

# Load SpaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logging.info("Downloading spacy model 'en_core_web_sm'...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# --- Load external data (CSV files) or use defaults ---
# Expanded default for better example scores based on previous analysis
default_sports_keywords = {
    "champion": 1.0, "win": 0.9, "record": 0.8, "goal": 0.7, "title": 0.95,
    "trophy": 0.85, "victory": 0.9, "cup": 0.75, "medal": 0.7, "stadium": 0.5,
    "athlete": 0.6, "coach": 0.55, "game": 0.6, "match": 0.6,
    "runs": 0.6, "wickets": 0.7, "innings": 0.5, "test": 0.4,
    "championship": 0.9, "final": 0.8, "world cup": 0.95, "odi": 0.7, "t20": 0.7
}

try:
    sports_keywords_df = pd.read_csv("sports_keywords.csv")
    sports_keywords_dict = dict(zip(sports_keywords_df['Word'].str.lower(), sports_keywords_df['AverageScore']))
    logging.info("Loaded sports_keywords.csv successfully for Buzzword Score.")
except FileNotFoundError:
    logging.warning("⚠️ Warning: sports_keywords.csv not found. Buzzword score will use an expanded default list.")
    sports_keywords_dict = default_sports_keywords
except KeyError as e:
    logging.warning(f"⚠️ Error reading sports_keywords.csv: Missing expected column {e}. Buzzword score will use an expanded default list.")
    sports_keywords_dict = default_sports_keywords

# Expanded default for better example scores, using hypothetical HPI values
default_famous_people = {
    "virat kohli": 850, "sachin tendulkar": 900, "ms dhoni": 800,
    "shubman gill": 250, "ravindra jadeja": 300, "pat cummins": 400,
    "cristiano ronaldo": 950, "lionel messi": 980, "serena williams": 880,
    "novak djokovic": 870, "stephen curry": 890, "michael jordan": 990,
    "max verstappen": 700, "fernando alonso": 650, "charles leclerc": 500,
    "temba bavuma": 100, "mushfiqur rahim": 120
}

try:
    famous_people_df = pd.read_csv("processed_persons.csv")
    famous_people_dict = dict(zip(famous_people_df['name'].str.lower(), famous_people_df['hpi']))
    logging.info("Loaded processed_persons.csv successfully for Person Score.")
except FileNotFoundError:
    logging.warning("⚠️ Warning: processed_persons.csv not found. Person score will use an expanded default list.")
    famous_people_dict = default_famous_people
except KeyError as e:
    logging.warning(f"⚠️ Error reading processed_persons.csv: Missing expected column {e}. Person score will use an expanded default list.")
    famous_people_dict = default_famous_people

# --- Global Feature Extraction Models/Tools ---

# For Semantic Score
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def generate_verb_dictionary():
    # Adding more relevant sports verbs and their forms
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

# For TFIDF Score (vectorizer will be fitted later)
tfidf_vectorizer = TfidfVectorizer()

# For Sarcasm Score
t5_sarcasm_tokenizer = T5Tokenizer.from_pretrained("mrm8488/t5-base-finetuned-sarcasm-twitter")
t5_sarcasm_model = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/t5-base-finetuned-sarcasm-twitter")
vader_sia = SentimentIntensityAnalyzer()

# -------------------- ScoreNet Model Class --------------------
class ScoreNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.randn(6) * 0.01) # 6 features

    def forward(self, x):
        weights = torch.softmax(self.logits, dim=0)
        return (x * weights).sum(dim=-1)

# -------------------- Feature Computation Functions (MODIFIED) --------------------

# This normalization is for internal use within the loss function, applied dynamically per sample
def normalize_feature(value, max_val, min_val=0.0):
    if max_val == min_val:
        return 0.0
    return float((value - min_val) / (max_val - min_val))

# Semantic Score (unchanged, inherently 0-1)
def calculate_semantic_score(sentence_text, verb_dict, model, index, verbs_list, nlp_model, k=1):
    if not isinstance(sentence_text, str):
        sentence_text = str(sentence_text)
    doc = nlp_model(sentence_text)
    token_scores = []
    for token in doc:
        if token.ent_type_: # Skip named entities
            continue
        token_lower = token.text.lower()
        if token_lower in verb_dict:
            token_scores.append(verb_dict[token_lower])
        else:
            token_embedding = model.encode([token_lower], convert_to_numpy=True)
            distances, indices = index.search(token_embedding, k)
            neighbor_scores = [verb_dict[verbs_list[indices[0][i]]] for i in range(k)]
            avg_neighbor_score = sum(neighbor_scores) / len(neighbor_scores) if neighbor_scores else 0.0
            token_scores.append(avg_neighbor_score)
    if token_scores:
        overall_score = sum(token_scores) / len(token_scores)
    else:
        overall_score = 0.0
    return overall_score

# Sentiment Score (unchanged, inherently 0-1)
def calculate_sentiment_score(text):
    inputs = sentiment_tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    current_device = sentiment_model.device
    inputs = {k: v.to(current_device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
    scores = torch.nn.functional.sigmoid(outputs.logits)
    intensity_score = torch.max(scores[0]).item()
    return intensity_score

# TFIDF Score (returns RAW sum)
def calculate_tfidf_raw_score(text, vectorizer):
    if not hasattr(vectorizer, 'vocabulary_') or not vectorizer.vocabulary_:
        return 0.0
    tfidf_matrix = vectorizer.transform([text])
    raw_sum = tfidf_matrix.sum()
    return raw_sum # Return raw sum

# Buzzword Score (returns RAW sum)
def calculate_buzzword_raw_score(sentence_text):
    score = sum(sports_keywords_dict.get(word, 0) for word in sentence_text.lower().split())
    return score # Return raw sum

# Person Score (returns MAX HPI value)
def calculate_person_raw_score(sentence_text):
    doc = nlp(sentence_text)
    hpi_values = []
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            hpi = famous_people_dict.get(ent.text.lower(), 0)
            hpi_values.append(hpi)
    return float(max(hpi_values)) if hpi_values else 0.0 # Return max HPI

# Sarcasm Score (unchanged, inherently 0-1)
def detect_sarcasm_t5(text):
    input_ids = t5_sarcasm_tokenizer.encode(text + '</s>', return_tensors='pt')
    current_device = t5_sarcasm_model.device
    input_ids = input_ids.to(current_device)
    with torch.no_grad():
        output = t5_sarcasm_model.generate(input_ids=input_ids, max_length=3)
    dec = [t5_sarcasm_tokenizer.decode(ids) for ids in output]
    label = dec[0].replace("<pad>", "").strip().replace("</s>", "").strip()
    return label == "sarcastic"

def calculate_sarcasm_score(sentence_text: str) -> float:
    if detect_sarcasm_t5(sentence_text):
        sentence_polarity = vader_sia.polarity_scores(sentence_text)['compound']
        words = sentence_text.split()
        word_polarities = [vader_sia.polarity_scores(word)['compound'] for word in words]
        if not word_polarities:
            return 0.0
        avg_word_polarity = sum(word_polarities) / len(word_polarities)
        sarcasm_val = abs(avg_word_polarity - sentence_polarity)
        return min(max(sarcasm_val, 0.0), 1.0)
    else:
        return 0.0

# MODIFIED: compute_features now returns a dictionary of raw/inherently normalized scores
def compute_features(pred_sentence): # Removed gold_context_str as it's not used by these feature fns directly
    return {
        "Semantic": calculate_semantic_score(pred_sentence, verb_dict, embedding_model, faiss_index, verbs, nlp),
        "Sentiment": calculate_sentiment_score(pred_sentence),
        "TFIDF_Raw": calculate_tfidf_raw_score(pred_sentence, tfidf_vectorizer),
        "Buzzword_Raw": calculate_buzzword_raw_score(pred_sentence),
        "Person_Raw": calculate_person_raw_score(pred_sentence),
        "Sarcasm": calculate_sarcasm_score(pred_sentence),
    }

# -------------------- ListNet-based Loss Function (for ScoreNet training) --------------------

def calculate_listnet_loss(
    score_model_instance,
    all_raw_features_per_sentence, # Now expects raw features
    gold_rank_values, # Expects 1-indexed ranks (1 is best)
    n_sentences,
    device # Pass device here
):
    """
    Calculates ListNet loss given raw features and gold ranks, applying dynamic normalization.
    """
    if n_sentences == 0:
        return torch.tensor(0.0, device=device)

    # Convert 1-indexed rank values (1=best, N=worst) to 0-indexed preference scores (higher score = better)
    gold_scores = torch.tensor([
        n_sentences - gold_rank_values[i]
        for i in range(n_sentences)
    ], dtype=torch.float32).to(device)

    # --- Dynamic Normalization within the Loss Function ---
    tfidf_raw_values = np.array([f["TFIDF_Raw"] for f in all_raw_features_per_sentence])
    buzzword_raw_values = np.array([f["Buzzword_Raw"] for f in all_raw_features_per_sentence])
    person_raw_values = np.array([f["Person_Raw"] for f in all_raw_features_per_sentence])

    min_tfidf, max_tfidf = np.min(tfidf_raw_values), np.max(tfidf_raw_values)
    min_buzzword, max_buzzword = np.min(buzzword_raw_values), np.max(buzzword_raw_values)
    min_person, max_person = np.min(person_raw_values), np.max(person_raw_values)

    # Create normalized feature tensors
    normalized_features_tensors = []
    for raw_scores_dict in all_raw_features_per_sentence:
        sem_score = raw_scores_dict["Semantic"]
        sent_score = raw_scores_dict["Sentiment"]
        sarc_score = raw_scores_dict["Sarcasm"]
        tfidf_score = normalize_feature(raw_scores_dict["TFIDF_Raw"], max_tfidf, min_tfidf)
        buzz_score = normalize_feature(raw_scores_dict["Buzzword_Raw"], max_buzzword, min_buzzword)
        person_score = normalize_feature(raw_scores_dict["Person_Raw"], max_person, min_person)

        feature_tensor = torch.tensor([
            sem_score, sent_score, tfidf_score, buzz_score, person_score, sarc_score,
        ], dtype=torch.float32)
        normalized_features_tensors.append(feature_tensor)
    
    # Stack features to pass to ScoreNet
    stacked_normalized_features = torch.stack(normalized_features_tensors).to(device)

    predicted_raw_scores = score_model_instance(stacked_normalized_features)

    P_gold = torch.softmax(gold_scores, dim=0)
    P_pred = torch.softmax(predicted_raw_scores, dim=0)

    listnet_loss = -torch.sum(P_gold * torch.log(P_pred + 1e-8))
    return listnet_loss

# -------------------- ScoreNet Training Function --------------------
def train_scorenet(
    score_model: ScoreNet,
    optimizer: torch.optim.Optimizer,
    training_samples: list,
    num_epochs: int = 500,
    batch_size: int = 1,
    device: str = "cpu",
    save_path: str = "trained_score_model.pth"
):
    score_model.train() # Set model to training mode
    logging.info(f"\n--- Starting ScoreNet Training for {num_epochs} epochs ---")

    sentiment_model.to(device) # Ensure sentiment model is on device
    t5_sarcasm_model.to(device) # Ensure T5 sarcasm model is on device

    # Outer tqdm for Epochs
    for epoch in tqdm(range(num_epochs), desc="ScoreNet Epochs"):
        total_loss = 0
        np.random.shuffle(training_samples) # Shuffle samples each epoch

        # Inner tqdm for Batches
        for i in tqdm(range(0, len(training_samples), batch_size),
                      desc=f"Epoch {epoch+1}/{num_epochs} Batches", leave=False):
            batch_samples = training_samples[i:i + batch_size]
            if not batch_samples:
                continue

            optimizer.zero_grad() # Clear gradients for this batch

            batch_loss = 0
            for sample_idx_in_batch, sample in enumerate(batch_samples):
                global_sample_idx = i + sample_idx_in_batch
                
                all_sentences = sample["candidates"]
                gold_rank_values = sample["ranking"] # 1-indexed ranks

                if not all_sentences:
                    logging.warning(f"Skipping empty candidate list for sample {global_sample_idx}")
                    continue

                n_sentences = len(all_sentences)
                
                # Compute RAW features for all sentences in the sample
                all_raw_features_per_sentence = []
                for s in all_sentences:
                    all_raw_features_per_sentence.append(compute_features(s))

                loss_per_sample = calculate_listnet_loss(
                    score_model,
                    all_raw_features_per_sentence, # Pass raw features
                    gold_rank_values,
                    n_sentences,
                    device # Pass device
                )
                batch_loss += loss_per_sample

            if batch_samples:
                batch_loss.backward()
                optimizer.step()
                total_loss += batch_loss.item()

        avg_epoch_loss = total_loss / len(training_samples) if training_samples else 0.0
        logging.info(f"Epoch {epoch+1}/{num_epochs} finished. Average Loss: {avg_epoch_loss:.4f}")

    logging.info("--- ScoreNet Training Complete ---")
    score_model.eval() # Set model back to evaluation mode
    torch.save(score_model.state_dict(), save_path)
    logging.info(f"Trained ScoreNet weights saved to: {save_path}")

# === Load Dataset from JSON/JSONL Files in Folder (Efficient) ===
def load_dataset_from_folder_efficient(root_folder):
    """
    Loads dataset from JSON/JSONL files in a folder, optimized for large datasets.
    It yields individual items to avoid loading the entire dataset into memory at once.
    """
    if not os.path.exists(root_folder):
        logging.error(f"Dataset folder {root_folder} does not exist or is inaccessible")
        return

    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith((".json", ".jsonl")):
                file_path = os.path.join(subdir, file)
                logging.info(f"Loading data from file: {file_path}")
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        if file.endswith(".jsonl"):
                            for line_number, line in enumerate(f, start=1):
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    data = json.loads(line)
                                    if "candidates" in data and "ranking" in data:
                                        ranking = data["ranking"]
                                        if len(ranking) == len(data["candidates"]):
                                            yield {
                                                "candidates": data["candidates"],
                                                "ranking": ranking
                                            }
                                        else:
                                            logging.warning(f"Skipping line {line_number} in {file_path}: 'ranking' length ({len(ranking)}) does not match 'candidates' length ({len(data['candidates'])}).")
                                    else:
                                        logging.warning(f"Skipping line {line_number} in {file_path}: Missing 'candidates' or 'ranking' keys.")
                                except json.JSONDecodeError as e:
                                    logging.error(f"JSON error in line {line_number} of {file_path}: {e}")
                        elif file.endswith(".json"):
                            data = json.load(f)
                            if isinstance(data, list):
                                for item_idx, item in enumerate(data):
                                    if "candidates" in item and "ranking" in item:
                                        ranking = item["ranking"]
                                        if len(ranking) == len(item["candidates"]):
                                            yield {
                                                "candidates": item["candidates"],
                                                "ranking": ranking
                                            }
                                        else:
                                            logging.warning(f"Skipping item {item_idx} in {file_path}: 'ranking' length ({len(ranking)}) does not match 'candidates' length ({len(item['candidates'])}).")
                                    else:
                                        logging.warning(f"Skipping item {item_idx} in {file_path}: Missing 'candidates' or 'ranking' keys.")
                            elif isinstance(data, dict):
                                if "candidates" in data and "ranking" in data:
                                    ranking = data["ranking"]
                                    if len(ranking) == len(data["candidates"]):
                                        yield {
                                            "candidates": data["candidates"],
                                            "ranking": ranking
                                        }
                                    else:
                                        logging.warning(f"Skipping single object in {file_path}: 'ranking' length ({len(ranking)}) does not match 'candidates' length ({len(data['candidates'])}).")
                                else:
                                    logging.warning(f"Skipping single object in {file_path}: Missing 'candidates' or 'ranking' keys.")
                            else:
                                logging.warning(f"Skipping {file_path}: Unexpected JSON format (not a list or dict).")

                except json.JSONDecodeError as e:
                    logging.error(f"JSON error in file {file_path}: {e}")
                except Exception as e:
                    logging.error(f"Error reading {file_path}: {e}")

# -------------------- Configuration (ADJUST THESE PATHS) --------------------
TRAINED_MODELS_DIR = "/scratch/../Model/Ranking/" # Directory to save trained ScoreNet weights
DATASET_FOLDER = "./Dataset" # Folder containing your JSON/JSONL dataset files for training

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    logging.info(f"ScoreNet training script started. Using device: {device}")

    # Ensure sentiment and T5 sarcasm models are on the correct device before feature calculation
    sentiment_model.to(device)
    t5_sarcasm_model.to(device)

    # --- Load Dataset ---
    logging.info(f"Loading dataset from {DATASET_FOLDER}...")
    raw_loaded_data = []
    for item in tqdm(load_dataset_from_folder_efficient(DATASET_FOLDER),
                     desc="Loading raw dataset items"):
        raw_loaded_data.append(item)

    training_data = [item for item in raw_loaded_data if len(item["candidates"]) > 1]
    logging.info(f"Loaded and filtered dataset size for training: {len(training_data)} samples.")
    if not training_data:
        logging.error("No valid samples loaded from the dataset folder for training. Exiting.")
        exit()

    # --- Dynamic TF-IDF Vectorizer Fitting ---
    logging.info("Fitting TF-IDF Vectorizer on all training samples for comprehensive vocabulary...")
    all_corpus_sentences = [s for item in training_data for s in item["candidates"]]
    if all_corpus_sentences:
        tfidf_vectorizer.fit(all_corpus_sentences)
        logging.info(f"TF-IDF Vectorizer fitted with {len(tfidf_vectorizer.vocabulary_)} terms.")
    else:
        logging.warning("Warning: No sentences in training samples to fit TF-IDF Vectorizer. TF-IDF scores will be 0.")

    # --- Initialize ScoreNet and Optimizer ---
    score_model = ScoreNet().to(device)
    score_optimizer = torch.optim.Adam(score_model.parameters(), lr=0.001)

    # --- Create directory for trained models if it doesn't exist ---
    os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)
    scorenet_save_path = os.path.join(TRAINED_MODELS_DIR, "trained_score_model.pth")

    # --- Train the ScoreNet ---
    train_scorenet(
        score_model=score_model,
        optimizer=score_optimizer,
        training_samples=training_data,
        num_epochs=5,
        batch_size=1,
        device=device,
        save_path=scorenet_save_path
    )
    logging.info("\n✅ ScoreNet training process completed. Weights saved.")
