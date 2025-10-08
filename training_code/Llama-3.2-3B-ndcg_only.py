import os
import re
import shutil
import json
import torch
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoConfig
from trl import AutoModelForCausalLMWithValueHead, PPOTrainer, PPOConfig
import logging

logging.basicConfig(level=logging.INFO)

# === Disk Space Check Function ===
def has_enough_space(path, required_gb=2):
    total, used, free = shutil.disk_usage(path)
    return free >= required_gb * 1024**3

# === Config ===
LLAMA_MODEL_PATH = "/scratch/../models/llama-3.2-3B"
device = "cuda" if torch.cuda.is_available() else "cpu"

# === Load Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_PATH, padding_side="left", local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# === Load and Update Config ===
config = AutoConfig.from_pretrained(LLAMA_MODEL_PATH, local_files_only=True)
config.rope_scaling = {"type": "dynamic", "factor": 8.0}

# === Load Models with Value Head ===
policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    LLAMA_MODEL_PATH,
    config=config,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    local_files_only=True,
    trust_remote_code=True,
)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    LLAMA_MODEL_PATH,
    config=config,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    local_files_only=True,
    trust_remote_code=True,
)
ref_model.eval()

# Enable gradient checkpointing
policy_model.pretrained_model.gradient_checkpointing_enable()
policy_model.pretrained_model.config.use_cache = False
ref_model.pretrained_model.gradient_checkpointing_enable()
ref_model.pretrained_model.config.use_cache = False

# === PPO Configuration ===
ppo_config = PPOConfig(
    batch_size=1,
    learning_rate=1e-6,
    mini_batch_size=1,
    gradient_accumulation_steps=1,
    early_stopping=False,
    seed=42,
    optimize_cuda_cache=True,
    target_kl=0.5,
    cliprange=0.2,
    cliprange_value=0.2,
    max_grad_norm=0.5,
)

trainer = PPOTrainer(
    config=ppo_config,
    model=policy_model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    dataset=None,
)

# === Load Dataset from JSON/JSONL Files in Folder ===
def load_dataset_from_folder(root_folder):
    dataset = []
    if not os.path.exists(root_folder):
        logging.error(f"Dataset folder {root_folder} does not exist or is inaccessible")
        return dataset
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith((".json", ".jsonl")):
                file_path = os.path.join(subdir, file)
                logging.info(f"Processing file: {file_path}")
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        if file.endswith(".json"):
                            data = json.load(f)
                            if isinstance(data, list):
                                for item in data:
                                    if "candidates" in item and "ranking" in item:
                                        ranking = item["ranking"]
                                        if len(ranking) == len(item["candidates"]):
                                            gold_perm = sorted(range(len(ranking)), key=lambda i: ranking[i])
                                            dataset.append({
                                                "sentences": item["candidates"],
                                                "gold_permutation": gold_perm
                                            })
                        elif file.endswith(".jsonl"):
                            for line_number, line in enumerate(f, start=1):
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    data = json.loads(line)
                                    if "candidates" in data and "ranking" in data:
                                        ranking = data["ranking"]
                                        if len(ranking) == len(data["candidates"]):
                                            gold_perm = sorted(range(len(ranking)), key=lambda i: ranking[i])
                                            dataset.append({
                                                "sentences": data["candidates"],
                                                "gold_permutation": gold_perm
                                            })
                                except json.JSONDecodeError as e:
                                    logging.error(f"JSON error in line {line_number} of {file_path}: {e}")
                except Exception as e:
                    logging.error(f"Error reading {file_path}: {e}")
    logging.info(f"Loaded {len(dataset)} valid examples")
    return dataset

dataset_folder = "./Dataset"
full_dataset = load_dataset_from_folder(dataset_folder)
dataset = [item for item in full_dataset if len(item["sentences"]) > 1]
logging.info(f"Filtered dataset size: {len(dataset)}")

# === NDCG@k Function ===
def dcg_at_k(rel_scores, k):
    return sum((2**rel - 1) / np.log2(idx + 2) for idx, rel in enumerate(rel_scores[:k]))

def ndcg_at_k(pred_indices, gold_permutation, k):
    # Convert gold_permutation (lower is better) to relevance scores (higher is better)
    max_rel = len(gold_permutation)
    relevance_scores = [max_rel - gold_permutation[i] for i in range(len(gold_permutation))]

    # Get predicted relevance scores based on predicted ranking
    pred_rels = [relevance_scores[i] for i in pred_indices]

    # Ideal ranking: sort gold permutation
    ideal_indices = sorted(range(len(gold_permutation)), key=lambda x: gold_permutation[x])
    ideal_rels = [relevance_scores[i] for i in ideal_indices]

    dcg = dcg_at_k(pred_rels, k)
    idcg = dcg_at_k(ideal_rels, k)
    return dcg / idcg if idcg != 0 else 0.0

# === Updated Reward Function using NDCG@k ===
def reward_fn(batch, responses, gold_permutations):
    rewards = []
    for sentences, pred, gold_perm in zip(batch, responses, gold_permutations):
        n = len(sentences)
        k = max(1, n // 2)
        ndcg = ndcg_at_k(pred, gold_perm, k)
        rewards.append(ndcg)
        logging.info(f"Computed NDCG@{k}: {ndcg:.4f}")
    return rewards


# === Training Loop ===
for epoch in range(3):
    logging.info(f"Starting Epoch {epoch}...")
    epoch_rewards = []

    for batch_idx, batch in enumerate(tqdm(dataset, desc=f"Epoch {epoch}")):
        n_sentences = len(batch["sentences"])
        k = max(1, n_sentences // 2 )

        prompt = (
            "You are an AI that ranks sports-related sentences based on importance using these criteria:\n"
            "1. Sports Relevance\n"
            "2. Emotional Intensity\n"
            "3. Sarcasm Presence\n"
            "4. Key People Mentions\n"
            "5. Buzzword Usage\n\n"
            f"Rank the following {n_sentences} sentences (0-based indices). Output ONLY numbers in order (best first), separated by spaces:\n\n"
            + "\n".join(f"{i}. {s}" for i, s in enumerate(batch["sentences"])) +
            "\n\nRanked indices:"
        )

        try:
            tokenized = tokenizer([prompt], return_tensors="pt", padding=True, truncation=True, max_length=256)
            if len(tokenized["input_ids"][0]) > 256:
                logging.warning(f"Skipping batch {batch_idx} due to excessive length: {len(tokenized['input_ids'][0])}")
                continue
            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)
            input_length = input_ids.shape[1]

            response_ids = trainer.model.pretrained_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=5,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True,
                top_p=0.9,
                num_beams=1,
                temperature=0.8,
                output_scores=False,
                return_dict_in_generate=True
            )

            generated_ids = response_ids.sequences[:, input_length:]
            responses_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            resp_clean = re.sub(r'[^0-9 ]', '', responses_text[0])
            numbers = re.findall(r'\b\d+\b', resp_clean)

            valid_indices = []
            for n in numbers:
                idx = int(n)
                if 0 <= idx < n_sentences and idx not in valid_indices:
                    valid_indices.append(idx)

            if len(valid_indices) < n_sentences:
                missing = [i for i in range(n_sentences) if i not in valid_indices]
                valid_indices += missing[:n_sentences - len(valid_indices)]

            parsed = [valid_indices[:n_sentences]]
            logging.info(f"Parsed indices: {parsed[0]} | Gold: {batch['gold_permutation']} | k: {k}")

            rewards_float = reward_fn([batch["sentences"]], parsed, [batch["gold_permutation"]])

            if not np.isfinite(rewards_float[0]):
                logging.warning("Skipping PPO step due to non-finite reward")
                continue

            epoch_rewards.append(rewards_float[0])
            logging.info(f"Reward: {rewards_float[0]:.4f}")

            queries = [input_ids[0].detach()]
            responses = [generated_ids[0].detach()]
            rewards = [torch.tensor(rewards_float[0], dtype=torch.float32).to(device)]

            stats = trainer.step(queries, responses, rewards)
            logging.info(f"PPO step completed | KL: {stats['objective/kl']:.4f}")

            torch.cuda.empty_cache()

        except RuntimeError as e:
            logging.error(f"OOM error at batch {batch_idx}. Skipping.")
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            logging.error(f"Error at batch {batch_idx}: {str(e)}")
            torch.cuda.empty_cache()
            continue

    avg_reward = np.mean(epoch_rewards) if epoch_rewards else 0
    logging.info(f"Epoch {epoch} Complete | Avg Reward: {avg_reward:.4f}")

    save_path = f"/scratch/../Model/ Llama-3.2-3B-ndcg/ppo_epoch{epoch}"
    parent_dir = os.path.dirname(save_path)
    os.makedirs(parent_dir, exist_ok=True)
    try:
        if has_enough_space(parent_dir, required_gb=2):
            policy_model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            logging.info(f"Model saved to {save_path}")
        else:
            logging.warning(f"Not enough disk space to save model at {save_path}")
    except Exception as e:
        logging.error(f"Failed to save model: {str(e)}")
