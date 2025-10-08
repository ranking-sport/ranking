# === Imports and Setup ===
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
import time

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
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True,
    trust_remote_code=True,
)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    LLAMA_MODEL_PATH,
    config=config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True,
    trust_remote_code=True,
)
ref_model.eval()

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

# === Load Dataset from Folder ===
def load_dataset_from_folder(root_folder):
    dataset = []
    if not os.path.exists(root_folder):
        logging.error(f"Dataset folder {root_folder} does not exist")
        return dataset
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith((".json", ".jsonl")):
                file_path = os.path.join(subdir, file)
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
                        else:
                            for line in f:
                                if not line.strip():
                                    continue
                                data = json.loads(line)
                                if "candidates" in data and "ranking" in data:
                                    ranking = data["ranking"]
                                    if len(ranking) == len(data["candidates"]):
                                        gold_perm = sorted(range(len(ranking)), key=lambda i: ranking[i])
                                        dataset.append({
                                            "sentences": data["candidates"],
                                            "gold_permutation": gold_perm
                                        })
                except Exception as e:
                    logging.error(f"Error reading {file_path}: {e}")
    return dataset

dataset_folder = "./Dataset"
full_dataset = load_dataset_from_folder(dataset_folder)
dataset = [item for item in full_dataset if len(item["sentences"]) > 1]

# === Recall@k Function ===
def recall_at_k(pred_indices, gold_top_k_indices, k):
    return len(set(pred_indices[:k]) & set(gold_top_k_indices)) / k

# === Reward Function ===
def reward_fn(batch, responses, gold_permutations):
    rewards = []
    for sentences, pred, gold_perm in zip(batch, responses, gold_permutations):
        k = max(1, len(sentences) // 2)
        gold_top_k = gold_perm[:k]
        r_rec = recall_at_k(pred, gold_top_k, k)
        rewards.append(r_rec)
    return rewards

# === Training Loop ===
for epoch in range(5):
    logging.info(f"Starting Epoch {epoch}...")
    epoch_rewards = []

    for batch_idx, batch in enumerate(tqdm(dataset, desc=f"Epoch {epoch}")):
        try:
            n_sentences = len(batch["sentences"])
            k = max(1, n_sentences // 2)

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

            tokenized = tokenizer([prompt], return_tensors="pt", padding=True, truncation=True, max_length=128)
            if len(tokenized["input_ids"][0]) > 128:
                continue

            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)
            input_length = input_ids.shape[1]

            response_ids = policy_model.pretrained_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=4,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True,
                top_p=0.9,
                num_beams=1,
                temperature=0.8,
            )

            generated_ids = response_ids[:, input_length:]
            responses_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            resp_clean = re.sub(r'[^0-9 ]', '', responses_text[0])
            numbers = re.findall(r'\b\d+\b', resp_clean)

            valid_indices = []
            for n in numbers:
                idx = int(n)
                if 0 <= idx < n_sentences and idx not in valid_indices:
                    valid_indices.append(idx)
            if len(valid_indices) < n_sentences:
                valid_indices += [i for i in range(n_sentences) if i not in valid_indices]

            parsed = [valid_indices[:n_sentences]]
            rewards_float = reward_fn([batch["sentences"]], parsed, [batch["gold_permutation"]])
            if not np.isfinite(rewards_float[0]):
                continue

            epoch_rewards.append(rewards_float[0])

            queries = [input_ids[0].detach().cpu()]
            responses = [generated_ids[0].detach().cpu()]
            rewards = [torch.tensor(rewards_float[0], dtype=torch.float32).cpu()]

            logging.info(f"[Batch {batch_idx}] PPO Step | Gold: {batch['gold_permutation']} | Parsed: {parsed[0]} | Reward: {rewards_float[0]:.4f} | k: {k}")

            stats = trainer.step(queries, responses, rewards)
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            del input_ids, attention_mask, tokenized, response_ids, generated_ids

            if batch_idx % 50 == 0:
                time.sleep(1)

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

    save_path = f"/scratch/../Model/Llama-3B-r/ppo_epoch{epoch}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if has_enough_space(os.path.dirname(save_path), 2):
        policy_model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        logging.info(f"Model saved to {save_path}")
    else:
        logging.warning(f"Not enough disk space to save model at {save_path}")
