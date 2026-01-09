import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import time
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# --- 📜 설정 부분 ---
MODEL_NAME = "Llama-3.2-3B-Instruct"
MODEL_REPO = "meta-llama/Llama-3.2-3B-Instruct"  # HF repo id
INPUT_FILE_PATH = "/Users/jisu/Desktop/대학원/CVPR_2026_VTG/MAD/test.jsonl"
OUTPUT_FILE_PATH = f"/Users/jisu/Desktop/대학원/CVPR_2026_VTG/MAD/llm_change/{MODEL_NAME}_sample100.jsonl"

# --- 🚀 실행 모드 설정 ---
DEMO_MODE = False
RANDOM_SAMPLE = True
SAMPLE_SIZE = 100
RANDOM_SEED = 42

def create_prompt(original_query: str) -> str:
    prompt = f"""### Primary Goal:
        Your main job is to analyze an Input sentence and transform it into exactly two complete, grammatically correct sentences (Sentence 1 and Sentence 2).

        ### 1. Splitting Strategy (Choose ONE)
        Identify the nature of the Input query and apply the appropriate splitting logic:
        
        1.1. Temporal Order Split (Mandatory when sequential actions exist): If the Input describes two or more distinct, sequential actions, YOU MUST apply this split.
        1.2. Special Case Split (Mandatory): Apply Rule 6 (Section 3) if the Input describes only a single, indivisible action, a static state, or is too short.

        ### 2. Common Execution Rules (APPLY TO ALL SPLITS)
        All resulting Sentence 1 and Sentence 2 pairs MUST adhere to the following rules:
        
        2.1. Analyze & Split: Identify the two distinct parts (action or context) in the Input.
        2.2. Temporal Order Execution: If using the Temporal Order Split (1.1), YOU MUST SPLIT THE SENTENCE based on the chronological sequence. Sentence 1 MUST describe the action or event that happens chronologically *before* the action or event in Sentence 2.
        2.3. Correct Grammar: Ensure BOTH output sentences are grammatically perfect. Fix all errors in subject-verb agreement, tense, or phrasing.
        2.4. Replace Noun/Pronoun: You MUST replace all pronouns (e.g., it, he, she, they) in Sentence 2 with the specific noun or entity they refer to from Sentence 1. Sentence 2 must be self-contained.
        2.5. No New Information: Stick ONLY to the information given in the Input. Do not add adjectives, emotions, or any new details.

        ### 3. Rule 6: Special Case Logic (Forced Splitting for Indivisible Queries)
        This logic provides the exact instruction for the Special Case Split (1.2):
        
        3.1. Sentence 1 (Context Initialization): Create a simple sentence that defines the main subject, location, or state using the most important noun/entity from the original query. This sentence must introduce the primary subject or set the scene.
        3.2. Sentence 2 (Action/State Statement): Sentence 2 MUST contain the entire original query's core information (including the main verb and object).

        ### Output Format: (STRICTLY Follow This Structure)
        The output MUST always consist of exactly two sentences separated by a period and a newline.
        
        --- CRITICAL FORMATTING RESTRICTION ---
        The output MUST NOT contain any brackets ([]), labels, numbering, metadata, unnecessary newlines, or explanatory text (e.g., 'Explanation:'). Output ONLY the two sentences.
        
        ### Your Task:
        Input: "{original_query}"
        Output:"""
    return prompt

def main():
    print("Loading model... This may take a moment.")

    # 디바이스 설정 (맥이면 cuda 대신 cpu / mps로 자동)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"✅ Using device: {device}")

    # HF Hub에서 모델 로드 (처음 한 번은 자동 다운로드 + 캐시)
    print(f"📥 Loading from HF Hub repo: {MODEL_REPO}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_REPO,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_REPO,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device).eval()

    # pad_token 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = tokenizer.eos_token_id

    # 입력 파일 읽기
    try:
        with open(INPUT_FILE_PATH, "r", encoding="utf-8") as f_in:
            lines = f_in.readlines()
    except FileNotFoundError:
        print(f"❌ Error: Input file not found at {INPUT_FILE_PATH}")
        return

    # 처리할 라인 선택
    if DEMO_MODE:
        lines_to_process = lines[:10]
        print(f"\n--- DEMO MODE: Processing first 10 queries ---")
    elif RANDOM_SAMPLE:
        if RANDOM_SEED is not None:
            random.seed(RANDOM_SEED)
            print(f"🎲 Random seed set to: {RANDOM_SEED}")
        total_lines = len(lines)
        sample_size = min(SAMPLE_SIZE, total_lines)
        lines_to_process = random.sample(lines, sample_size)
        print(f"\n--- RANDOM SAMPLE MODE: Processing {sample_size} out of {total_lines} queries ---")
    else:
        lines_to_process = lines
        print(f"\n--- FULL MODE: Processing {len(lines)} queries ---")

    processing_times = []

    with open(OUTPUT_FILE_PATH, "w", encoding="utf-8") as f_out:
        for line in tqdm(lines_to_process, desc="Splitting Queries"):
            try:
                data = json.loads(line)
                original_query = data.get("query")
                if not original_query:
                    print(f"Warning: Skipping line with no 'query' field: {line.strip()}")
                    continue

                start_time = time.time()

                prompt = create_prompt(original_query)
                
                messages = [
                    {"role": "system", "content": "You are a helpful assistant that analyzes sentences and splits them when they contain multiple sequential actions."},
                    {"role": "user", "content": prompt}
                ]

                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                model_inputs = tokenizer(
                    [text], 
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                ).to(device)

                with torch.no_grad():
                    generated_ids = model.generate(
                        **model_inputs,
                        max_new_tokens=128,
                        do_sample=True,
                        temperature=0.3,
                        top_p=0.9,
                        repetition_penalty=1.1,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                gen_only = generated_ids[0][len(model_inputs.input_ids[0]):]
                clean_output = tokenizer.decode(gen_only, skip_special_tokens=True).strip()
                
                if clean_output.startswith("Output:"):
                    clean_output = clean_output[7:].strip()

                end_time = time.time()
                elapsed_time = end_time - start_time
                processing_times.append(elapsed_time)

                data["query_original"] = original_query
                data["query"] = clean_output
                data["processing_time_seconds"] = round(elapsed_time, 4)
                f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON line: {line.strip()}")
            except Exception as e:
                print(f"\nAn error occurred on line '{line.strip()}'. Error: {e}")
                import traceback
                traceback.print_exc()

    if processing_times:
        avg_time = sum(processing_times) / len(processing_times)
        min_time = min(processing_times)
        max_time = max(processing_times)
        total_time = sum(processing_times)
        
        print(f"\n{'='*60}")
        print(f"MODEL_NAME: {MODEL_NAME}")
        print(f"⏱️  Processing Time Statistics")
        print(f"{'='*60}")
        print(f"Total queries processed: {len(processing_times)}")
        print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"Average time per query: {avg_time:.4f} seconds")
        print(f"Minimum time: {min_time:.4f} seconds")
        print(f"Maximum time: {max_time:.4f} seconds")
        print(f"{'='*60}")

    print(f"\n✅ Processing complete. Split queries saved to: {OUTPUT_FILE_PATH}")

if __name__ == "__main__":
    main()