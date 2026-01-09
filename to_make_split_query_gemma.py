import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import time
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# --- 📜 설정 부분 ---
MODEL_NAME = "gemma-2-2b-it"
MODEL_REPO = "google/gemma-2-2b-it"  # HF repo id
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
        3.1. Sentence 1 (Context Initialization): Create a simple sentence that defines the main subject, location, or state using the most important noun/entity from the original query.
        3.2. Sentence 2 (Action/State Statement): Sentence 2 MUST contain the entire original query's core information.

        ### Output Format:
        - Output exactly TWO sentences.
        - Separate them with a period and a newline.
        - No labels, brackets, numbering, or explanations.

        ### Your Task:
        Input: "{original_query}"
        Output:"""
    return prompt


def main():
    print("Loading model... This may take a moment.")

    # 디바이스 설정 (CUDA → MPS → CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"✅ Using device: {device}")

    # HF Hub 모델 로드
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
        total_lines = len(lines)
        sample_size = min(SAMPLE_SIZE, total_lines)
        lines_to_process = random.sample(lines, sample_size)
        print(f"\n--- RANDOM SAMPLE MODE: Processing {sample_size} out of {total_lines} queries ---")
    else:
        lines_to_process = lines

    processing_times = []

    # 출력 파일 열기
    with open(OUTPUT_FILE_PATH, "w", encoding="utf-8") as f_out:
        for line in tqdm(lines_to_process, desc="Splitting Queries"):
            try:
                data = json.loads(line)
                original_query = data.get("query")

                if not original_query:
                    continue

                start_time = time.time()

                # 프롬프트 생성
                prompt = create_prompt(original_query)

                # 🔥 Gemma는 system role을 지원하지 않으므로 user 하나로 합침
                system_instruction = (
                    "You are a helpful assistant that analyzes sentences and splits them when they contain multiple sequential actions.\n\n"
                )

                messages = [
                    {
                        "role": "user",
                        "content": system_instruction + prompt,
                    }
                ]

                # Chat template 적용
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

                # 생성
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
                clean_output = tokenizer.decode(
                    gen_only, skip_special_tokens=True
                ).strip()

                if clean_output.startswith("Output:"):
                    clean_output = clean_output[7:].strip()

                end_time = time.time()

                data["query_original"] = original_query
                data["query"] = clean_output
                data["processing_time_seconds"] = round(end_time - start_time, 4)

                f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

                processing_times.append(end_time - start_time)

            except Exception as e:
                print(f"\nError processing line: {e}")
                continue

    # 통계 출력
    if processing_times:
        avg_time = sum(processing_times) / len(processing_times)
        print("\n================ Statistics ================")
        print(f"MODEL_NAME: {MODEL_NAME}")
        print(f"Total processed: {len(processing_times)}")
        print(f"Avg time/query: {avg_time:.4f} sec")

    print(f"\n✅ Done! Saved to: {OUTPUT_FILE_PATH}")


if __name__ == "__main__":
    main()