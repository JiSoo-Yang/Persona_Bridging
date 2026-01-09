from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 더 작은 모델로 교체 (속도 확인용)
model_name = "Qwen/Qwen3-1.7B"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map=None,                 # 자동배치 끔
    dtype=torch.float32,       # CPU fp32
    low_cpu_mem_usage=True
)
model.to("cpu")

# CPU 스레드 제한(과도한 스레드로 오히려 느려지는 경우 방지)
torch.set_num_threads(4)

messages = [{"role": "user", "content": "Give me a short introduction to large language model."}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to("cpu")

with torch.inference_mode():
    out = model.generate(
        **inputs,
        max_new_tokens=32,                            # 👈 과한 길이 금지
        do_sample=False,
        temperature=0.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        use_cache=True
    )

gen = out[0][len(inputs.input_ids[0]):]
print("content:", tokenizer.decode(gen, skip_special_tokens=True))