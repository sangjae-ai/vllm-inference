from vllm import LLM, SamplingParams

model_name = "google/gemma3-4b-it"
local_model_path = "./gemma3-4b-it/"

llm = LLM(
    # model=model_name,     # huggingface  or
    model=local_model_path, # Local file
    tensor_parallel_size=4,
    dtype="bfloat16",
    max_model_len=4096,
    gpu_memory_utilization=0.85,
    swap_space=4,
    trust_remote_code=True  # 이미지나 커스텀 코드 모델일 경우 필요
)

# 예시 프롬프트 생성 및 텍스트 생성 예
prompt = "안녕하세요, vLLM Python API 테스트입니다."

outputs = llm.generate([prompt], SamplingParams(temperature=0.7, max_tokens=100))

print(outputs[0].outputs[0].text)