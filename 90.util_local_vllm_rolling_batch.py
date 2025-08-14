from vllm import LLM, SamplingParams

model_name = "google/gemma3-4b-it"
local_model_path = "./gemma3-4b-it/"

# model_name = "google/gemma-3-4b-it"
# local_model_path = "./Qwen2.5-7B-Instruct/"


llm = LLM(
    # model=model_name,     # huggingface  or
    model=local_model_path, # Local file
    tensor_parallel_size=4,
    dtype="bfloat16",
    max_model_len=512,
    gpu_memory_utilization=0.7,
    swap_space=8,
    trust_remote_code=True,

    # Rolling Batch 관련 설정
    enable_chunked_prefill=True,        # 청크 단위 prefill로 메모리 효율성 향상
    max_num_batched_tokens=512,        # 배치당 최대 토큰 수
    max_num_seqs=8,                   # 동시 처리 가능한 최대 시퀀스 수

    # 성능 최적화 설정
    enable_prefix_caching=True,         # 프리픽스 캐싱으로 반복 처리 최적화
    use_v2_block_manager=True,          # 향상된 블록 매니저 사용

    # 메모리 최적화
    block_size=32,                      # 메모리 블록 크기 (기본값: 16)
    preemption_mode="swap",        # 메모리 부족 시 재계산 방식
)

# 사용 예시
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512
)

# 여러 요청을 동시에 처리 (Rolling Batch의 장점 활용)
prompts = [
    "안녕하세요, 오늘 날씨는 어떤가요?",
    "Python에서 리스트와 튜플의 차이점을 설명해주세요.",
    "머신러닝의 기본 개념을 간단히 설명해주세요.",
    "AI agent에 대해서 설명 해 주세요..",
]

# 배치 처리로 효율적인 추론
outputs = llm.generate(prompts, sampling_params)

for idx,output in enumerate(outputs):
    print(f"--------------------  {idx}  --------------------")
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")
    print("-" * 50)