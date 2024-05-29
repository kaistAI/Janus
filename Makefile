CUDA_VISIBLE_DEVICES=0,1
TEST_DATA="kaist-ai/Multifaceted-Bench"
RESPONSES_DIR="responses/"
EVALUATION_DIR="eval/"

run_inference_janus:
	CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) python run_inference.py \
	--model_name kaist-ai/janus-7b \
	--input_file $(TEST_DATA) \
	--output_dir $(RESPONSES_DIR) \
	--system_key system \
	--user_key prompt \
	--num_gpus 1

run_inference_janus_bo4:
	CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) python run_inference.py \
	--model_name kaist-ai/janus-7b \
	--input_file $(TEST_DATA) \
	--output_dir $(RESPONSES_DIR) \
	--system_key system \
	--user_key prompt \
	--num_gpus 1 \
	--suffix best-of-4 \
	--N 4 \
	--reward_model_name kaist-ai/janus-rm-7b \
	--reward_model_device_num 1 \
	--bf16

run_inference_gpt35:
	python run_inference_openai.py \
	--model_name gpt-3.5-turbo-0125 \
	--input_file $(TEST_DATA) \
	--output_dir $(RESPONSES_DIR) \
	--system_key system \
	--user_key prompt

run_eval_janus:
	python run_eval_openai.py \
	--model_name gpt-4-0125-preview \
	--input_file $(TEST_DATA) \
	--response_file responses/janus-7b_responses.json \
	--output_dir $(EVALUATION_DIR) \
	--user_key prompt \
	--answer_key reference_answer \
	--rubric_key rubric