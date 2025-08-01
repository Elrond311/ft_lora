# train.py

import time
import json
from openai import OpenAI
from openai.openai_object import OpenAIObject
from tqdm import tqdm

# 配置区域
API_KEY          = ""
BASE_URL         = "https://llm-oneapi.bytebroad.com.cn/v1"
BASE_MODEL       = "Qwen/Qwen2.5-VL-32B-Instruct"
TRAIN_FILE_JSONL = "lora_medical.jsonl"    # 同目录

# 初始化客户端
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def upload_training_file(path: str) -> str:
    print("上传训练文件...")
    resp: OpenAIObject = client.files.create(
        file=open(path, "rb"),
        purpose="fine-tune"
    )
    file_id = resp["id"]
    print(f"文件上传完成，file_id = {file_id}")
    return file_id

def create_finetune_job(file_id: str) -> str:
    print("创建微调任务...")
    resp: OpenAIObject = client.fine_tunes.create(
        model=BASE_MODEL,
        training_file=file_id,
        n_epochs=3,
        batch_size=4,
        learning_rate_multiplier=0.1,
        prompt_loss_weight=0.01
    )
    job_id = resp["id"]
    print(f"提交成功，fine_tune_job_id = {job_id}")
    return job_id

def wait_for_completion(job_id: str) -> str:
    print("等待任务完成...")
    with tqdm(total=100, desc="微调进度", ncols=80) as pbar:
        while True:
            status_resp = client.fine_tunes.get(id=job_id)
            status = status_resp["status"]
            pct = {"pending": 0, "running": 50, "succeeded": 100}.get(status, 0)
            pbar.n = pct
            pbar.refresh()
            if status in ("succeeded", "failed"):
                break
            time.sleep(10)
    if status != "succeeded":
        raise RuntimeError(f"微调失败，状态：{status}")
    fine_tuned_model = status_resp["fine_tuned_model"]
    print(f"微调完成！模型名称：{fine_tuned_model}")
    return fine_tuned_model

def chat_test(model_name: str, prompt: str) -> None:
    print(f"测试输入：{prompt}")
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        stream=False
    )
    data = json.loads(resp.model_dump_json())
    reply = data["choices"][0]["message"]["content"].strip()
    print(f"模型回复：{reply}")

def main():
    file_id = upload_training_file(TRAIN_FILE_JSONL)
    job_id  = create_finetune_job(file_id)
    model_name = wait_for_completion(job_id)
    chat_test(model_name, "请简要说明高血压的常见并发症有哪些？")

if __name__ == "__main__":
    main()
