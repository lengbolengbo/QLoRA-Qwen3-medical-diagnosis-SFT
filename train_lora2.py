import json
import pandas as pd
import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq, BitsAndBytesConfig  # 添加量化配置
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training  # 添加k位训练准备
import os
import swanlab

os.environ["SWANLAB_PROJECT"]="qwen3-sft-medical"
PROMPT = "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。"
MAX_LENGTH = 2048

swanlab.config.update({
    "model": "Qwen/Qwen3-1.7B",
    "prompt": PROMPT,
    "data_max_length": MAX_LENGTH,
    "quantization": "4-bit QLoRA"  # 添加量化标识
    })

def dataset_jsonl_transfer(origin_path, new_path):
    """
    将原始数据集转换为大模型微调所需数据格式的新数据集
    """
    messages = []

    # 读取旧的JSONL文件
    with open(origin_path, "r", encoding="utf-8") as file:
        for line in file:
            # 解析每一行的json数据
            data = json.loads(line)
            input = data["question"]
            think = data["think"]
            answer = data["answer"]
            output = f"<think>{think}</think> \n {answer}"
            message = {
                "instruction": PROMPT,
                "input": f"{input}",
                "output": output,
            }
            messages.append(message)

    # 保存重构后的JSONL文件
    with open(new_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")


def process_func(example):
    """
    将数据集进行预处理
    """ 
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        f"<|im_start|>system\n{PROMPT}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = (
        instruction["attention_mask"] + response["attention_mask"] + [1]
    )
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}   


def predict(messages, model, tokenizer):
    device = "cuda"
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=MAX_LENGTH,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

# # 在modelscope上下载Qwen模型到本地目录下
# model_dir = snapshot_download("Qwen/Qwen3-1.7B", cache_dir="./", revision="master")

# 4-bit量化配置 - 核心修改
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                 # 启用4位量化
    bnb_4bit_quant_type="nf4",         # 使用nf4量化类型
    bnb_4bit_compute_dtype=torch.bfloat16,  # 计算时使用bfloat16
    bnb_4bit_use_double_quant=True,    # 启用双重量化进一步压缩
)

# Transformers加载模型权重
# 将相对路径转换为绝对路径
local_model_path = os.path.abspath("./Qwen/Qwen3-1___7B")

tokenizer = AutoTokenizer.from_pretrained(local_model_path, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    local_model_path,  # 使用绝对路径
    quantization_config=bnb_config,  # 应用量化配置
    device_map="auto",
    # torch_dtype=torch.bfloat16,
    trust_remote_code=True
)




model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

# 配置lora - 关键修改
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=64,              # 增加秩以提升性能 (原8->64)
    lora_alpha=16,     # 降低alpha (原32->16)
    lora_dropout=0.1,  ## 保持原作者dropout
    modules_to_save=["embed_tokens", "lm_head"],  ## 适配嵌入层和输出层
    bias="none"
)

model = get_peft_model(model, config)
model.print_trainable_parameters()  # 打印可训练参数数量


# 加载、处理数据集和测试集
train_dataset_path = "train.jsonl"
test_dataset_path = "val.jsonl"

train_jsonl_new_path = "train_format.jsonl"
test_jsonl_new_path = "val_format.jsonl"

if not os.path.exists(train_jsonl_new_path):
    dataset_jsonl_transfer(train_dataset_path, train_jsonl_new_path)
if not os.path.exists(test_jsonl_new_path):
    dataset_jsonl_transfer(test_dataset_path, test_jsonl_new_path)

# 得到训练集
train_df = pd.read_json(train_jsonl_new_path, lines=True)
train_ds = Dataset.from_pandas(train_df)
train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)

# 得到验证集
eval_df = pd.read_json(test_jsonl_new_path, lines=True)
eval_ds = Dataset.from_pandas(eval_df)
eval_dataset = eval_ds.map(process_func, remove_columns=eval_ds.column_names)

args = TrainingArguments(
    output_dir="./output/Qwen3-1.7B-v2",
    per_device_train_batch_size=4,      # 增加batch size (原1->4)
    per_device_eval_batch_size=1,       # 增加eval batch size(原1->2)
    gradient_accumulation_steps=4,      # 减少梯度累积步数 (原4->2)
    optim="paged_adamw_8bit",           # 使用8bit优化器节省显存
    eval_strategy="steps",
    eval_steps=100,
    logging_steps=10,
    num_train_epochs=3,                 # 增加训练轮次（原2->3）
    save_strategy="steps",
    save_steps=400,
    learning_rate=2e-4,                 # 提高学习率 (原1e-4->2e-4)
    lr_scheduler_type="cosine",         ## 学习率调整（余弦衰减代替线性衰减）
    warmup_ratio=0.1,                   ## 10%训练步数作为预热
    save_on_each_node=True,
    gradient_checkpointing=True,        # 确保启用梯度检查点
    report_to="swanlab",
    run_name="qwen3-1.7B-QLoRA-v2",
    bf16=True,                          # 使用BF16替代FP16
    max_grad_norm=0.3,                  # 梯度裁剪防止梯度爆炸
    logging_dir='./logs',               # 添加日志目录
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8  # 填充到8的倍数提升效率
    ),
)

# 训练前检查显存
print(f"训练前显存使用: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
trainer.train()

# 仅保存适配器权重
model.save_pretrained(args.output_dir)
print(f"QLoRA适配器权重已保存至 {args.output_dir}")

# 用测试集的前3条，主观看模型
test_df = pd.read_json(test_jsonl_new_path, lines=True)[:3]

test_text_list = []

for index, row in test_df.iterrows():
    instruction = row['instruction']
    input_value = row['input']

    messages = [
        {"role": "system", "content": f"{instruction}"},
        {"role": "user", "content": f"{input_value}"}
    ]

    response = predict(messages, model, tokenizer)

    response_text = f"""
    Question: {input_value}

    LLM:{response}
    """
    
    test_text_list.append(swanlab.Text(response_text))
    print(response_text)

swanlab.log({"Prediction": test_text_list})

swanlab.finish()