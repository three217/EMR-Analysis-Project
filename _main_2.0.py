# EMR Struct Agent v2.0 | For SCI Reproduction
import os
import re
import json
import zipfile
import torch
import pandas as pd
import streamlit as st
from io import BytesIO
from PIL import Image
from datetime import datetime
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

# ===================== 【1. 全局配置区】=====================
MODEL_DTYPE = "fp16"
DEV_PASSWORD = "dev"  
DEFAULT_EXTRACT_FIELDS = """
患者基础档案,首次病程记录,查房记录,医嘱与用药信息,
辅助检查与检验报告,医疗文书与知情同意,护理与生命体征记录,出院与诊疗计划
"""
if "model_path" not in st.session_state:
    st.session_state.model_path = ""  # 大模型路径
if "lora_output_path" not in st.session_state:
    st.session_state.lora_output_path = "./lora_checkpoints"  # 相对路径，通用
if "lora_load_path" not in st.session_state:
    st.session_state.lora_load_path = ""  # LoRA加载路径
if "record_archive_path" not in st.session_state:
    st.session_state.record_archive_path = "./病历归档"  # 相对路径，通用
if "txt_case_folder" not in st.session_state:
    st.session_state.txt_case_folder = ""  # 病例文件夹

# 动态赋值（从会话状态读取，永远不会读死路径）
MODEL_PATH = st.session_state.model_path
LORA_OUTPUT_PATH = st.session_state.lora_output_path
LORA_LOAD_PATH = st.session_state.lora_load_path
RECORD_ARCHIVE_PATH = st.session_state.record_archive_path
TXT_CASE_FOLDER = st.session_state.txt_case_folder

AGENT_CONFIG = {
    "enable_dynamic_adapt": True,
    "enable_self_check": True,
    "enable_doc_type_split": True,
    "enable_auto_ocr": True,
    "enable_medical_check": True,
    "enable_desensitize": True
}
TEMPERATURE = 0.1
MAX_NEW_TOKENS = 4096
MAX_CONTEXT_LENGTH = 8192

# ===================== 【2. 全局初始化】=====================
if "dev_mode" not in st.session_state:
    st.session_state.dev_mode = False
if "ocr_text" not in st.session_state:
    st.session_state.ocr_text = ""
if "debug_log" not in st.session_state:
    st.session_state.debug_log = []
if "uploaded_images" not in st.session_state:
    st.session_state.uploaded_images = []
if "case_info" not in st.session_state:
    st.session_state.case_info = {"name": "未知", "case_no": "未知"}
if "train_data" not in st.session_state:
    st.session_state.train_data = []

# ===================== 【3. OCR工具集成（表格专项优化）】=====================
@st.cache_resource(show_spinner="正在加载OCR识别引擎...")
def load_ocr_engine():
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(
            lang="ch",
            use_angle_cls=True,
            show_log=False,
            enable_mkldnn=True,
            rec_batch_num=4,
            layout=True
        )
        return ocr, "✅ OCR引擎加载成功（已开启表格识别）"
    except Exception as e:
        return None, f"❌ OCR引擎加载失败：{str(e)}"

def ocr_image(ocr_engine, img: Image.Image) -> str:
    try:
        img = img.convert("RGB")
        result = ocr_engine.ocr(img, cls=True, layout=True)
        ocr_text = ""
        if result and result[0]:
            for line in result[0]:
                text = line[1][0]
                if any(keyword in text for keyword in ["检验", "项目", "结果", "参考值", "医嘱", "体温", "日期"]):
                    ocr_text += f"【表格内容】{text}\n"
                else:
                    ocr_text += f"{text}\n"
        return ocr_text
    except Exception as e:
        st.session_state.debug_log.append(f"OCR识别失败：{str(e)}")
        return ""

# ===================== 【4. 智能体Prompt（Few-shot优化+表格专项）】=====================
class EMRAgentPrompt:
    def __init__(self):
        self.default_fields = DEFAULT_EXTRACT_FIELDS.replace("\n", "").strip()
        self.enable_dynamic = AGENT_CONFIG["enable_dynamic_adapt"]
        self.enable_check = AGENT_CONFIG["enable_self_check"]
        self.enable_doc_split = AGENT_CONFIG["enable_doc_type_split"]

    def get_analysis_system_prompt(self) -> str:
        return """
你是三甲医院病案管理专家，严格遵循以下规则完成任务，零差错拆分病历文书：
1.  精准识别输入病历的所有临床文书类型，包括但不限于：
    患者基础档案、首次病程记录、查房记录、疑难病例讨论、会诊记录、
    长期医嘱单、临时医嘱单、检验报告单、检查报告单、体温单、护理记录、
    知情同意书、授权委托书、医保告知书、24小时入出院记录、出院记录
2.  每个文书类型必须单独拆分为一个大类，**禁止将多个文书的内容合并到同一个大类下**。
3.  每个大类下必须生成对应的细分临床字段，字段必须贴合文书实际内容，符合临床规范，禁止遗漏。
4.  同时识别患者姓名、病案号，用于病历归档。
5.  仅输出标准JSON格式，结构如下：
    {
        "patient_info": {
            "patient_name": "患者姓名",
            "case_no": "病案号"
        },
        "extract_categories": {
            "文书类型1": ["细分字段1", "细分字段2", "..."],
            "文书类型2": ["细分字段1", "细分字段2", "..."]
        }
    }
6.  禁止输出任何解释、说明、备注，仅输出纯JSON字符串，禁止编造内容。
"""

    def get_analysis_user_prompt(self, emr_text: str) -> str:
        return f"""
以下是完整的住院病历文本：
{emr_text}
请严格按照规则拆分所有临床文书类型，生成对应抽取字段，仅输出标准JSON。
"""

    def get_extract_system_prompt(self, extract_categories: dict) -> str:
        category_str = json.dumps(extract_categories, ensure_ascii=False, indent=2)
        return f"""
你是一名专业的三甲医院电子病历结构化抽取专家，严格遵循以下核心规则，零差错完成抽取：
1.  从输入的完整电子病历中，精准抽取以下指定文书类型和细分字段的全部内容，输出标准JSON格式，JSON的key为文书类型名称，value为该文书下的字段键值对。
2.  需抽取的文书类型与细分字段：
{category_str}
3.  必须完整覆盖病历中所有对应字段的内容，无对应内容的字段，value统一填充为"无"，禁止留空、禁止编造。
4.  【表格类内容专项规则】：
    - 检验报告单、医嘱单、体温单等表格内容，必须逐行拆分为「项目名称-结果-参考值-单位-异常提示」的键值对。
    - 例如：白细胞 9.64×10^9/L 参考值3.5-9.5×10^9/L 结果异常↑，必须拆分为：
        "白细胞": "9.64×10^9/L",
        "白细胞参考值": "3.5-9.5×10^9/L",
        "白细胞异常提示": "↑"
    - 禁止将表格内容合并为大段文本。
5.  抽取内容100%忠于原文，不得修改、补充任何病历中不存在的信息，不得篡改医学术语。
6.  仅输出纯JSON字符串，禁止任何额外内容。

-------------------
【正确示例】
输入病历文本：
患者姓名：王玉祥，性别：男，年龄：65岁，病案号：20000xxx
2026-03-09 首次病程记录：确诊左肺下叶鳞癌3月余，行4周期化疗，方案：替雷利珠单抗200mg D0+紫杉醇200mg D1+卡铂400mg D2
【表格内容】白细胞 9.64×10^9/L 参考值3.5-9.5×10^9/L 结果异常↑

正确输出JSON：
{{
    "患者基础档案": {{
        "姓名": "王玉祥",
        "性别": "男",
        "年龄": "65岁",
        "病案号": "20000xxx"
    }},
    "首次病程记录": {{
        "确诊疾病": "左肺下叶鳞癌",
        "确诊时长": "3月余",
        "已行化疗周期": "4周期",
        "化疗方案": "替雷利珠单抗200mg D0+紫杉醇200mg D1+卡铂400mg D2"
    }},
    "辅助检查结果": {{
        "白细胞": "9.64×10^9/L",
        "白细胞参考值": "3.5-9.5×10^9/L",
        "白细胞异常提示": "↑"
    }}
}}
-------------------
"""

    def get_extract_user_prompt(self, emr_text: str) -> str:
        return f"""
以下是完整的住院病历文本：
{emr_text}
请严格按照规则完成分文书结构化抽取，仅输出标准JSON。
"""

    def get_check_system_prompt(self) -> str:
        return """
你是病历结构化结果校验专家，严格遵循以下规则：
1.  对比原始病历和抽取结果，校验：
    - 是否所有文书类型都被单独拆分，有无大段内容合并
    - 所有字段内容是否100%忠于原文，有无编造、遗漏
    - JSON格式是否标准，字段命名是否符合临床规范
    - 表格类内容是否已拆分为键值对
2.  对错误内容进行修正，最终仅输出修正后的标准JSON字符串，无额外内容。
"""

    def get_check_user_prompt(self, emr_text: str, extract_result: dict) -> str:
        return f"""
原始病历文本：
{emr_text}

抽取的结构化结果：
{json.dumps(extract_result, ensure_ascii=False, indent=4)}

请完成校验与修正，仅输出修正后的标准JSON。
"""

    def get_medical_check_system_prompt(self) -> str:
        return """
你是三甲医院临床质控专家，严格遵循以下规则：
1.  对比原始病历和结构化结果，识别以下异常信息：
    - 检验结果的异常值（升高↑/降低↓）
    - 诊断与治疗方案的逻辑矛盾
    - 病历前后信息不一致
    - 病历缺失的核心文书
2.  仅输出标准JSON格式，结构如下：
{
    "abnormal_items": ["异常项1", "异常项2"],
    "missing_docs": ["缺失的文书1", "缺失的文书2"],
    "quality_check": "通过/不通过"
}
3.  禁止额外内容，仅输出JSON。
"""

    def get_medical_check_user_prompt(self, emr_text: str, extract_result: dict) -> str:
        return f"""
原始病历文本：{emr_text}
结构化结果：{json.dumps(extract_result, ensure_ascii=False)}
"""

# ===================== 【5. 医疗隐私脱敏】=====================
def desensitize_emr_data(struct_result: dict, emr_text: str) -> tuple[dict, str]:
    """医疗隐私数据脱敏"""
    desensitize_rules = [
        (re.compile(r"姓名[：:]\s*([\u4e00-\u9fa5]{2,4})"), lambda m: f"姓名：{m.group(1)[0]}某某"),
        (re.compile(r"病案号[：:]\s*(\d+)"), lambda m: f"病案号：{m.group(1)[:4]}xxxx"),
        (re.compile(r"住院号[：:]\s*(\d+)"), lambda m: f"住院号：{m.group(1)[:4]}xxxx"),
        (re.compile(r"1[3-9]\d{9}"), lambda m: m.group(0)[:3] + "****" + m.group(0)[7:]),
        (re.compile(r"\d{17}[\dXx]"), lambda m: m.group(0)[:6] + "********" + m.group(0)[14:]),
    ]
    
    desensitized_result = json.loads(json.dumps(struct_result, ensure_ascii=False))
    for doc_type, content in desensitized_result.items():
        if isinstance(content, dict):
            for field, value in content.items():
                for pattern, repl in desensitize_rules:
                    value = pattern.sub(repl, str(value))
                content[field] = value
    
    desensitized_text = emr_text
    for pattern, repl in desensitize_rules:
        desensitized_text = pattern.sub(repl, desensitized_text)
    
    return desensitized_result, desensitized_text

# ===================== 【6. 模型加载】=====================
@st.cache_resource(show_spinner="正在加载医疗病历结构化智能体，请稍候...")
def load_local_model():
    if not st.session_state.model_path:
        return None, None, "⚠️ 请先在侧边栏填写大模型路径！"
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.cuda.empty_cache()

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map={"": 0},
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            offload_folder=None
        )

        # ========== 新增：加载LoRA权重（核心改动，仅新增这几行） ==========
        import os
        from peft import PeftModel
        if os.path.exists(LORA_LOAD_PATH):
            model = PeftModel.from_pretrained(model, LORA_LOAD_PATH)
            model = model.merge_and_unload()  # 可选：合并LoRA权重到主模型（节省显存/提升速度）
            lora_info = " | ✅ 已加载LoRA权重"
        else:
            lora_info = " | ❌ 未找到LoRA权重，使用基础模型"
        # ==============================================================

        model = model.to(device)
        model.eval()
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_grad_enabled(False)

        gpu_info = ""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            mem_used = torch.cuda.memory_allocated()/1024/1024/1024
            gpu_info = f" | 显卡：{gpu_name} | 显存已用：{mem_used:.2f}GB"

        # 新增lora_info到状态提示
        return model, tokenizer, f"✅ 智能体加载成功，CUDA可用：{torch.cuda.is_available()}{gpu_info}{lora_info}"
    except Exception as e:
        return None, None, f"❌ 智能体加载失败：{str(e)}"

# ===================== 【7. 智能体核心推理】=====================
def agent_inference(model, tokenizer, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt", truncation=True, max_length=MAX_CONTEXT_LENGTH).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=0.95,
            repetition_penalty=1.05,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return response.replace("```json", "").replace("```", "").strip()

def extract_emr_struct_agent(model, tokenizer, emr_text: str, progress_bar) -> tuple[dict, dict]:
    prompt_handler = EMRAgentPrompt()
    try:
        # 第一步：文书拆分
        progress_bar.progress(20, text="📄 正在拆分病历文书...")
        analysis_system = prompt_handler.get_analysis_system_prompt()
        analysis_user = prompt_handler.get_analysis_user_prompt(emr_text)
        analysis_response = agent_inference(model, tokenizer, analysis_system, analysis_user)
        analysis_result = json.loads(analysis_response)
        
        st.session_state.case_info = analysis_result["patient_info"]
        extract_categories = analysis_result["extract_categories"]
        
        if st.session_state.dev_mode:
            st.session_state.debug_log.append(f"文书拆分结果：{json.dumps(extract_categories, ensure_ascii=False)}")
        
        doc_types = list(extract_categories.keys())
        st.info(f"✅ 已识别病历文书类型：{'、'.join(doc_types)}")

        # 第二步：分文书抽取
        progress_bar.progress(50, text="🔍 正在执行分文书结构化抽取...")
        extract_system = prompt_handler.get_extract_system_prompt(extract_categories)
        extract_user = prompt_handler.get_extract_user_prompt(emr_text)
        extract_response = agent_inference(model, tokenizer, extract_system, extract_user)
        extract_result = json.loads(extract_response)
        
        if st.session_state.dev_mode:
            st.session_state.debug_log.append(f"抽取原始结果：{extract_response}")

        # 第三步：自我校验
        progress_bar.progress(70, text="✅ 正在校验抽取结果...")
        if prompt_handler.enable_check:
            check_system = prompt_handler.get_check_system_prompt()
            check_user = prompt_handler.get_check_user_prompt(emr_text, extract_result)
            final_response = agent_inference(model, tokenizer, check_system, check_user)
            final_result = json.loads(final_response)
        else:
            final_result = extract_result

        # 第四步：医疗逻辑校验
        medical_check_result = {}
        if AGENT_CONFIG["enable_medical_check"]:
            progress_bar.progress(85, text="⚕️ 正在执行医疗逻辑校验...")
            medical_check_system = prompt_handler.get_medical_check_system_prompt()
            medical_check_user = prompt_handler.get_medical_check_user_prompt(emr_text, final_result)
            medical_check_response = agent_inference(model, tokenizer, medical_check_system, medical_check_user)
            medical_check_result = json.loads(medical_check_response)

        progress_bar.progress(100, text="🎉 处理完成！")
        return final_result, medical_check_result
    except Exception as e:
        st.session_state.debug_log.append(f"处理失败：{str(e)}")
        raise Exception(f"智能体处理失败：{str(e)}")

# ===================== 【8. 病历归档】=====================
def archive_case_images(images: list, case_info: dict, struct_result: dict, desensitized: bool = False):
    try:
        case_no = case_info["case_no"] if case_info["case_no"] != "未知" else "未知病案号"
        patient_name = case_info["patient_name"] if case_info["patient_name"] != "未知" else "未知患者"
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = "_脱敏版" if desensitized else ""
        archive_folder = os.path.join(RECORD_ARCHIVE_PATH, f"{case_no}_{patient_name}_{time_str}{suffix}")
        os.makedirs(archive_folder, exist_ok=True)

        img_folder = os.path.join(archive_folder, "原始病历影像")
        os.makedirs(img_folder, exist_ok=True)
        for idx, img_info in enumerate(images):
            img_path = os.path.join(img_folder, f"病历影像_{idx+1}_{img_info['name']}")
            img_info["image"].save(img_path)

        json_path = os.path.join(archive_folder, "病历结构化结果.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(struct_result, f, ensure_ascii=False, indent=4)

        ocr_path = os.path.join(archive_folder, "病历OCR文本.txt")
        with open(ocr_path, "w", encoding="utf-8") as f:
            f.write(st.session_state.ocr_text)

        return archive_folder
    except Exception as e:
        st.session_state.debug_log.append(f"归档失败：{str(e)}")
        return None
# ===================== 【新增：TXT文件夹自动生成训练数据】=====================
def load_txt_to_train_data(folder_path):
    train_data = []
    if not os.path.exists(folder_path):
        return train_data

    for fname in os.listdir(folder_path):
        if fname.lower().endswith(".txt"):
            try:
                with open(os.path.join(folder_path, fname), "r", encoding="utf-8") as f:
                    text = f.read().strip()
                if len(text) < 100:
                    continue

                # 自动跑一遍结构化，生成标准target_json
                dummy_progress = None
                try:
                    target_json, _ = extract_emr_struct_agent(
                        model, tokenizer, text, dummy_progress
                    )
                except:
                    target_json = {}

                train_data.append({
                    "emr_text": text,
                    "target_json": json.dumps(target_json, ensure_ascii=False)
                })
            except:
                continue
    return train_data
# ===================== 【9. LoRA训练集成】=====================
def train_lora_simple(model, tokenizer, train_data: list, epochs: int = 3, batch_size: int = 1, lr: float = 2e-4, lora_rank: int = 8):
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
        from datasets import Dataset
        import torch

        # 【核心修复1】完整的文本编码逻辑（生成input_ids）
        prompt_handler = EMRAgentPrompt()
        
        def tokenize_function(examples):
            """对文本进行完整编码，生成model需要的input_ids/labels等"""
            # 构建对话格式
            texts = []
            for emr, target in zip(examples["emr_text"], examples["target_json"]):
                try:
                    target_dict = json.loads(target)
                    extract_categories = target_dict.get("extract_categories", {
                        "患者基础档案": ["姓名", "性别", "年龄", "病案号"],
                        "首次病程记录": ["确诊疾病", "治疗方案"],
                        "辅助检查结果": ["检验项目", "结果", "参考值"]
                    })
                except:
                    extract_categories = {
                        "患者基础档案": ["姓名", "性别", "年龄", "病案号"],
                        "首次病程记录": ["确诊疾病", "治疗方案"],
                        "辅助检查结果": ["检验项目", "结果", "参考值"]
                    }
                
                # 构建标准Chat模板
                system_prompt = prompt_handler.get_extract_system_prompt(extract_categories)
                user_prompt = prompt_handler.get_extract_user_prompt(emr)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": target}
                ]
                
                # 应用tokenizer的chat模板并编码
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                    padding=False,
                    truncation=False
                )
                texts.append(text)
            
            # 关键：对文本进行编码，生成input_ids（模型必需）
            encoding = tokenizer(
                texts,
                truncation=True,
                max_length=2048,  # 适配消费级GPU，避免超长
                padding="max_length",
                return_tensors="pt"
            )
            
            # 构建labels（与input_ids一致，LM训练必需）
            encoding["labels"] = encoding["input_ids"].clone()
            
            # 返回numpy格式（datasets要求）
            return {
                "input_ids": encoding["input_ids"].numpy(),
                "attention_mask": encoding["attention_mask"].numpy(),
                "labels": encoding["labels"].numpy()
            }

        # 【核心修复2】正确处理数据集
        # 1. 转换为Dataset格式
        dataset = Dataset.from_list(train_data)
        # 2. 编码生成input_ids（关键步骤）
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,  # 批量处理，提升速度
            remove_columns=["emr_text", "target_json"]  # 删除不需要的列
        )
        # 3. 设置数据格式为torch张量
        tokenized_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"]
        )

        # 【核心修复3】适配的DataCollator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # 因果语言模型，关闭MLM
            pad_to_multiple_of=8  # 对齐显存，提升效率
        )

        # 配置LoRA（兼容Qwen2.5）
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            inference_mode=False  # 训练模式
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()  # 打印可训练参数

        # 训练参数（适配消费级GPU，避免显存溢出）
        training_args = TrainingArguments(
            output_dir=LORA_OUTPUT_PATH,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,  # 梯度累积，等效增大批次
            learning_rate=lr,
            logging_steps=5,  # 更频繁打印日志，方便监控
            save_steps=10,
            fp16=True,  # 混合精度训练
            remove_unused_columns=False,
            push_to_hub=False,
            report_to="none",  # 关闭wandb报告
            save_total_limit=2,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            weight_decay=0.01,
            # 解决显存/编码问题的配置
            gradient_checkpointing=True,  # 梯度检查点，节省显存
            max_grad_norm=1.0,  # 梯度裁剪，防止梯度爆炸
            dataloader_pin_memory=False,  # 关闭pin_memory，适配低配GPU
            skip_memory_metrics=True  # 跳过内存指标计算，加快训练
        )

        # 启动训练
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        # 清空显存，避免溢出
        torch.cuda.empty_cache()
        # 执行训练
        trainer.train()

        # 保存模型
        save_path = os.path.join(LORA_OUTPUT_PATH, "final_lora")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)  # 同时保存tokenizer
        return save_path
    
    except Exception as e:
        st.session_state.debug_log.append(f"LoRA训练失败：{str(e)}")
        # 打印完整报错栈，方便调试
        import traceback
        st.session_state.debug_log.append(f"详细报错信息：\n{traceback.format_exc()}")
        raise Exception(f"LoRA训练失败：{str(e)}")

# ===================== 【10. 批量处理工具函数】=====================
def batch_process_emr(model, tokenizer, file_list, progress_bar):
    """批量处理病历文件"""
    batch_results = []
    total_files = len(file_list)
    
    for idx, file in enumerate(file_list):
        try:
            # 读取文件内容
            if file.type == "text/plain":
                emr_text = file.read().decode("utf-8")
            elif file.type in ["image/jpeg", "image/png"]:
                img = Image.open(file)
                ocr_engine, _ = load_ocr_engine()
                emr_text = ocr_image(ocr_engine, img) if ocr_engine else ""
            else:
                st.warning(f"跳过不支持的文件类型：{file.name}")
                continue
            
            # 单文件结构化处理
            sub_progress = st.progress(0, text=f"处理文件 {idx+1}/{total_files}：{file.name}")
            struct_res, medical_res = extract_emr_struct_agent(model, tokenizer, emr_text, sub_progress)
            
            # 保存结果
            batch_results.append({
                "file_name": file.name,
                "struct_result": struct_res,
                "medical_check": medical_res,
                "raw_text": emr_text
            })
            
            # 更新总进度
            progress_bar.progress((idx+1)/total_files, text=f"已完成 {idx+1}/{total_files} 文件处理")
            sub_progress.empty()
            
        except Exception as e:
            st.error(f"处理文件 {file.name} 失败：{str(e)}")
            batch_results.append({
                "file_name": file.name,
                "error": str(e),
                "struct_result": {},
                "medical_check": {}
            })
    
    return batch_results

def zip_batch_results(batch_results):
    """打包批量处理结果为ZIP文件"""
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # 汇总结果
        summary_data = []
        for res in batch_results:
            summary_item = {
                "文件名": res["file_name"],
                "处理状态": "成功" if "error" not in res else "失败",
                "错误信息": res.get("error", ""),
                "识别文书数": len(res["struct_result"]) if "struct_result" in res else 0
            }
            summary_data.append(summary_item)
        
        # 写入汇总表
        summary_df = pd.DataFrame(summary_data)
        zip_file.writestr("批量处理汇总表.csv", summary_df.to_csv(index=False, encoding="utf-8-sig"))
        
        # 写入单个文件结果
        for res in batch_results:
            file_name = res["file_name"].replace("/", "_").replace("\\", "_")
            
            # 结构化结果JSON
            if "struct_result" in res and res["struct_result"]:
                json_str = json.dumps(res["struct_result"], ensure_ascii=False, indent=4)
                zip_file.writestr(f"结构化结果_{file_name}.json", json_str)
            
            # 医疗校验结果
            if "medical_check" in res and res["medical_check"]:
                check_str = json.dumps(res["medical_check"], ensure_ascii=False, indent=4)
                zip_file.writestr(f"医疗校验_{file_name}.json", check_str)
            
            # 原始文本
            if "raw_text" in res and res["raw_text"]:
                zip_file.writestr(f"原始文本_{file_name}.txt", res["raw_text"])
    
    zip_buffer.seek(0)
    return zip_buffer

# ===================== 【11. Streamlit界面】=====================
def main():
    # 以代码所在目录为基准，生成绝对路径，彻底解决相对路径错位
    BASE_DIR = Path(__file__).parent.resolve()

    # 统一转为绝对路径，同步更新会话状态
    global RECORD_ARCHIVE_PATH, LORA_OUTPUT_PATH
    if RECORD_ARCHIVE_PATH and not os.path.isabs(RECORD_ARCHIVE_PATH):
        RECORD_ARCHIVE_PATH = os.path.join(BASE_DIR, RECORD_ARCHIVE_PATH.lstrip('./'))
        st.session_state.record_archive_path = RECORD_ARCHIVE_PATH
    if LORA_OUTPUT_PATH and not os.path.isabs(LORA_OUTPUT_PATH):
        LORA_OUTPUT_PATH = os.path.join(BASE_DIR, LORA_OUTPUT_PATH.lstrip('./'))
        st.session_state.lora_output_path = LORA_OUTPUT_PATH

    # 带权限兜底的安全创建函数
    def safe_mkdir(path):
        if not path:
            return
        try:
            os.makedirs(path, exist_ok=True)
        except PermissionError:
            st.warning(f"⚠️ 无权限写入 {path}，自动切换到用户目录下的安全路径")
            # 兜底到用户目录（Windows 100%有权限）
            fallback_root = os.path.join(os.path.expanduser("~"), "EMR_Struct_Agent")
            os.makedirs(fallback_root, exist_ok=True)
            fallback_path = os.path.join(fallback_root, os.path.basename(path))
            os.makedirs(fallback_path, exist_ok=True)
            # 同步更新全局变量和会话状态
            if path == RECORD_ARCHIVE_PATH:
                RECORD_ARCHIVE_PATH = fallback_path
                st.session_state.record_archive_path = fallback_path
            elif path == LORA_OUTPUT_PATH:
                LORA_OUTPUT_PATH = fallback_path
                st.session_state.lora_output_path = fallback_path

    # 执行文件夹创建
    safe_mkdir(RECORD_ARCHIVE_PATH)
    safe_mkdir(LORA_OUTPUT_PATH)

    st.set_page_config(
        page_title="住院病历结构化智能体",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 【新增】初始化关键变量，避免未定义
    struct_result = {}
    medical_check_result = {}

    st.markdown("""
    <style>
    :root {
        --primary-color: #0066cc;
        --secondary-color: #0099cc;
        --light-bg: #f5f9fc;
        --card-bg: #ffffff;
        --text-main: #333333;
        --text-secondary: #666666;
        --success-color: #00b42a;
        --warning-color: #ff7d00;
        --error-color: #e53935;
    }
    * {
        font-family: "Microsoft YaHei", "SimHei", sans-serif;
    }
    .main-title {
        text-align: center;
        color: var(--primary-color);
        font-weight: 800;
        margin-bottom: 0.3rem;
        font-size: 2.2rem;
    }
    .sub-title {
        text-align: center;
        color: var(--text-secondary);
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    .section-title {
        color: var(--primary-color);
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 0.8rem;
        font-size: 1.3rem;
        border-left: 4px solid var(--primary-color);
        padding-left: 0.8rem;
    }
    .card-container {
        background-color: var(--card-bg);
        border-radius: 0.8rem;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 2px 12px rgba(0,102,204,0.08);
        border: 1px solid #e8f4fc;
    }
    .field-card {
        background-color: var(--light-bg);
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 0.8rem;
        border-left: 3px solid var(--secondary-color);
    }
    .field-key {
        font-weight: 700;
        color: var(--primary-color);
        font-size: 0.95rem;
        margin-bottom: 0.3rem;
    }
    .field-value {
        color: var(--text-main);
        white-space: pre-wrap;
        margin: 0;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    .stButton>button {
        background-color: var(--primary-color);
        color: white;
        border-radius: 0.5rem;
        font-weight: 700;
        border: none;
        padding: 0.6rem 1.2rem;
        box-shadow: 0 2px 8px rgba(0,102,204,0.2);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: var(--secondary-color);
        box-shadow: 0 4px 12px rgba(0,102,204,0.3);
        transform: translateY(-1px);
    }
    .css-1d391kg {
        background-color: var(--light-bg);
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 0.5rem 0.5rem 0 0;
        padding: 0.6rem 1rem;
        font-weight: 600;
        color: var(--text-secondary);
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--primary-color) !important;
        color: white !important;
    }
    .stDataFrame {
        border-radius: 0.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='main-title'>🏥 住院病历全量结构化智能体</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>基于Qwen2.5本地部署 | 多文书自动拆分 | 医疗逻辑校验 | 隐私脱敏 | LoRA训练集成</p>", unsafe_allow_html=True)

    model, tokenizer, model_status = load_local_model()
    ocr_engine, ocr_status = load_ocr_engine() if AGENT_CONFIG["enable_auto_ocr"] else (None, "OCR功能已关闭")

    # ===================== 侧边栏 =====================
    with st.sidebar:
        st.markdown("<div class='section-title'>📂 路径配置（必填）</div>", unsafe_allow_html=True)
        st.warning("⚠️ 请先填写以下路径，否则模型无法加载！")
    
        # 大模型路径
        new_model_path = st.text_input(
            "大模型存放路径（如：./Qwen2.5-3B-Instruct）", 
            value=st.session_state.model_path,
            placeholder="请输入模型文件夹绝对/相对路径"
        )
        if new_model_path != st.session_state.model_path:
            st.session_state.model_path = new_model_path
            st.info("✅ 模型路径已更新，刷新页面生效")
    
        # LoRA加载路径
        new_lora_load_path = st.text_input(
            "LoRA权重加载路径（如：./lora_checkpoints/final_lora）", 
            value=st.session_state.lora_load_path,
            placeholder="请输入LoRA权重文件夹路径"
        )
        if new_lora_load_path != st.session_state.lora_load_path:
            st.session_state.lora_load_path = new_lora_load_path
            st.info("✅ LoRA加载路径已更新，刷新页面生效")
    
        # LoRA输出路径
        new_lora_output_path = st.text_input(
            "LoRA训练输出路径", 
            value=st.session_state.lora_output_path,
            placeholder="默认相对路径：./lora_checkpoints"
        )
        if new_lora_output_path != st.session_state.lora_output_path:
            st.session_state.lora_output_path = new_lora_output_path
            st.info("✅ LoRA输出路径已更新")
    
        # 病历归档路径
        new_record_archive_path = st.text_input(
            "病历归档路径", 
            value=st.session_state.record_archive_path,
            placeholder="默认相对路径：./病历归档"
        )
        if new_record_archive_path != st.session_state.record_archive_path:
            st.session_state.record_archive_path = new_record_archive_path
            st.info("✅ 归档路径已更新")
        
        # 测试病例文件夹路径
        new_txt_case_folder = st.text_input(
            "测试病例文件夹路径", 
            value=st.session_state.txt_case_folder,
            placeholder="请输入病例TXT文件夹路径"
        )
        if new_txt_case_folder != st.session_state.txt_case_folder:
            st.session_state.txt_case_folder = new_txt_case_folder
            st.info("✅ 测试病例路径已更新")
            st.markdown("<div class='section-title'>⚙️ 系统状态</div>", unsafe_allow_html=True)
        if model:
            st.success(model_status)
            st.info(f"📦 模型路径：{MODEL_PATH}")
            st.info(f"🔍 加载精度：fp8")
        else:
            st.error(model_status)
        
        st.divider()
        st.markdown(f"📷 OCR引擎状态：{ocr_status}")

        st.divider()
        st.markdown("<div class='section-title'>🤖 智能体配置</div>", unsafe_allow_html=True)
        for key, value in AGENT_CONFIG.items():
            st.markdown(f"- {key}: {'✅ 开启' if value else '❌ 关闭'}")

        st.divider()
        st.markdown("<div class='section-title'>🔧 开发者模式</div>", unsafe_allow_html=True)
        dev_password = st.text_input("开发者密码（默认：dev）", type="password")
        if "dev_password" not in st.session_state:
            st.session_state.dev_password = "dev"
        if dev_password == st.session_state.dev_password:
            st.session_state.dev_mode = True
        st.success("✅ 已进入开发者模式")
        new_pwd = st.text_input("修改新密码", type="password", key="new_pwd")
        if st.button("确认修改密码") and new_pwd:
            st.session_state.dev_password = new_pwd
            st.success("✅ 密码修改成功！")
            
        elif dev_password != "" and dev_password != st.session_state.dev_password:
            st.error("❌ 密码错误")
            st.session_state.dev_mode = False

        if st.session_state.dev_mode:
            st.divider()
            st.markdown("<div class='section-title'>📋 调试面板</div>", unsafe_allow_html=True)
            global TEMPERATURE
            TEMPERATURE = st.slider("推理温度值", min_value=0.0, max_value=1.0, value=TEMPERATURE, step=0.05)
            global MAX_NEW_TOKENS
            MAX_NEW_TOKENS = st.number_input("最大生成长度", min_value=1024, max_value=8192, value=MAX_NEW_TOKENS, step=512)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("清空调试日志", use_container_width=True):
                    st.session_state.debug_log = []
            with col2:
                if st.button("清空GPU缓存", use_container_width=True):
                    torch.cuda.empty_cache()
                    st.success("✅ 缓存已清空")
            
            st.markdown("#### 调试日志")
            if st.session_state.debug_log:
                for log in st.session_state.debug_log:
                    st.code(log, language="text")
            else:
                st.info("暂无调试日志")

        st.divider()
        st.markdown("<div class='section-title'>📋 默认抽取大类</div>", unsafe_allow_html=True)
        for field in DEFAULT_EXTRACT_FIELDS.replace("\n", "").split(","):
            st.markdown(f"- {field.strip()}")

    # ===================== 主界面：三大Tab =====================
    # 【核心修复】显式创建并完整实现三个Tab
    main_tab1, main_tab2, main_tab3 = st.tabs(["📝 病历结构化推理", "🎯 LoRA模型训练", "📦 批量病历处理"])

    # -------------------- Tab1：病历结构化推理 --------------------
    with main_tab1:
        st.markdown("<div class='card-container'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>📥 病历输入</div>", unsafe_allow_html=True)
        input_tab1, input_tab2, input_tab3 = st.tabs(["📝 粘贴病历文本", "📂 上传TXT病历文件", "📷 上传病历影像图片"])

        emr_text = ""
        uploaded_images = []

        with input_tab1:
            emr_text = st.text_area(
                "请粘贴完整病历文本",
                height=200,
                placeholder="粘贴完整住院病历内容，支持所有临床文书类型，智能体将自动拆分文书并完成结构化抽取...",
                label_visibility="visible"
            )

        with input_tab2:
            uploaded_txt = st.file_uploader("上传TXT格式病历文件", type=["txt"], accept_multiple_files=True)
            if uploaded_txt:
                txt_content_list = []
                for txt_file in uploaded_txt:
                    try:
                        content = txt_file.read().decode("utf-8")
                        txt_content_list.append(content)
                        st.success(f"✅ 文件上传成功：{txt_file.name}")
                        with st.expander(f"查看《{txt_file.name}》内容"):
                            st.text(content)
                    except Exception as e:
                        st.error(f"文件《{txt_file.name}》读取失败：{str(e)}")
                emr_text = "\n\n".join(txt_content_list)

        with input_tab3:
            st.markdown("#### 上传病历影像图片（支持JPG/PNG，自动OCR识别）")
            uploaded_img_files = st.file_uploader(
                "可多选上传，支持病程记录、检验报告、医嘱单、知情同意书等所有病历影像",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True
            )
            if uploaded_img_files:
                uploaded_images = []
                ocr_text_list = []
                st.markdown("#### 识别结果预览")
                for img_file in uploaded_img_files:
                    try:
                        img = Image.open(img_file)
                        uploaded_images.append({
                            "name": img_file.name,
                            "image": img
                        })
                        if ocr_engine:
                            img_ocr_text = ocr_image(ocr_engine, img)
                            ocr_text_list.append(img_ocr_text)
                            st.success(f"✅ 影像识别完成：{img_file.name}")
                            with st.expander(f"查看《{img_file.name}》OCR文本"):
                                st.text(img_ocr_text)
                    except Exception as e:
                        st.error(f"图片《{img_file.name}》处理失败：{str(e)}")
                if ocr_text_list:
                    emr_text = "\n\n".join(ocr_text_list)
                    st.session_state.ocr_text = emr_text
                    st.info("✅ 所有影像OCR文本已自动合并，可直接点击处理")
            st.session_state.uploaded_images = uploaded_images

        st.markdown("</div>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            process_btn = st.button("🚀 开始病历结构化处理", type="primary", use_container_width=True)

        st.divider()
        if process_btn:
            if not emr_text or len(emr_text.strip()) < 10:
                st.warning("⚠️ 请输入/上传有效的病历文本或影像图片")
            elif not model:
                st.error("❌ 智能体未加载成功，请检查模型路径和CUDA环境")
            else:
                progress_bar = st.progress(0, text="🚀 开始处理...")
                try:
                    struct_result, medical_check_result = extract_emr_struct_agent(model, tokenizer, emr_text, progress_bar)
                    st.markdown("<h2 style='color: #0066cc; text-align: center;'>📊 病历结构化结果</h2>", unsafe_allow_html=True)

                    # 医疗校验结果展示
                    if medical_check_result:
                        if medical_check_result.get("abnormal_items"):
                            st.warning(f"⚠️ 识别到异常信息：{'、'.join(medical_check_result['abnormal_items'])}")
                        if medical_check_result.get("missing_docs"):
                            st.info(f"📋 病历缺失文书：{'、'.join(medical_check_result['missing_docs'])}")
                        if medical_check_result.get("quality_check"):
                            st.success(f"✅ 病历质控结果：{medical_check_result['quality_check']}")

                    # 结果标签页
                    result_tab1, result_tab2, result_tab3, result_tab4, result_tab5 = st.tabs(
                        ["📋 分文书卡片展示", "📈 全量表格展示", "📝 JSON原始数据", "📷 原始病历影像", "📦 归档与脱敏"]
                    )

                    with result_tab1:
                        for doc_type, content in struct_result.items():
                            st.markdown(f"<div class='section-title'>📂 {doc_type}</div>", unsafe_allow_html=True)
                            if isinstance(content, dict):
                                cols = st.columns(2)
                                for idx, (field, value) in enumerate(content.items()):
                                    with cols[idx % 2]:
                                        st.markdown(f"""
                                        <div class='field-card'>
                                            <p class='field-key'>{field}</p>
                                            <p class='field-value'>{value}</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class='field-card'>
                                    <p class='field-value'>{content}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            st.divider()

                    with result_tab2:
                        table_data = []
                        for doc_type, content in struct_result.items():
                            if isinstance(content, dict):
                                for field, value in content.items():
                                    table_data.append({
                                        "文书类型": doc_type,
                                        "临床字段": field,
                                        "抽取内容": value
                                    })
                            else:
                                table_data.append({
                                    "文书类型": doc_type,
                                    "临床字段": "内容",
                                    "抽取内容": content
                                })
                        df = pd.DataFrame(table_data)
                        st.dataframe(df, use_container_width=True, hide_index=True, height=800)

                    with result_tab3:
                        st.json(struct_result)

                    with result_tab4:
                        if st.session_state.uploaded_images:
                            st.markdown("<div class='section-title'>本次上传的原始病历影像</div>", unsafe_allow_html=True)
                            img_cols = st.columns(3)
                            for idx, img_info in enumerate(st.session_state.uploaded_images):
                                with img_cols[idx % 3]:
                                    st.image(
                                        img_info["image"],
                                        caption=img_info["name"],
                                        use_column_width=True,
                                        output_format="PNG"
                                    )
                        else:
                            st.info("本次未上传病历影像图片")

                    with result_tab5:
                        st.markdown("<div class='section-title'>病历归档与隐私脱敏</div>", unsafe_allow_html=True)
                        
                        # 普通归档
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("📦 普通归档（不脱敏）", use_container_width=True):
                                archive_path = archive_case_images(
                                    st.session_state.uploaded_images,
                                    st.session_state.case_info,
                                    struct_result,
                                    desensitized=False
                                )
                                if archive_path:
                                    st.success(f"✅ 病历已归档至：{archive_path}")
                                else:
                                    st.error("❌ 归档失败，请查看调试日志")
                        
                        # 脱敏归档
                        with col2:
                            if st.button("🔒 脱敏归档（隐私保护）", use_container_width=True):
                                des_struct, des_text = desensitize_emr_data(struct_result, emr_text)
                                archive_path = archive_case_images(
                                    st.session_state.uploaded_images,
                                    st.session_state.case_info,
                                    des_struct,
                                    desensitized=True
                                )
                                if archive_path:
                                    st.success(f"✅ 脱敏病历已归档至：{archive_path}")
                                    # 展示脱敏效果
                                    with st.expander("查看脱敏效果对比"):
                                        col_a, col_b = st.columns(2)
                                        with col_a:
                                            st.markdown("### 原始数据（示例）")
                                            st.json(struct_result, expanded=False)
                                        with col_b:
                                            st.markdown("### 脱敏后数据（示例）")
                                            st.json(des_struct, expanded=False)
                                else:
                                    st.error("❌ 脱敏归档失败，请查看调试日志")
                        
                        # 导出JSON
                        st.divider()
                        json_str = json.dumps(struct_result, ensure_ascii=False, indent=4)
                        des_json_str = json.dumps(desensitize_emr_data(struct_result, emr_text)[0], ensure_ascii=False, indent=4)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button(
                                label="📥 导出原始JSON",
                                data=json_str,
                                file_name=f"{st.session_state.case_info['case_no']}_原始结构化结果.json",
                                mime="application/json",
                                use_container_width=True
                            )
                        with col2:
                            st.download_button(
                                label="🔒 导出脱敏JSON",
                                data=des_json_str,
                                file_name=f"{st.session_state.case_info['case_no']}_脱敏结构化结果.json",
                                mime="application/json",
                                use_container_width=True
                            )
                except Exception as e:
                    st.error(f"❌ 处理失败：{str(e)}")
                    st.session_state.debug_log.append(f"推理错误：{str(e)}")
        # 调试模式下展示原始文本
        if st.session_state.dev_mode and emr_text:
            st.divider()
            st.markdown("<div class='section-title'>🔧 原始输入文本（开发者模式）</div>", unsafe_allow_html=True)
            st.text_area("原始EMR文本", value=emr_text, height=300, label_visibility="collapsed")

    # -------------------- Tab2：LoRA模型训练 --------------------
    with main_tab2:
        st.markdown("<div class='card-container'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>🎯 LoRA微调训练（兼容TXT文件夹）</div>", unsafe_allow_html=True)

        st.markdown("### ① 从 TXT 文件夹加载病历（自动训练）")
        st.info(f"当前 TXT 病历文件夹：\n{TXT_CASE_FOLDER}")

        colA, colB = st.columns(2)
        with colA:
            if st.button("📂 加载所有 TXT 病历 → 生成训练数据", use_container_width=True):
                with st.spinner("正在读取 TXT 并生成训练集..."):
                    train_data = load_txt_to_train_data(TXT_CASE_FOLDER)
                    st.session_state.train_data = train_data
                    st.success(f"✅ 成功加载 {len(train_data)} 条 TXT 病历！")

        with colB:
            st.markdown("")
            st.markdown(f"已加载训练样本数：**{len(st.session_state.train_data)}**")
    
        st.markdown("---")
        st.markdown("### ② 训练参数配置")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            epochs = st.number_input("训练轮数", 1, 10, 3)
        with col2:
            batch_size = st.number_input("批次大小", 1, 8, 1)
        with col3:
            lr = st.number_input("学习率", 1e-5, 1e-3, 2e-4, format="%.5f")
        with col4:
            lora_rank = st.number_input("LoRA秩", 4, 64, 8)

        st.markdown("---")
        st.markdown("### ③ 开始训练")
        if st.button("🚀 启动 LoRA 训练（从 TXT 文件夹）", type="primary", use_container_width=True):
            if not model:
                st.error("❌ 模型未加载")
            elif len(st.session_state.train_data) == 0:
                st.error("❌ 请先加载 TXT 病历")
            else:
                with st.spinner("训练中..."):
                    try:
                        save_path = train_lora_simple(
                            model, tokenizer,
                            st.session_state.train_data,
                            epochs=epochs,
                            batch_size=batch_size,
                            lr=lr,
                            lora_rank=lora_rank
                        )
                        st.success(f"✅ 训练完成！模型保存到：{save_path}")
                    except Exception as e:
                        st.error(f"训练失败：{str(e)}")

        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------- Tab3：批量病历处理 --------------------
    with main_tab3:
        st.markdown("<div class='card-container'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>📦 批量病历处理</div>", unsafe_allow_html=True)
        
        # 批量文件上传
        st.markdown("### 1. 上传批量文件")
        batch_files = st.file_uploader(
            "支持TXT文本/图片格式（JPG/PNG），可多选上传",
            type=["txt", "jpg", "jpeg", "png"],
            accept_multiple_files=True
        )
        
        if batch_files:
            st.success(f"✅ 已上传{len(batch_files)}个文件，可开始批量处理")
            # 列出上传的文件
            with st.expander("已上传文件列表"):
                for idx, file in enumerate(batch_files):
                    st.markdown(f"{idx+1}. {file.name}（{file.type}）")
        
        # 开始批量处理
        st.markdown("### 2. 开始批量处理")
        if st.button("🚀 启动批量处理", type="primary", use_container_width=True):
            if not model:
                st.error("❌ 模型未加载成功，无法处理")
            elif not batch_files:
                st.error("❌ 未上传任何文件，请先选择文件")
            else:
                try:
                    total_progress = st.progress(0, text="📦 开始批量处理...")
                    batch_results = batch_process_emr(model, tokenizer, batch_files, total_progress)
                    
                    # 展示处理结果
                    st.markdown("### 3. 处理结果")
                    success_count = sum(1 for res in batch_results if "error" not in res)
                    st.success(f"✅ 批量处理完成：成功{success_count}/{len(batch_files)}，失败{len(batch_files)-success_count}")
                    
                    # 打包并提供下载
                    zip_buffer = zip_batch_results(batch_results)
                    st.download_button(
                        label="📥 下载批量处理结果（ZIP）",
                        data=zip_buffer,
                        file_name=f"批量病历处理结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
                    
                    # 展示失败文件列表
                    failed_files = [res for res in batch_results if "error" in res]
                    if failed_files:
                        st.markdown("### 4. 失败文件列表")
                        failed_data = [{"文件名": res["file_name"], "错误信息": res["error"]} for res in failed_files]
                        st.dataframe(pd.DataFrame(failed_data), use_container_width=True, hide_index=True)
                        
                except Exception as e:
                    st.error(f"❌ 批量处理失败：{str(e)}")
                    if st.session_state.dev_mode:
                        st.exception(e)
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
