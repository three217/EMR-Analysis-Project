# 住院病历智能分析系统
### 基于轻量大模型智能体的电子病历自动结构化研究（SCI 三区复现项目）

论文题目
Development and Validation of a Lightweight Large Language Model-Based Agent for Automatic Electronic Medical Record Structurization: A Single-Center Real-World Study

环境要求
- Python 3.11.9
- CUDA 11.8
- GPU显存 ≥ 10GB

快速开始
1. 创建虚拟环境
python -m venv emr_env
Windows: emr_env\Scripts\activate
Linux/macOS: source emr_env/bin/activate

2. 安装依赖
pip install -r requirements.txt

3. 启动
python run.py

核心功能
- 病历文书自动拆分
- 医学信息结构化抽取
- 病历图片OCR
- 医疗质控与异常检测
- 隐私脱敏
- LoRA微调
- 批量处理与导出

使用步骤
1. 填写本地大模型路径
2. 粘贴文本/上传TXT/上传图片
3. 自动结构化抽取
4. 导出结果与归档

声明
1. 本项目仅限学术研究，禁止商业使用。
2. 使用医疗数据需通过伦理审查。
3. 遵守医疗隐私保护法规。
