# 景区多模态智能讲解助手

文本提问 → 多模态知识库（文本 / 图片 / 视频）固定 RAG → 答案 + **程序化检索证据**（必要时附图 / 视频引用）。

求职作品集：**无 Agent 编排**；主链路云端 Embedding（DashScope）+ LLM；对照实验（BGE-M3 / GTE-Qwen2）仅作 H5 证据，不写入主索引。

## 产品演示

约 1 分钟无旁白短片：提问 → 多模态入库 → 检索回答 → 来源可追溯。

[![产品演示封面](assets/demo-poster.jpg)](https://iam-myung.github.io/scenic-multimodal-guide/demo.html)

**[在线观看（GitHub Pages）](https://iam-myung.github.io/scenic-multimodal-guide/demo.html)** · 下载 [`assets/demo.mp4`](assets/demo.mp4)（约 8MB / 1 分钟）

> 首次使用需在 GitHub 仓库 **Settings → Pages** 中：Source 选 `Deploy from a branch`，Branch 选 `main` / `/docs`，保存后等待 1～2 分钟再打开上方链接。
>
> 片源工程在 `demo-video/`（HyperFrames，默认不入库）；需要重渲时进入该目录执行 `npm run render`。

## 能力一览

| 能力 | 说明 |
| :--- | :--- |
| 固定 RAG | 白名单语料入库 → FAISS → Top-k 文本上下文 + 可选图/视频引用 |
| 程序化证据 | `[检索证据]` 由检索结果组装，不依赖 LLM 自报来源 |
| CLI 主入口 | `python -m src.cli`（交互 / 单次 ask / build / experiment / serve） |
| 本地 Web 演示 | FastAPI + `web/`，与 CLI 共用 `ask()`（非 P0） |
| 验收 | 日常 mock 单测；P0 = 真实 e2e T1–T4 + 对照 H5 |

## 仓库结构

```text
.
├── src/                 # 正式代码（ingest / embedding / retrieve / rag / experiment / api）
├── tests/               # unit · integration · e2e
├── data/                # 白名单语料、video_sources、experiment testset（索引 data/index/ 不入库）
├── web/                 # 本地演示静态页
├── assets/              # 产品演示片（demo.mp4）
├── .docs/               # 本地工程文档（gitignore，不随仓库分发）
├── requirements.txt     # 主链路依赖
├── requirements-web.txt # 可选 Web
├── requirements-experiment.txt  # 可选 H5 对照
└── .env.example         # 环境变量模板（复制为 .env，勿提交）
```

## 快速开始

**前提：** Python **3.10–3.12**（推荐 3.11）、DashScope API Key（建库 / 问答）。

```bash
git clone <your-repo-url>
cd 景区多模态智能讲解助手   # 或你克隆后的目录名

python -m venv .venv
# Windows: .\.venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env        # Windows 可用: copy .env.example .env
# 编辑 .env，填写 DASHSCOPE_API_KEY=

python -m src.cli build
python -m src.cli ask "一日票提前多久可以免费修改入园日期？需要什么条件？"
# 或交互模式：
python -m src.cli
```

输出含：`answer`、`[检索证据]`（程序化 `sources[]`）、可选 `[相关图片]` / `[相关视频]`。  
回滚索引：删除 `data/index/` 后重新 `build`。

> **Windows：** 中文路径写 FAISS 可能失败；真实 `build` / `RUN_E2E=1` 建议放到纯 ASCII 目录（例：`E:\scenic_p0_work`）。

### 可选：本地 Web 演示

与 CLI **共用** `src.rag.ask.ask()`，**不改变**主链路；**P0 验收仍只认** CLI + e2e + H5。

```bash
pip install -r requirements-web.txt
# 须已 build 索引；工作目录 = 项目根
python -m src.cli serve --open
# 或: uvicorn src.api.app:app --host 127.0.0.1 --port 8000
# 浏览器: http://127.0.0.1:8000/
```

| 端点 | 说明 |
| :--- | :--- |
| `GET /api/health` | 存活探测 |
| `GET /api/status` | 索引是否就绪 |
| `POST /api/build` | 一键建库（需 Embedding 可用） |
| `POST /api/ask` | 问答（同 CLI） |

静态页：`web/`；图片同源 `/kb/...` → `data/knowledge_base/`；Key 仅服务端 `.env`，默认只监听本机回环。

## 正式入口（CLI）

```bash
python -m src.cli                      # 默认：交互问答（问题>）
python -m src.cli ask "你的问题"       # 一次性提问
python -m src.cli build                # 建/重建索引
python -m src.cli build --mock-embedding   # 无 Key 冒烟建库（仅演示）
python -m src.cli experiment --model all   # 对照 H5（需对照依赖）
python -m src.cli serve --open         # 本地 Web
```

正式代码在 `src/`，正式测试在 `tests/`，演示数据仅 `data/` 白名单内文件。

## 环境变量

复制 `.env.example` → `.env`（由 `python-dotenv` 加载；**勿提交** `.env`）。

| 变量 | 必填 | 说明 |
| :--- | :--- | :--- |
| `DASHSCOPE_API_KEY` | 主链路是 | Embedding + 默认 LLM（qwen-flash） |
| `DASHSCOPE_COMPAT_BASE_URL` | 否 | OpenAI 兼容网关；默认可指向官方；按量 MaaS 工作空间可改 |
| `DASHSCOPE_HTTP_API_URL` | 否 | HTTP API 基址（多模态 Embedding 等） |
| `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` | 否 | 覆盖默认 LLM（例：DeepSeek）；一般可留空 |
| `MODELSCOPE_CACHE_DIR` | 否 | 对照模型缓存，默认 `./models` |
| `SCENIC_DEMO_VIDEO_URL` | 否 | 覆盖 `video_sources.json` 中的演示视频 URL |
| `RUN_E2E` | 否 | 设为 `1` 时跑真实 e2e |
| `SCENIC_EMBEDDING_MOCK` | 否 | 非空时走 mock Embedding（本地冒烟） |

### 对照实验依赖（可选；跑 H5 / 完整 P0 时需要）

```bash
# 先按本机 Python/OS 安装匹配的 torch：https://pytorch.org/get-started/locally/
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-experiment.txt
# 已钉 transformers==4.46.3（GTE 兼容；勿随意升到 4.57.x）
```

本地模型缓存默认 `./models`（已 gitignore）。首次 `experiment` 会经 ModelScope **自动下载**权重（约数 GB～十余 GB）。

## 测试与验收

### 日常 CI（可跳过真实 API）

外部 Embedding / LLM **必须 mock**；不要求 Key：

```bash
pytest tests/unit tests/integration
```

推送到 GitHub 后，[`.github/workflows/ci.yml`](.github/workflows/ci.yml) 会在 `main`/`master` 的 push 与 PR 上自动跑同上测试（Python 3.11 + `requirements.txt` + `requirements-web.txt`；**不含** e2e / H5）。

### P0 完整验收（两项都不可跳过）

缺任一 = P0 **Fail**：

```bash
# ① 主链路真实 API：T1–T4 硬断言（SPEC §15.1）
RUN_E2E=1 pytest tests/e2e -m e2e

# ② 对照实验 H5（BGE + GTE 均须 Pass）
pip install -r requirements-experiment.txt   # 若尚未安装
python -m src.cli experiment --model all
```

- e2e 默认 **skip**，仅当 `RUN_E2E=1` 时调用真实 DashScope
- T1–T4 禁止弱断言（不得仅凭「有回答 / sources 非空」通过）
- 跨样例门禁：集合内至少一次图片引用成功 + 一次视频引用成功
- 验收细节以本地工程文档为准（不随本仓库分发）

### H5 对照说明（摘要）

H5 是 **对照证据**（BGE-M3 + GTE-Qwen2 在固定 `data/experiment/testset.json` 上分别 Pass），**不是**日常问答依赖。`experiment` **不写**主 FAISS。

| 用途 | ModelScope ID | 模型页 |
| :--- | :--- | :--- |
| 对照 BGE-M3 | `BAAI/bge-m3` | https://www.modelscope.cn/models/BAAI/bge-m3 |
| 对照 GTE-Qwen2-1.5B | `iic/gte_Qwen2-1.5B-instruct` | https://www.modelscope.cn/models/iic/gte_Qwen2-1.5B-instruct |

GTE 须钉 **`transformers==4.46.3`**；过新可能触发 `DynamicCache.get_usable_length` 报错。测完可删 `./models` 腾盘。

## 建库前自检

1. `data/corpus_whitelist.json`（path **相对 `data/`**）与磁盘文件一致  
2. `data/video_sources.json` **无** `car.mp4` / 汽车剐蹭类演示  
3. `python -m src.cli build` → `data/index/` 下 faiss + metadata + **必需 manifest**

## 作品集交付与合规

本仓库是**独立求职作品**：运行与讲述只认 `src.cli` + 白名单 `data/`。

- **License**：源代码为 [MIT](LICENSE)；语料 / 第三方模型与 API 各自遵循其原授权，不受本 License 覆盖  
- **交付**：`src/`、`tests/`、`web/`、`requirements*.txt`、白名单语料、本文档  
- **不随仓库分发**：`.docs/`（PRD / SPEC / 验收证据等，仅本地维护）、`.env`、生成索引与模型缓存  
- **不交付**：外部教学 CASE 目录（历史名如 `18_` / `19_`）——不是本作品组成部分  
- **合规**：未授权 PDF/PPT/图片/内部资料不得进入 `data/`；「完整扩展知识库」类路径已 gitignore  
- **gitignore**：`.env`、`.docs/`、`models/`、`data/index/`、`**/迪士尼RAG知识库（完整）/`（历史扩展语料红线，勿提交）  
- **授权**：白名单演示语料为虚构「某大型游乐园」内容（`owner_original`）；回答不代表任何真实乐园官方政策

人设与免责：回答仅基于演示知识库，**不代表**景区官方实时信息。

## 文档说明

| 文档 | 用途 | 是否随仓库 |
| :--- | :--- | :--- |
| 本文 `README.md` | 安装、运行、环境、本地坑 | 是 |
| `requirements*.txt` | 依赖与对照钉版 | 是 |
| 本地 `.docs/`（PRD / SPEC / EVIDENCE / prompt-step 等） | 产品意图、行为验收、开发步进 | **否**（已 gitignore） |
