# Temporal Reasoning for Small LLMs (< 10B)

Nghiên cứu cải thiện khả năng **temporal reasoning** của LLM nhỏ hơn 10B tham số trên hai ngôn ngữ (English, Vietnamese) và hai loại task (Date Arithmetic, Duration Reasoning). Phase 1 tập trung **benchmark baseline** của `Qwen/Qwen3-8B` với prompting techniques (zero-shot, few-shot), làm nền cho các phương pháp nâng cao ở phase sau.

---

## Mục lục

- [Tổng quan](#tổng-quan)
- [Datasets](#datasets)
- [Metric & Output](#metric--output)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Cài đặt cục bộ](#cài-đặt-cục-bộ)
- [Chạy trên Google Colab](#chạy-trên-google-colab)
- [Chạy thủ công (CLI)](#chạy-thủ-công-cli)
- [Verbose & Debugging](#verbose--debugging)
- [Kiến trúc code](#kiến-trúc-code)
- [Mở rộng](#mở-rộng)
- [Reproducibility](#reproducibility)
- [Testing](#testing)

---

## Tổng quan

| Hạng mục | Giá trị |
|---|---|
| **Model** | `Qwen/Qwen3-8B` (HuggingFace, hỗ trợ thinking / non-thinking mode) |
| **Ngôn ngữ** | English, Vietnamese |
| **Task** | Date Arithmetic (open-ended), Duration Reasoning (binary yes/no) |
| **Phương pháp Phase 1** | Zero-shot, Few-shot (k cố định) |
| **Phương pháp tương lai** | Dynamic Few-shot, Chain-of-Thought, Majority Voting, Hybrid |
| **Môi trường** | Google Colab Pro (A100 / L4); code cũng chạy cục bộ không GPU cho preprocess + tests |
| **Pipeline** | Source code trên GitHub → Colab `git clone/pull` → chạy eval → dump kết quả sang Google Drive |

---

## Datasets

| Ngôn ngữ | Task | Source | Phase 1 size |
|---|---|---|---|
| English | Duration | [UDST-DurationQA](Dataset/Raw/English/UDST-DurationQA/data/test.tsv) | 1500 rows đầu của `test.tsv` |
| English | Date Arithmetic | [BigBench DateUnderstanding](Dataset/Raw/English/BigBench_DateUnderstanding/task.json) | **Toàn bộ 369 examples** |
| Vietnamese | Date Arithmetic | [VLSP 2025 ViTempQA — DateArith](Dataset/Raw/Vietnamese/VLSP%202025%20ViTempQA%20%28DateArith%20%2B%20DurationQA%29%20Task/TrainingDataset/date_train_dataset/date_training_dataset.txt) | 1500 dòng đầu JSONL |
| Vietnamese | Duration | [VLSP 2025 ViTempQA — DurationQA](Dataset/Raw/Vietnamese/VLSP%202025%20ViTempQA%20%28DateArith%20%2B%20DurationQA%29%20Task/TrainingDataset/durationQA_train_dataset/duration_training_dataset.txt) | **1500 rows sau khi expand** 4 options/question (≈ 375 questions gốc) |

### Chuẩn hoá (xem [src/data/](src/data/))

- **UDST** — TSV 4 cột (`context, question, candidate_answer, label`) không header → binary classification.
- **BigBench** — JSON `examples[i].target_scores` → giữ duy nhất đáp án có score=1, prompt open-ended sinh `MM/DD/YYYY`.
- **VLSP DateArith** — JSONL `{question, answer, context}` → gold `"Tháng M, YYYY"`.
- **VLSP DurationQA** — JSONL `{context, question, options[4], labels[4], qid}` → expand 4 row binary / question.

Preprocess dump JSONL chuẩn schema vào [Dataset/Preprocessed/](Dataset/Preprocessed/):

```bash
python -m src.data.preprocess
```

---

## Metric & Output

| Task | Metric | Ghi chú |
|---|---|---|
| Duration (EN + VI) | **F1** binary, positive class = `yes` | Sample không parse được coi như predict `no` |
| Date Arithmetic (EN + VI) | **Accuracy** string match sau normalize | |
| Chung | **Avg inference time / sample** (giây) | Log per-dataset |

Mỗi experiment sinh:

- `outputs/<method>/<dataset>/predictions.jsonl` — mỗi dòng: `sample_id, task, language, dataset, question, gold_raw, gold_normalized, raw_output, extracted, correct, elapsed_sec`.
- `outputs/<method>/<dataset>/metrics.json` — snapshot metric + config.
- `outputs/summary.csv` — bảng tổng quan (append mỗi lần chạy).

---

## Cấu trúc thư mục

```
Temporal_Reasoning/
├── Dataset/
│   ├── Raw/                  # read-only, gitignored
│   └── Preprocessed/         # JSONL chuẩn schema (sinh bởi preprocess.py)
├── src/
│   ├── data/                 # loaders + registry + preprocess
│   ├── models/               # Qwen wrapper + ChatLM protocol
│   ├── prompts/              # templates per (task × ngôn ngữ) + shot pools
│   ├── methods/              # zero_shot, few_shot, registry (mở rộng dễ)
│   ├── evaluation/           # extractor, metrics, evaluate
│   ├── utils/                # io, seed, timing
│   └── runner.py             # entry-point chạy 1 experiment
├── configs/                  # 8 YAML (4 datasets × {zero_shot, few_shot})
├── notebooks/
│   └── run_phase1_colab.ipynb
├── tests/                    # 22 unit/integration tests
├── requirements.txt
├── CLAUDE.md                 # hướng dẫn dành cho Claude Code
└── README.md
```

---

## Cài đặt cục bộ

Dùng Python 3.10+ (đã test với 3.13).

```bash
git clone <repo_url> Temporal_Reasoning
cd Temporal_Reasoning

# (Khuyên) venv
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Lưu ý: để **chạy inference** Qwen3-8B bạn cần GPU ≥ 16 GB VRAM (khuyến nghị A100/L4 trên Colab). Nếu chỉ muốn thử loader / extractor / preprocess / unit tests thì không cần GPU.

---

## Chạy trên Google Colab

Notebook chính: [notebooks/run_phase1_colab.ipynb](notebooks/run_phase1_colab.ipynb).

Bố cục notebook:

1. **5 SETUP cells** (chạy 1 lần):
   1. Cài `transformers`, `accelerate`, `scikit-learn`, `pyyaml`.
   2. Mount Drive + `git clone/pull` repo.
   3. Config path + symlink `Dataset/` từ Drive.
   4. Chạy preprocess → dump JSONL chuẩn schema.
   5. **Load Qwen3-8B một lần** và định nghĩa helper `run_exp(cfg_path, ...)`.
2. **8 EXPERIMENT cells** (mỗi cell 1 experiment, rerun độc lập):

   | Cell | Method | Dataset | Metric |
   |---|---|---|---|
   | EXP 1/8 | zero_shot | udst_duration (EN) | F1 |
   | EXP 2/8 | zero_shot | bigbench_date (EN) | Accuracy |
   | EXP 3/8 | zero_shot | vlsp_date (VI) | Accuracy |
   | EXP 4/8 | zero_shot | vlsp_duration (VI) | F1 |
   | EXP 5/8 | few_shot k=4 | udst_duration (EN) | F1 |
   | EXP 6/8 | few_shot k=3 | bigbench_date (EN) | Accuracy |
   | EXP 7/8 | few_shot k=3 | vlsp_date (VI) | Accuracy |
   | EXP 8/8 | few_shot k=4 | vlsp_duration (VI) | F1 |
3. **3 DEBUG cells**:
   - **A** — audit parse-fail + sample sai đầu tiên của 1 run.
   - **B** — probe 1 sample bất kỳ: in đầy đủ messages + raw output + extracted + gold (dùng để chỉnh prompt).
   - **C** — hiển thị `summary.csv`.

Trước khi chạy: đổi `REPO_URL` ở SETUP cell 2 thành URL GitHub của bạn và upload `Dataset/Raw/` lên Drive ở path trùng với `DATASET_ROOT`.

---

## Chạy thủ công (CLI)

Mỗi experiment là 1 YAML config. Chạy:

```bash
# Chạy 1 experiment
python -m src.runner --config configs/zero_shot_udst_duration.yaml

# Bật verbose (in raw output + extracted + gold cho sample đầu)
python -m src.runner --config configs/few_shot_vlsp_date.yaml --verbose --verbose-first-n 10 --verbose-every 200
```

Output lưu vào `outputs/<method>/<dataset>/` theo đúng cấu trúc ở [Metric & Output](#metric--output).

---

## Verbose & Debugging

`RunConfig` expose các trường kiểm soát log per-sample:

| Field | Mặc định | Ý nghĩa |
|---|---|---|
| `verbose` | `False` | Bật log per-sample |
| `verbose_first_n` | 5 | In **full** (question + raw + extracted + gold + ✓/✗ + time) cho N sample đầu |
| `verbose_every` | 0 | `>0` → in log rút gọn mỗi N sample |
| `running_score_every` | 100 | In **running F1 / accuracy** mỗi N sample để phát hiện sớm prompt kém hoặc extractor sai |

Trong Colab notebook, helper `run_exp(cfg_path, verbose=True, verbose_every=200, ...)` override 4 trường này mà không cần sửa YAML.

Ví dụ log khi verbose:

```
[runner] experiment=zero_shot_vlsp_date method=zero_shot dataset=vlsp_date verbose=True
[runner] loaded 1500 samples (task=date_arith, lang=vi)
  [1/1500] ✓ gold='Tháng 4, 1321' extracted='Tháng 4, 1321' elapsed=0.42s
      Q: Giả sử bạn đang ở tháng 4, 1316, thời gian sau 4 năm 12 tháng, thì là thời điểm nào?
      raw: Tháng 4, 1321
  ...
  [100/1500] running: acc=0.720 correct=72/100 parse_fail=3 avg_time=0.48s
```

---

## Kiến trúc code

### Flow 1 experiment

```
YAML config ──► RunConfig ──► load_dataset(name) ──► [Sample, ...]
                                 │
                                 ▼
                      build_method(model, k_shot, shots)
                                 │
                                 ▼ for each sample
                  method.predict(sample) ──► raw_output
                                 │
                                 ▼
                  build_record(sample, raw, time)   ← extractor + normalize_gold
                                 │
                                 ▼
                  score_records(records, task, lang)
                                 │
                                 ▼
                  predictions.jsonl + metrics.json + summary.csv
```

### Các interface chính

- [src/data/schema.py](src/data/schema.py) — `Sample` TypedDict (`sample_id, task, language, dataset, context, question, gold, meta`).
- [src/models/base.py](src/models/base.py) — `ChatLM` protocol (`generate(messages, ...)`), `ChatMessage`.
- [src/methods/base.py](src/methods/base.py) — `Method` protocol (`predict(sample) -> str`).
- [src/prompts/templates.py](src/prompts/templates.py) — `PromptTemplate` (`system, render_user, render_shot_user, render_shot_assistant`) + `build_messages(sample, shots)`.
- [src/evaluation/extractor.py](src/evaluation/extractor.py) — `extract(task, lang, raw)`, `normalize_gold(task, lang, gold)`; strip `<think>...</think>` cho thinking mode.
- [src/evaluation/metrics.py](src/evaluation/metrics.py) — `binary_f1_yes`, `accuracy`, `avg_inference_time`.

### Registry

- **Datasets** → [src/data/registry.py](src/data/registry.py) (`DATASET_LOADERS`, `DEFAULT_PATHS`, `DEFAULT_MAX_SAMPLES`).
- **Methods** → [src/methods/registry.py](src/methods/registry.py) (`METHOD_BUILDERS`).
- **Prompt templates** → [src/prompts/templates.py](src/prompts/templates.py) (`TEMPLATES` theo key `(task, language)`).

---

## Mở rộng

### Thêm phương pháp mới (ví dụ Chain-of-Thought)

1. Tạo [src/methods/cot.py](src/methods/cot.py) với class `CoTMethod` implement `predict(sample) -> str`.
2. Đăng ký trong [src/methods/registry.py](src/methods/registry.py):

   ```python
   def build_cot(model, **kwargs): return CoTMethod(model, **kwargs)
   METHOD_BUILDERS["cot"] = build_cot
   ```

3. Tạo YAML config `configs/cot_<dataset>.yaml` với `method: cot`.
4. Chạy: `python -m src.runner --config configs/cot_<dataset>.yaml`.

### Thêm dataset mới

1. Viết loader trong `src/data/<new_dataset>.py` trả về `list[Sample]` đúng schema.
2. Đăng ký trong [src/data/registry.py](src/data/registry.py) (`DATASET_LOADERS`, `DEFAULT_PATHS`, `DEFAULT_MAX_SAMPLES`).
3. Nếu task/ngôn ngữ chưa có: bổ sung `PromptTemplate` trong [src/prompts/templates.py](src/prompts/templates.py) và `extract/normalize_gold` trong [src/evaluation/extractor.py](src/evaluation/extractor.py).

### Dynamic Few-shot

Class `FewShotMethod` đã nhận `shot_selector: Callable[[Sample], Sequence[Sample]]`. Chỉ cần implement selector dùng retrieval (TF-IDF / embedding) thay cho `fixed_shots(...)`.

---

## Reproducibility

- Seed Python + NumPy + PyTorch + transformers được fix trong mỗi run qua [src/utils/seed.py](src/utils/seed.py) (default `seed=42`).
- Eval hyperparams: `temperature=0`, `do_sample=False`, `max_new_tokens` nhỏ theo task (duration: 8, date_arith: 24). Xem [src/methods/base.py](src/methods/base.py).
- `metrics.json` snapshot toàn bộ config cho mỗi run.

---

## Testing

22 test cho loader / extractor / metrics / prompts / evaluate. Chạy:

```bash
python -m pytest tests/ -q
```

Test integration cho 4 loader sẽ **skip tự động** nếu `Dataset/Raw/` không tồn tại — an toàn cho CI không có dataset.

---

## Giấy phép & nguồn dataset

- **UDST-DurationQA** — [Original repo](https://github.com/UBC-NLP/UDST) (license theo tác giả).
- **BigBench DateUnderstanding** — [BIG-bench](https://github.com/google/BIG-bench) (Apache-2.0).
- **VLSP 2025 ViTempQA** — VLSP 2025 shared task (sử dụng theo điều khoản ban tổ chức).

Vui lòng trích nguồn khi công bố kết quả.
