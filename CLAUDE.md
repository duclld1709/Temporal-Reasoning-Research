# Temporal Reasoning — Hướng dẫn cho Claude

## Tổng quan dự án

Dự án nghiên cứu nhằm **cải thiện khả năng temporal reasoning của các LLM < 10B tham số** trên hai ngôn ngữ (English, Vietnamese) và hai loại task (Date Arithmetic, Duration Reasoning).

- **Ngôn ngữ giao tiếp mặc định**: tiếng Việt (giải thích, comment PR, docstring kỹ thuật có thể tiếng Anh).
- **Giai đoạn hiện tại**: Phase 1 — đánh giá baseline các phương pháp hiện có trên 1500 samples đầu mỗi dataset.
- **Model thực nghiệm**: `Qwen/Qwen3-8B` (HuggingFace).
- **Môi trường chạy**: Google Colab Pro (GPU A100/L4). Mọi thiết kế code phải chạy được trên Colab (RAM/VRAM hữu hạn).
- **Workflow code ↔ Colab**:
  - Toàn bộ **source code** được đẩy lên GitHub; notebook Colab dùng `git clone` / `git pull` để lấy code — **không** lưu code trong Drive.
  - **Google Drive chỉ mount để lưu kết quả**: file `predictions.jsonl` của mỗi run + bảng metric tổng quan (ví dụ `outputs/summary.csv`) sau khi test xong. Mục đích là giữ kết quả sau khi session Colab hết hạn, không phải đồng bộ code.
  - Output path trên Drive cần config được (qua YAML / biến trong notebook), không hardcode.

## Phạm vi Phase 1

### Datasets & kích thước eval Phase 1

| Ngôn ngữ | Task | File | Số samples Phase 1 |
|---|---|---|---|
| English | Duration Reasoning | [Dataset/Raw/English/UDST-DurationQA/data/test.tsv](Dataset/Raw/English/UDST-DurationQA/data/test.tsv) | 1500 rows đầu |
| English | Date Arithmetic (Temporal Reasoning) | [Dataset/Raw/English/BigBench_DateUnderstanding/task.json](Dataset/Raw/English/BigBench_DateUnderstanding/task.json) | **Toàn bộ 369 examples** (file chỉ có 369) |
| Vietnamese | Date Arithmetic | [Dataset/Raw/Vietnamese/VLSP 2025 ViTempQA (DateArith + DurationQA) Task/TrainingDataset/date_train_dataset/date_training_dataset.txt](Dataset/Raw/Vietnamese/VLSP%202025%20ViTempQA%20%28DateArith%20%2B%20DurationQA%29%20Task/TrainingDataset/date_train_dataset/date_training_dataset.txt) | 1500 dòng đầu |
| Vietnamese | Duration Reasoning | [Dataset/Raw/Vietnamese/VLSP 2025 ViTempQA (DateArith + DurationQA) Task/TrainingDataset/durationQA_train_dataset/duration_training_dataset.txt](Dataset/Raw/Vietnamese/VLSP%202025%20ViTempQA%20%28DateArith%20%2B%20DurationQA%29%20Task/TrainingDataset/durationQA_train_dataset/duration_training_dataset.txt) | **1500 rows đã expand** (≈ 375 questions gốc) |

### Đặc điểm từng dataset (đã khảo sát thực tế)

- **UDST-DurationQA (EN)** — TSV không header, **4 cột**: `context \t question \t candidate_answer \t label(yes/no)` (không có cột `id`). Mỗi sample là 1 cặp (ứng viên, yes/no) → binary classification → **F1**. File `test.tsv` có 4868 dòng; Phase 1 dùng 1500 dòng đầu.
- **BigBench DateUnderstanding (EN)** — JSON với `examples[i].input` + `target_scores` (dict đáp án→0/1). **Phải chuyển** thành 1 đáp án đúng duy nhất (đáp án có score=1) để prompt **open-ended**, LLM tự sinh date `MM/DD/YYYY` rồi so khớp chuỗi → **Accuracy**. File chỉ có **369 examples**, Phase 1 dùng toàn bộ.
- **VLSP ViTempQA DateArith (VI)** — JSONL mỗi dòng: `{"question", "answer": [str], "context"}`. Answer dạng "Tháng M, YYYY" → **Accuracy** (so chuỗi đã chuẩn hoá). File gốc 3000 dòng; Phase 1 dùng 1500 dòng đầu.
- **VLSP ViTempQA DurationQA (VI)** — JSONL mỗi dòng: `{"context","question","options":[4],"labels":["yes"/"no" x4],"qid"}`. Mỗi sample gốc expand thành 4 row binary (option, yes/no) → **F1**. File gốc 1489 questions (≈ 5956 rows sau expand); Phase 1 **dùng 1500 rows đã expand đầu tiên** (≈ 375 questions gốc đầu theo thứ tự file).

### Metric & Output
- **F1 Score**: Duration Reasoning (EN + VI). Binary, positive class = `yes`.
- **Accuracy**: Date Arithmetic / Temporal Reasoning (EN + VI). Open-ended string match sau normalize.
- **Avg inference time per sample** (giây): log riêng cho mỗi dataset.
- **Bắt buộc lưu raw output** của mô hình cho từng sample vào `outputs/<method>/<dataset>/predictions.jsonl` để phục vụ error analysis.

### Phương pháp

**Phase 1 (triển khai ngay):**
- Zero-shot
- Few-shot (k cố định, chọn thủ công từ train)

**Về sau (code phải để chỗ mở rộng, không hardcode):**
- Dynamic Few-shot (retrieval từ train set)
- Chain-of-Thought
- Majority Voting (ensemble nhiều seeds / nhiều models)
- Hybrid (tổ hợp các phương pháp trên)

→ Tổ chức mỗi phương pháp như một module độc lập trong `src/methods/` implement cùng interface (ví dụ `predict(sample) -> str`), để thêm method mới = thêm file + đăng ký.

## Cấu trúc thư mục (layout chuẩn)

```
Temporal_Reasoning/
├── Dataset/
│   ├── Raw/              # giữ nguyên, read-only
│   └── Preprocessed/     # dataset đã normalize (first 1500, đã chuyển BigBench → single-answer)
├── src/
│   ├── data/             # loader + preprocess cho từng dataset
│   ├── models/           # wrapper Qwen3.5-9B, generation config
│   ├── prompts/          # template system/user prompt theo task × ngôn ngữ
│   ├── methods/          # zero_shot.py, few_shot.py, (về sau) dynamic_few_shot.py, cot.py, voting.py, hybrid.py
│   ├── evaluation/       # metrics.py (f1, accuracy), extractor.py (clean output)
│   └── utils/            # io, timing, logging
├── notebooks/            # entry point chạy trên Colab (1 notebook / experiment)
├── configs/              # YAML per experiment (method, dataset, hyperparams, seed)
└── outputs/              # predictions + metric reports (gitignored)
```

`Dataset/`, `outputs/`, `*.txt/*.tsv/*.csv` đã được liệt kê trong `.gitignore`.

## Quy tắc kỹ thuật quan trọng (đã được xác nhận)

### 1. Output format của LLM — xử lý format noise
Qwen3-8B có thể sinh `<think>...</think>`, `\n\n`, prefix `yes\n` / `no\n`, v.v. So chuỗi thô với ground truth sẽ đánh giá sai năng lực model.

**Bắt buộc:**
- Thiết kế `system prompt` ràng buộc format chặt (ví dụ "Respond with only 'yes' or 'no'. No reasoning, no extra tokens.").
- Truyền hyperparameter hợp lý: `temperature=0` hoặc thấp cho evaluation, `do_sample=False`, set `max_new_tokens` đủ nhỏ cho các task format cố định.
- Test cả chế độ **thinking** và **non-thinking** của Qwen3-8B (tham số `enable_thinking` trong `apply_chat_template`); chế độ thinking phải strip `<think>...</think>` trước khi so sánh.
- Implement `src/evaluation/extractor.py` với các hàm regex / rule-based để trích:
  - `yes`/`no` cho Duration
  - date `MM/DD/YYYY` cho BigBench
  - "Tháng M, YYYY" cho VLSP DateArith
- Mọi metric phải chạy trên output **đã extract**, không trên raw string.

### 2. Colab — môi trường
Qwen3-8B (arch `qwen3`) đã có trong bản `transformers` gần đây. Trong notebook, cell đầu tiên cài bản mới nhất để tránh lệch version:
```python
!pip install -q -U transformers accelerate scikit-learn
# Nếu cần bản dev: !pip install -q -U git+https://github.com/huggingface/transformers.git
# ⚠️ Sau khi cài, Runtime → Restart session trước khi chạy cell kế tiếp
```

### 3. Giới hạn số samples Phase 1
Mọi loader trong `src/data/` phải hỗ trợ tham số `max_samples`. Giá trị mặc định theo dataset:
- UDST-DurationQA (EN): `max_samples=1500` (rows).
- BigBench DateUnderstanding (EN): `max_samples=None` → dùng toàn bộ 369 examples.
- VLSP DateArith (VI): `max_samples=1500` (dòng JSONL).
- VLSP DurationQA (VI): `max_samples=1500` áp dụng **sau khi expand 4 options thành 4 rows** (≈ 375 questions gốc đầu).

Không hardcode các giá trị này trong logic downstream — truyền qua config.

### 4. Reproducibility
- Fix seed (Python, NumPy, torch, transformers) trong mọi experiment.
- Mỗi run output: `predictions.jsonl` (raw + extracted + gold + correct) + `metrics.json` (f1/acc, avg_time, config snapshot).

## Khi nào hỏi người dùng

- Khi phát hiện format dataset khác mô tả ở trên → hỏi trước khi tự suy đoán.
- Khi cần chọn giữa nhiều prompt template / nhiều cách normalize answer (ví dụ "Tháng 4, 1321" vs "tháng 4/1321") → hỏi để chốt convention rồi mới viết extractor.
- Khi thêm phương pháp mới ngoài danh sách trên.

## Không cần xin phép

- Đọc file trong `Dataset/Raw/` để kiểm tra format.
- Tạo / sửa code trong `src/`, `notebooks/`, `configs/`.
- Chạy Python cục bộ cho unit test loader/extractor (không cần GPU).
