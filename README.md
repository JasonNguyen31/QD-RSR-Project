# QD-RSR: Quality and Diversity Augmented Rank-Surprisal Selection

# Project Structure

```text
Duancntt/
├── configs/
│   ├── base.yaml                  # tham số dùng chung mọi phương án
│   ├── data.yaml                  # thành phần dữ liệu, tỷ lệ độ khó, seed
│   ├── models.yaml                # 3 teacher + judge + embedding (tên, mã tham số sinh)
│   │
│   ├── method/                    # mỗi phương án chỉ ghi phần khác biệt
│   │   ├── qd_rsr.yaml
│   │   ├── rsr.yaml
│   │   ├── lark.yaml
│   │   ├── grape.yaml
│   │   ├── local_nat.yaml
│   │   ├── token_length.yaml
│   │   ├── correct_only.yaml
│   │   └── no_filter.yaml
│   │
│   ├── ablation/                  # sáu biến thể loại bỏ thành phần
│   │   ├── fit_only.yaml          # b=0, λ=0  → tương đương RSR
│   │   ├── quality_only.yaml      # a=0, λ=0
│   │   ├── diversity_only.yaml    # a=0, b=0
│   │   ├── fit_quality.yaml       # λ=0
│   │   ├── fit_diversity.yaml     # b=0
│   │   └── no_prefilter.yaml      # bỏ bước lọc sơ bộ
│   │
│   └── student/
│       ├── qwen1_5b.yaml
│       └── qwen7b.yaml
│
├── src/
│   ├── common/
│   │   ├── __init__.py
│   │   ├── config.py              # đọc YAML, gộp base + method + student
│   │   ├── api.py                 # gọi API, tenacity, ghi từng phần, đếm token
│   │   ├── prompts.py             # system prompt \boxed{}, prompt giám khảo JSON
│   │   ├── answers.py             # tách + chuẩn hoá + so khớp (quy ước PRM800K)
│   │   └── io_utils.py            # đọc ghi jsonl, checkpoint
│   │
│   ├── stage_a/                   # chạy MỘT LẦN, không phụ thuộc mô hình học
│   │   ├── a1_prepare.py          # tải dữ liệu, loại trùng MATH-500, tách tập
│   │   ├── a2_generate.py         # sinh chuỗi từ 3 teacher
│   │   ├── a3_filter.py           # gán nhãn đúng/sai, lọc sơ bộ, thống kê
│   │   ├── a4_score_quality.py          # rule_score + judge → Qual(t)
│   │   └── a5_embed.py            # BGE-M3 → embed(t)
│   │
│   ├── stage_b/                   # LẶP theo từng mô hình học × từng phương án
│   │   ├── b1_fit.py              # RSR/Fit + GRAPE + LocalNat + LARK (cùng một lượt logits)
│   │   ├── b2_select.py           # greedy QD-RSR + mọi baseline
│   │   └── b3_train.py            # fine-tune QLoRA
│   │
│   ├── stage_c/
│   │   ├── c1_evaluate.py         # 6 benchmark, Acc@4 + Acc@1
│   │   └── c2_deploy.py           # merge LoRA → GGUF → đo tốc độ, bộ nhớ
│   │
│   └── tools/                     # tiện ích, không thuộc pipeline chính
│       ├── test_api.py            # kiểm tra 4 mô hình còn gọi được
│       ├── verify_dedup.py        # kiểm chứng hàm lọc MATH-500
│       ├── inspect_rsr_ref.py     # khảo sát dữ liệu công khai của bài gốc
│       └── make_figures.py        # vẽ Hình 2 và Hình 3 cho paper
│
├── data/
│   ├── raw/                       # pool câu hỏi sau khi lọc trùng
│   ├── pilot/                     # lô thử nghiệm nhỏ
│   ├── stage_a/                   # 18.000 chuỗi + Qual + embed (dùng chung)
│   ├── stage_b/                   # điểm Fit và tập đã chọn, chia theo mô hình học
│   └── rsr_ref/                   # dữ liệu tham chiếu tải từ HuggingFace
│
├── outputs/
│   ├── models/                    # LoRA adapter sau fine-tune
│   ├── results/                   # bảng kết quả từng benchmark
│   ├── logs/                      # nhật ký chạy, đường cong huấn luyện
│   └── figures/                   # hình cho paper
│
├── .env
├── .gitignore
├── pyrightconfig.json
└── README.md
```
