# QD-RSR

Chọn lọc chuỗi suy luận cho chưng cất mô hình ngôn ngữ nhỏ, kết hợp ba tín hiệu:
mức phù hợp với mô hình học (RSR), chất lượng lập luận, và tính đa dạng.

```
F(S, m) = Σ_{t ∈ S} [ Fit(t,m)^a × Qual(t)^b ] + λ × Div(S)
```

## Chạy nhanh

```bash
qdrsr                                   # vào môi trường
python -m pytest tests -q               # 101 bài, offline, miễn phí
python -m src.tools.test_api            # bốn mô hình còn gọi được không
```

Sổ tay lệnh đầy đủ, kèm cách tinh chỉnh `--workers` và `--max-cost` cho từng tình huống,
nằm ở trang Notion "SỔ TAY LỆNH" dưới PHASE 2.

## Dòng chảy dữ liệu

```
a1_prepare  →  questions.jsonl            2.000 câu
a2_generate →  trajectories.<lô>.jsonl    18.000 chuỗi   (ba lô song song)
a3_filter   →  candidates.jsonl           chuỗi đúng, câu đủ k
a4_score    →  quality.jsonl              Qual(t)
a5_embed    →  embeddings.npz             embed(t)
b1_fit      →  fit.jsonl                  RSR, GRAPE, LocalNat, LARK (một lượt logits)
b2_select   →  selected.jsonl             6.000 mẫu mỗi phương án
b3_train    →  LoRA adapter
c1_evaluate →  results/                   Acc@4 trên sáu benchmark
c2_deploy   →  GGUF + chỉ số triển khai
```

Mọi bảng nối với nhau qua một khoá duy nhất: `tid = "<qid>|<teacher>|<sample_idx>"`.
Đổi định dạng này là hỏng toàn bộ liên kết.

## Cấu trúc

```
configs/          tham số, không có logic
  base.yaml         dùng chung mọi phương án, gồm signal_direction
  data.yaml         thành phần dữ liệu và tên file chuẩn
  models.yaml       ba mô hình dạy, giám khảo, embedding (tên biến môi trường, KHÔNG chứa khoá)
  method/           tám phương án so sánh
  ablation/         sáu biến thể loại bỏ thành phần
  student/          hai mô hình học

src/
  common/           config, io_utils, answers, api, prompts, rule_score
  stage_a/          a1..a5  — chạy MỘT LẦN, không phụ thuộc mô hình học   [máy Mac]
  stage_b/          b1..b3  — LẶP theo từng mô hình học × từng phương án   [máy Windows]
  stage_c/          c1..c2  — đánh giá và triển khai
  tools/            tiện ích ngoài pipeline

tests/            101 bài, offline — xem tests/README.md
data/             không nằm trong git — xem data/README.md
outputs/          mô hình, kết quả, nhật ký, hình
```

## Phân công hai máy

| Máy | Việc | Vì sao |
| --- | --- | --- |
| MacBook Pro M4 Pro 24GB | a1..a5: gọi API, điểm quy tắc, embedding, thuật toán greedy | Thuần mạng và tính nhẹ trên MPS, để GPU rảnh |
| Windows RTX 3060 12GB | b1..b3: điểm phụ thuộc mô hình học, tinh chỉnh QLoRA | Độ chính xác số học giữa MPS và CUDA lệch nhau có thể đổi thứ tự xếp hạng chuỗi; bitsandbytes không chạy trên Apple Silicon |

Nguyên tắc bộ nhớ: không để chiếm quá 10GB dù card có 12GB.

## Chốt cấu hình Giai đoạn A (20/09/2026)

| Hạng mục | Giá trị | Căn cứ |
| --- | --- | --- |
| Prompt sinh chuỗi | chỉ dẫn nối cuối đề bài, vai trò user | Phụ lục A.2 bài RSR; hoà 0,6 điểm với cách cũ |
| Nhiệt độ | 0,8 | Khoảng cách trung vị trong cùng mô hình dạy là 0,256, đủ xa 0 |
| max_tokens / max_seq_len | 3072 | Cho hẳn 4096 thì chuỗi đúng dài nhất vẫn chỉ 2800 |
| Embedding | toàn chuỗi, BGE-M3 | Độ trải p25–p75 là 0,157, không dồn cục |
| Giám khảo | gemini-2.5-flash-lite | Rẻ nhất, nhanh nhất, phân biệt tốt nhất (18 cặp hoà so với 99) |
| Mô hình dạy | DeepSeek-V3, Qwen2.5-VL-72B, Llama-3.3-70B | Ba họ kiến trúc khác nhau |

## Bảo mật

`.env` và `data/` nằm ngoài git. Khoá API đặt trong `.env` ở gốc repo, tên biến khai báo ở
`configs/models.yaml`. Không bao giờ ghi khoá vào mã, vào cấu hình, hay vào tài liệu.
