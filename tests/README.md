# Bộ kiểm thử

101 bài, chạy hoàn toàn offline: không gọi mạng, không cần GPU, không cần khoá API.
Mọi lượt gọi API được thay bằng client giả, nên chạy bao nhiêu lần cũng miễn phí.

```bash
python -m pytest tests -q          # toàn bộ
python -m pytest tests/stage_a -q  # chỉ một nhóm
```

| Thư mục | Kiểm thử cái gì |
| --- | --- |
| `common/` | Thư viện dùng chung: bộ chấm đáp án, điểm chất lượng theo quy tắc |
| `stage_a/` | Năm bước của Giai đoạn A |
| `tools/` | Các tiện ích ngoài pipeline |

## common/

- **test_answers.py** (38 bài) — nhiều nhất trong dự án. Tách `\boxed{}` có ngoặc lồng nhau, 16 cặp
  phải coi là bằng nhau, 7 cặp phải coi là khác nhau, ba lý do chấm, tính bất biến của chuẩn hoá.
- **test_rule_score.py** (8 bài) — bốn tiêu chí LIMO, dạng biến thể không được tính,
  z-score ở ba trường hợp biên, và kiểm tra tổng trọng số bằng 1.

## stage_a/

- **test_a1_prepare.py** (7 bài) — tính ổn định của qid, và **tái lập thuật toán cũ**: so trực tiếp
  cách rút tập kiểm định mới với đoạn `random.seed(42); sample; sample` của bản cũ.
- **test_a2_a3_pipeline.py** (10 bài) — mô phỏng nguyên luồng a2 và a3 bằng client giả.
  Đáng đọc nhất nếu muốn hiểu dòng chảy dữ liệu. Gồm bài chạy bù và bài chia lô song song.
- **test_a4_quality.py** (20 bài) — đọc JSON giám khảo ở mọi dạng hỏng gặp thật:
  bọc trong dấu nháy ba, LaTeX làm sai ký tự thoát, xuống dòng thật trong chuỗi, bị cắt cụt, và lớp cứu hộ.
- **test_a5_embed.py** (5 bài) — phép đo phân bố khoảng cách, gồm bài phát hiện phân bố dồn cục.

## stage_b/, stage_c/ (thêm 30/09)

- **stage_b/test_seven_b_plan.py** — ba phương án của mô hình 7 tỷ, một seed, bỏ LocalNat; `b1_fit` không
  ghi trường `local_nat` khi bỏ, và từ chối chạy bù làm file fit lẫn hai loại dòng.
- **stage_c/test_metrics.py** — Acc@4 là TRUNG BÌNH 4 lượt (Phụ lục A.4 bài RSR), Pass@4 là có ít nhất một
  lượt đúng; cài đặt đánh giá khớp A.4 và độ dài sinh bằng độ dài huấn luyện. Các bài này cố ý ghi cứng
  giá trị đã chốt: sửa cấu hình mà không sửa quyết định thì bài phải hỏng.

- **stage_b/test_b2_select.py** (thêm 02/10) — dữ liệu giả dựng qua `a4.combine` và `b1_fit.aggregate` để
  đúng tên khoá của file thật. Giữ các lời hứa của bảng phương án: RSR lấy giá trị THẤP nhất còn mọi tín hiệu
  khác lấy cao nhất; Fit-only chọn đúng cùng tập với RSR; hàm mục tiêu trả nghiệm tối ưu so với cách tính tay
  trên mọi tập con; chuỗi thiếu Qual bị loại ở mọi phương án và ĝ của LARK tính lại trên kho còn lại;
  Correct-Only không đổi khi đổi mô hình học; mọi phương án ra cùng tập câu hỏi, đúng k chuỗi mỗi câu;
  biến thể không lọc sơ bộ dừng chứ không âm thầm trở thành QD-RSR.
- **stage_b/test_backfill_wrong.py** (thêm 02/10) — công cụ chấm bù chuỗi sai cho biến thể không lọc sơ bộ.
  Giám khảo, bộ mã hoá và `b1_fit` được thay bằng bản giả; phần ghép Qual dùng đúng `a4.combine`. Giữ các điều
  sau: chấm bù không đụng tới `quality.jsonl`, `judge.jsonl`, `embeddings.npz` và file fit hiện có; bước giám
  khảo không chạy khi thiếu `--max-cost`; Qual của kho mở rộng được chuẩn hoá lại trên nhóm lớn hơn còn điểm
  giám khảo thô giữ nguyên; chuỗi sai giám khảo không chấm được chỉ bị loại khỏi kho mở rộng; No-Filter không
  đổi sau khi chấm bù. `test_b2_select.py` thêm bài chạy hai tiến trình với `PYTHONHASHSEED` khác nhau và so
  md5 của mọi file train (thứ tự cố định dùng `hashlib`, không dùng `hash()`).
  Sửa 02/10 sau khi đo trên dữ liệu thật: điểm quy tắc của chuỗi sai chấm trên thang của kho gốc
  (`rule_scores_with_reference`), nên câu không có chuỗi sai giữ nguyên Qual và tập chọn; thêm kho
  `all_complete` (bỏ chuỗi bị cắt) và bộ đếm chuỗi bị cắt trong tóm tắt của `b2_select`.
- **stage_b/test_b3_train.py** (thêm 02/10) — phần không cần torch của `b3_train`: token kết thúc lượt lấy từ
  khuôn hội thoại; mẫu là đề, chuỗi, rồi token kết thúc lượt, mất mát chỉ tính trên chuỗi; mẫu bị cắt không gắn
  token kết thúc lượt; mỗi mẫu xuất hiện đúng một lần mỗi epoch; xếp theo độ dài không đổi thành phần của lô hiệu
  dụng; tốc độ học theo cosine có khởi động; từ chối tập chọn sai md5, bị khoá, sai k hoặc của mô hình học khác.
  Phần toán của mất mát (tích luỹ gradient bằng đúng một lô lớn, trọng số mẫu) cần torch nên nằm trong
  `python -m src.stage_b.b3_train --selftest`, chạy trên máy có torch.

## tools/
- **test_segment_steps.py** (thêm 03/10) — công cụ cắt bước bằng GLM-4.5-Air cho Local Naturalness đúng bài gốc.
  Mô hình được thay bằng bản giả. Giữ các điều sau: ranh giới bước được áp lên văn bản GỐC nên các bước luôn phủ
  kín chuỗi, kể cả khi mô hình bỏ sót hoặc viết lại câu; LaTeX chép với một dấu gạch chéo (JSON hỏng) vẫn định vị
  được; ký hiệu mở đầu như `\[` đi cùng bước của nó; lô thử rút mẫu cố định, tắt chế độ suy nghĩ, chạy lại không
  gọi thêm; chạy toàn bộ bắt buộc có `--max-cost`; file lời nhắc còn là chỗ giữ chỗ thì dừng.

- **test_api_tool.py** — đếm đúng số mô hình lỗi, cảnh báo khi không có `\boxed`.
- **test_convert_old_pilot.py** — giữ nguyên qid, ba tình huống lỗi.
- **test_check_grader.py** — đếm đúng từng loại đồng thuận và lệch.
- **test_compare_judges.py** — hạng có giá trị bằng nhau, Spearman ở biên, bảng thiên lệch cùng họ.

## Nguyên tắc khi viết thêm

**Không viết cứng giá trị lấy từ cấu hình.** Hai lần bộ kiểm thử đã hỏng vì lý do này:
một lần khi đổi tên khoá `local_nat`, một lần khi đổi mô hình Qwen sang bản VL.
Đọc từ `load_config()` thay vì gõ thẳng chuỗi.
