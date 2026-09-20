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

## tools/

- **test_api_tool.py** — đếm đúng số mô hình lỗi, cảnh báo khi không có `\boxed`.
- **test_convert_old_pilot.py** — giữ nguyên qid, ba tình huống lỗi.
- **test_check_grader.py** — đếm đúng từng loại đồng thuận và lệch.
- **test_compare_judges.py** — hạng có giá trị bằng nhau, Spearman ở biên, bảng thiên lệch cùng họ.

## Nguyên tắc khi viết thêm

**Không viết cứng giá trị lấy từ cấu hình.** Hai lần bộ kiểm thử đã hỏng vì lý do này:
một lần khi đổi tên khoá `local_nat`, một lần khi đổi mô hình Qwen sang bản VL.
Đọc từ `load_config()` thay vì gõ thẳng chuỗi.
