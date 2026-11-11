#!/usr/bin/env bash
# Ô Random@GRAPE (thêm 10/11/2026, theo phản biện bài báo 07/11): tỷ lệ mô hình dạy của GRAPE, rút ngẫu nhiên trong
# từng mô hình dạy. Chọn một lần, huấn luyện ba seed, chấm tầng 1 (MATH-500, AMC23, AIME24, AIME25), in bảng so ghép cặp.
# Chạy trong tmux, từ thư mục gốc của repo, với môi trường Python đã kích hoạt:
#   tmux new -s rag 'bash scripts/exp_random_at_grape.sh 2>&1 | tee outputs/logs/exp_random_at_grape.log'
set -euo pipefail
M=random_at_grape
echo "=== chọn $M ($(date '+%F %T'))"
python -m src.stage_b.b2_select --student qwen1_5b --method $M --expect-questions 1895
python - <<'P'
import json
a = json.load(open("data/stage_b/qwen1_5b_base/select.random_at_grape.k3.json"))["teacher_share"]
g = json.load(open("data/stage_b/qwen1_5b_base/select.grape.k3.json"))["teacher_share"]
print("tỷ lệ mô hình dạy  Random@GRAPE:", a, "\n                   GRAPE       :", g)
assert a == g, "tỷ lệ mô hình dạy khác GRAPE, dừng"
P
for S in 42 43 44; do
  echo "=== huấn luyện $M seed $S ($(date '+%F %T'))"
  python -m src.stage_b.b3_train --student qwen1_5b --method $M --seed $S
  echo "=== chấm $M seed $S ($(date '+%F %T'))"
  python -m src.stage_c.c1_evaluate --student qwen1_5b --method $M --seed $S --tier 1
done
echo "=== so với Correct-Only ($(date '+%F %T'))"
python -m src.tools.eval_table --student qwen1_5b --seeds 42 43 44 --benchmarks math500 \
  --only correct_only random_at_grape familiarity_within_teacher grape --versus correct_only.k3
echo "=== so với GRAPE"
python -m src.tools.eval_table --student qwen1_5b --seeds 42 43 44 --benchmarks math500 \
  --only random_at_grape grape --versus grape.k3
echo "=== xong ($(date '+%F %T'))"
