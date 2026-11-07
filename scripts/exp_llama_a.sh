#!/usr/bin/env bash
# Hướng 1, giai đoạn A: mô hình học Llama (mặc định llama3b từ 07/11; llama1b đã dừng ở cổng sàn), seed 42, val và MATH-500 (AMC/AIME thêm sau bằng c1_evaluate, nó ghi nối vào summary.json).
# Chạy trên máy Windows (WSL, ~/qd-rsr, .venv đã bật), trong tmux. KHÔNG sửa file này khi nó đang chạy.
#   bash scripts/exp_llama_a.sh            # chạy tuần tự, bỏ qua lần chạy đã xong (train và eval đều chạy bù được)
# Đánh giá hai bộ: 'val' (tập kiểm định 200 câu, dùng cho điều kiện chạy tiếp seed 43, 44) và math500 (bộ báo cáo).
# Thứ tự cố ý: Correct-Only trước (cổng sàn 15%), GRAPE, FWT (cặp để tách lợi ích như ở Qwen), rồi ba đối chứng mô hình dạy.
# Cổng sàn: nếu Acc@4 MATH-500 của Correct-Only seed 42 < 15% thì script DỪNG, không huấn luyện tiếp.
set -euo pipefail
cd "$(dirname "$0")/.."
STUDENT=${STUDENT:-llama3b}      # đổi bằng: STUDENT=llama1b bash scripts/exp_llama_a.sh
SEED=42
METHODS=(correct_only grape familiarity_within_teacher teacher_qwen_first teacher_llama_first teacher_deepseek_first)
LOG=outputs/logs/exp_llama_a.$STUDENT.log
mkdir -p outputs/logs

for m in "${METHODS[@]}"; do
  echo "=== $(date '+%F %T') | $m seed $SEED: huấn luyện" | tee -a "$LOG"
  python -m src.stage_b.b3_train --student "$STUDENT" --method "$m" --seed "$SEED" --resume 2>&1 | tee -a "$LOG"
  echo "=== $(date '+%F %T') | $m seed $SEED: đánh giá" | tee -a "$LOG"
  python -m src.stage_c.c1_evaluate --student "$STUDENT" --method "$m" --seed "$SEED" --benchmarks val math500 2>&1 | tee -a "$LOG"
  if [ "$m" = "correct_only" ]; then
    STUDENT="$STUDENT" python - <<'PY' 2>&1 | tee -a "$LOG"
import json, os, sys
from pathlib import Path
p = Path(f"outputs/results/eval/{os.environ['STUDENT']}_base/correct_only.k3/seed42/summary.json")
if not p.exists():
    sys.exit(f"Không thấy {p}; kiểm tra thư mục kết quả rồi chạy lại.")
acc = json.loads(p.read_text())["benchmarks"]["math500"]["acc@4"]      # tỷ lệ 0..1
print(f"{os.environ['STUDENT']} Correct-Only seed 42, MATH-500 Acc@4 = {100 * acc:.1f}%")
if acc < 0.15:
    sys.exit("CỔNG SÀN: Correct-Only < 15% trên MATH-500. Dừng, bàn lại (3B hoặc cách khác) trước khi chạy tiếp.")
PY
  fi
done
echo "=== $(date '+%F %T') | xong giai đoạn A" | tee -a "$LOG"
