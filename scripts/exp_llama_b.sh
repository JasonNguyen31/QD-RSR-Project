#!/usr/bin/env bash
# Hướng 1 với Llama-3.2-3B (QLoRA 4-bit), chạy cả ba seed theo thứ tự ưu tiên (chốt 08/11/2026).
# Chạy trên máy Windows (WSL, ~/qd-rsr, .venv đã bật), trong tmux. KHÔNG sửa file này khi nó đang chạy.
#   bash scripts/exp_llama_b.sh        # chạy lại sau khi ngắt: mọi bước đã xong đều được bỏ qua
#
# Bước 0  chấm Fam cả kho ở 4-bit (cùng độ chính xác với huấn luyện), rồi b2_select năm phương án.
# Bước 1  seed 42: Correct-Only (cổng sàn 15% MATH-500), Ưu tiên Qwen, Ưu tiên Llama. Đánh giá val và MATH-500.
# Bước 2  điều kiện chạy tiếp trên tập kiểm định (val), seed 42: phương án chỉ chạy seed 43, 44 khi Acc@4 val hơn
#         Correct-Only (đúng chiều L1, L2). Correct-Only là mốc nên luôn chạy đủ ba seed.
# Bước 3  seed 43 rồi 44 cho Correct-Only và các phương án qua bước 2. Chỉ đánh giá MATH-500.
# Bước 4  seed 42: GRAPE và FWT (val và MATH-500). GRAPE chạy tiếp nếu hơn Correct-Only (L0) hoặc hơn FWT (L4) trên
#         val; FWT chạy tiếp nếu GRAPE hơn FWT trên val (L4). Rồi seed 43, 44 cho các phương án chạy tiếp.
# Bỏ Ưu tiên DeepSeek (chốt 08/11). Cổng sàn và điều kiện chạy tiếp đã đăng ký ở trang Notion tuần 11.
set -euo pipefail
cd "$(dirname "$0")/.."
STUDENT=llama3b
LOG=outputs/logs/exp_llama_b.$STUDENT.log
mkdir -p outputs/logs
say() { echo "=== $(date '+%F %T') | $*" | tee -a "$LOG"; }

train_eval() {   # $1 phương án, $2 seed, $3 danh sách bộ đánh giá
  say "$1 seed $2: huấn luyện"
  python -m src.stage_b.b3_train --student "$STUDENT" --method "$1" --seed "$2" --resume 2>&1 | tee -a "$LOG"
  say "$1 seed $2: đánh giá ($3)"
  python -m src.stage_c.c1_evaluate --student "$STUDENT" --method "$1" --seed "$2" --benchmarks $3 2>&1 | tee -a "$LOG"
}

acc() {          # $1 phương án, $2 seed, $3 bộ -> Acc@4 (0..1)
  python - "$STUDENT" "$1" "$2" "$3" <<'PY'
import json, sys
st, m, seed, bench = sys.argv[1:]
p = f"outputs/results/eval/{st}_base/{m}.k3/seed{seed}/summary.json"
print(json.load(open(p))["benchmarks"][bench]["acc@4"])
PY
}

beats() {        # $1 phương án, $2 mốc: in 1 nếu Acc@4 val seed 42 của $1 cao hơn $2
  python -c "import sys; print(int(float(sys.argv[1]) > float(sys.argv[2])))" "$(acc "$1" 42 val)" "$(acc "$2" 42 val)"
}

# ---------------------------------------------------------------- bước 0
say "bước 0: chấm Fam 4-bit"
python -m src.stage_b.b1_fit --student "$STUDENT" --base --load-4bit 2>&1 | tee -a "$LOG"
for m in correct_only teacher_qwen_first teacher_llama_first grape familiarity_within_teacher; do
  python -m src.stage_b.b2_select --student "$STUDENT" --method "$m" 2>&1 | tee -a "$LOG"
done

# ---------------------------------------------------------------- bước 1
for m in correct_only teacher_qwen_first teacher_llama_first; do
  train_eval "$m" 42 "val math500"
  if [ "$m" = correct_only ]; then
    a=$(acc correct_only 42 math500)
    say "Correct-Only seed 42 MATH-500 Acc@4 = $a"
    if python -c "import sys; sys.exit(0 if float(sys.argv[1]) >= 0.15 else 1)" "$a"; then :; else
      say "CỔNG SÀN: Correct-Only < 15% trên MATH-500. Dừng, bàn lại."; exit 1
    fi
  fi
done

# ---------------------------------------------------------------- bước 2, 3
NEXT=(correct_only)
for m in teacher_qwen_first teacher_llama_first; do
  if [ "$(beats "$m" correct_only)" = 1 ]; then NEXT+=("$m"); say "điều kiện chạy tiếp: $m QUA (val hơn Correct-Only)"
  else say "điều kiện chạy tiếp: $m KHÔNG QUA, dừng ở seed 42"; fi
done
for s in 43 44; do for m in "${NEXT[@]}"; do train_eval "$m" "$s" "math500"; done; done

# ---------------------------------------------------------------- bước 4
for m in grape familiarity_within_teacher; do train_eval "$m" 42 "val math500"; done
LATE=()
if [ "$(beats grape correct_only)" = 1 ] || [ "$(beats grape familiarity_within_teacher)" = 1 ]; then LATE+=(grape); fi
if [ "$(beats grape familiarity_within_teacher)" = 1 ]; then LATE+=(familiarity_within_teacher); fi
say "điều kiện chạy tiếp nhóm GRAPE: ${LATE[*]:-không phương án nào}"
for s in 43 44; do for m in "${LATE[@]}"; do train_eval "$m" "$s" "math500"; done; done
say "xong exp_llama_b"
