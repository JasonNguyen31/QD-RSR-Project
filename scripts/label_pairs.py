"""Đo độ đồng thuận giữa người và Gemini trên nhãn chiến lược của 180 cặp (thêm 10/11/2026, phản biện B1.3d).

  1. make   rút N cặp (seed cố định) từ embed_cmp_labels.jsonl, ghi một trang HTML để người gán nhãn đọc đề và
            hai lời giải rồi bấm "Cùng" hay "Khác". Trang KHÔNG chứa nhãn của Gemini. Mỗi người mở trang, gán
            độc lập, rồi bấm "Tải kết quả" để lưu file labels_<tên>.json.
        python scripts/label_pairs.py make --n 60
  2. score  so nhãn của từng người với Gemini (và giữa hai người nếu có hai file): tỉ lệ trùng và Cohen's kappa.
        python scripts/label_pairs.py score labels_viet.json labels_vi.json
"""
from __future__ import annotations

import argparse
import html
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT = Path("data/label_check")


def verdict(text: str):
    s = re.sub(r"^\s*```(?:json)?|```\s*$", "", (text or "").strip())
    try:
        v = str(json.loads(s[s.find("{"): s.rfind("}") + 1]).get("verdict", "")).strip().lower()
    except Exception:  # noqa: BLE001
        v = ""
    if v in ("same", "different"):
        return v
    low = (text or "").lower()
    if "different" in low and "same" not in low:
        return "different"
    if "same" in low and "different" not in low:
        return "same"
    return None


def read_jsonl(p: Path):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


PAGE = """<!doctype html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Gán nhãn chiến lược</title>
<script>window.MathJax={tex:{inlineMath:[['$','$'],['\\\\(','\\\\)']]}};</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-chtml.js"></script>
<style>
:root{--bg:#fafaf7;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#e3e2dc;--acc:#2f5d8a}
@media (prefers-color-scheme: dark){:root{--bg:#1b1b1a;--fg:#ecebe6;--mut:#a3a29c;--card:#252524;--line:#3a3a37;--acc:#8db4dc}}
body{background:var(--bg);color:var(--fg);font:16px/1.55 Georgia,serif;margin:0;padding:16px}
main{max-width:1100px;margin:auto}.bar{position:sticky;top:0;background:var(--bg);padding:8px 0;border-bottom:1px solid var(--line);
display:flex;gap:8px;flex-wrap:wrap;align-items:center;z-index:2}button{font:inherit;padding:6px 14px;border:1px solid var(--line);
background:var(--card);color:var(--fg);border-radius:6px;cursor:pointer}button.on{background:var(--acc);color:var(--bg);border-color:var(--acc)}
.q{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;margin:12px 0;white-space:pre-wrap}
.sol{display:grid;grid-template-columns:1fr 1fr;gap:12px}@media(max-width:760px){.sol{grid-template-columns:1fr}}
.sol div{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;white-space:pre-wrap;overflow-x:auto}
.rule{color:var(--mut);font-size:14px}h3{margin:4px 0}
</style></head><body><main>
<p class="rule"><b>Cùng chiến lược</b>: cùng hướng đi tới đáp án (cùng ý chính, cách đặt bài, chuỗi biến đổi). Khác cách diễn đạt,
kí hiệu, lượng phép tính viết ra, thứ tự các bước kiểm tra độc lập, hay một bên kiểm tra lại đáp án: vẫn là <b>cùng</b>.<br>
<b>Khác chiến lược</b>: bản thân hướng đi khác, ví dụ lập phương trình so với thử từng trường hợp, đi xuôi từ giả thiết so với
đi ngược từ đáp án, đại số so với hình học hay tổ hợp.</p>
<div class="bar"><span id="pos"></span><button id="prev">← Trước</button><button id="next">Sau →</button>
<button id="same">Cùng (C)</button><button id="diff">Khác (K)</button><span id="done"></span>
<input id="who" placeholder="tên người gán" size="12"><button id="save">Tải kết quả</button></div>
<div id="view"></div></main>
<script>
const PAIRS = __DATA__;
let i = 0, lab = {};
try { lab = JSON.parse(localStorage.getItem('lab') || '{}'); } catch (e) {}
const esc = s => s;
function show() {
  const p = PAIRS[i];
  document.getElementById('pos').textContent = 'Cặp ' + (i + 1) + '/' + PAIRS.length;
  document.getElementById('view').innerHTML = '<div class="q"><h3>Đề bài</h3>' + p.q + '</div><div class="sol"><div><h3>Lời giải A</h3>'
    + p.a + '</div><div><h3>Lời giải B</h3>' + p.b + '</div></div>';
  document.getElementById('same').className = lab[p.id] === 'same' ? 'on' : '';
  document.getElementById('diff').className = lab[p.id] === 'different' ? 'on' : '';
  document.getElementById('done').textContent = 'đã gán ' + Object.keys(lab).length + '/' + PAIRS.length;
  if (window.MathJax && MathJax.typesetPromise) MathJax.typesetPromise();
  window.scrollTo(0, 0);
}
function set(v) { lab[PAIRS[i].id] = v; try { localStorage.setItem('lab', JSON.stringify(lab)); } catch (e) {}
  if (i < PAIRS.length - 1) i++; show(); }
document.getElementById('prev').onclick = () => { if (i > 0) i--; show(); };
document.getElementById('next').onclick = () => { if (i < PAIRS.length - 1) i++; show(); };
document.getElementById('same').onclick = () => set('same');
document.getElementById('diff').onclick = () => set('different');
document.addEventListener('keydown', e => { if (e.target.tagName === 'INPUT') return;
  if (e.key === 'c' || e.key === 'C') set('same'); if (e.key === 'k' || e.key === 'K') set('different');
  if (e.key === 'ArrowLeft') document.getElementById('prev').click(); if (e.key === 'ArrowRight') document.getElementById('next').click(); });
document.getElementById('save').onclick = () => {
  const who = (document.getElementById('who').value || 'nguoi_gan').trim();
  const blob = new Blob([JSON.stringify({who, labels: lab}, null, 1)], {type: 'application/json'});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'labels_' + who + '.json'; a.click(); };
show();
</script></body></html>"""


def make(args) -> int:
    wd = Path(args.workdir)
    labels = read_jsonl(wd / "embed_cmp_labels.jsonl")
    cands = {c["tid"]: c for c in read_jsonl(wd / "candidates.jsonl")}
    questions = {q["qid"]: q for q in read_jsonl(wd / "questions.jsonl")}
    rng = random.Random(args.seed)
    pick = rng.sample(labels, args.n)
    pairs, key = [], {}
    for k, r in enumerate(pick):
        a, b = r["key"].split("__")
        if rng.random() < 0.5:                       # đảo vị trí A/B để người gán không đoán theo thứ tự
            a, b = b, a
        pid = f"p{k:03d}"
        key[pid] = {"key": r["key"], "gemini": verdict(r["text"]),
                    "same_teacher": a.split("|")[1] == b.split("|")[1]}
        pairs.append({"id": pid, "q": html.escape(questions[a.split("|")[0]]["question"]),
                      "a": html.escape(cands[a]["text"]), "b": html.escape(cands[b]["text"])})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "label_pairs.html").write_text(PAGE.replace("__DATA__", json.dumps(pairs, ensure_ascii=False)),
                                          encoding="utf-8")
    (OUT / "label_key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    n_diff = sum(v["gemini"] == "different" for v in key.values())
    print(f"Đã ghi {OUT}/label_pairs.html ({args.n} cặp) và {OUT}/label_key.json (nhãn Gemini, đừng mở khi đang gán).")
    print(f"Trong mẫu: Gemini gán 'khác' cho {n_diff}/{args.n} cặp.")
    return 0


def kappa(x, y):
    n = len(x)
    po = sum(a == b for a, b in zip(x, y)) / n
    cats = ("same", "different")
    pe = sum((x.count(c) / n) * (y.count(c) / n) for c in cats)
    return po, (po - pe) / (1 - pe) if pe < 1 else float("nan")


def score(args) -> int:
    key = json.loads((OUT / "label_key.json").read_text(encoding="utf-8"))
    people = []
    for f in args.files:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        people.append((d.get("who", Path(f).stem), d["labels"]))
    for who, lab in people:
        ids = [p for p in key if p in lab and key[p]["gemini"]]
        po, k = kappa([lab[p] for p in ids], [key[p]["gemini"] for p in ids])
        print(f"{who} so với Gemini: {len(ids)} cặp, trùng {100 * po:.1f}%, kappa {k:.2f}; "
              f"người gán 'khác' {sum(lab[p] == 'different' for p in ids)}, Gemini {sum(key[p]['gemini'] == 'different' for p in ids)}")
    if len(people) == 2:
        (w1, l1), (w2, l2) = people
        ids = [p for p in key if p in l1 and p in l2]
        po, k = kappa([l1[p] for p in ids], [l2[p] for p in ids])
        print(f"{w1} so với {w2}: {len(ids)} cặp, trùng {100 * po:.1f}%, kappa {k:.2f}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="step", required=True)
    m = sub.add_parser("make")
    m.add_argument("--workdir", default="data/stage_a")
    m.add_argument("--n", type=int, default=60)
    m.add_argument("--seed", type=int, default=20261110)
    s = sub.add_parser("score")
    s.add_argument("files", nargs="+")
    args = ap.parse_args(argv)
    return make(args) if args.step == "make" else score(args)


if __name__ == "__main__":
    raise SystemExit(main())
