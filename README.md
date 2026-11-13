# FQ-Select: selecting reasoning traces by familiarity and quality

Code for the paper *Selecting Reasoning Traces by Familiarity and Quality for Small Language Model Distillation*
(Dinh-Hong Vu, Hoang-Viet Nguyen, Ha-Vi Nguyen Le; Ton Duc Thang University). The paper is under review.

Three teachers (DeepSeek-V3, Qwen2.5-VL-72B-Instruct, Llama-3.3-70B-Instruct) write three solutions to each of 2,000
GSM8K and MATH questions. A selection rule keeps three correct solutions per question, a Qwen2.5-1.5B base student is
fine-tuned with LoRA on the 5,685 kept solutions, and the student is evaluated on MATH-500 and five other benchmarks.
The paper compares more than ten rules and decomposes the gain of GRAPE, the familiarity rule, into the choice of
teacher and the choice of solution within a teacher.

| What | Where |
| --- | --- |
| This code, version used for the paper (v1.0.1-paper) | https://doi.org/10.5281/zenodo.24128917 |
| Every generated solution, all scores, every selection, every evaluation sample | https://huggingface.co/datasets/VietNguyenHoang/fq-select-math-distillation |
| 8-bit GGUF of the FQ-Select v2 student (seed 42) | https://huggingface.co/VietNguyenHoang/Llama-FQSelect-v2-Qwen2.5-1.5B-Q8_0-GGUF |
| 8-bit GGUF of the GRAPE student (seed 42) | https://huggingface.co/VietNguyenHoang/Llama-GRAPE-Qwen2.5-1.5B-Q8_0-GGUF |

The comments and messages inside the code are in Vietnamese.

## Pipeline

| Stage | Module | What it does | Machine we used |
| --- | --- | --- | --- |
| A | `src.stage_a.a1_prepare` | builds the 2,000-question pool (MATH-500 excluded) | Mac |
| A | `src.stage_a.a2_generate` | three teachers write three solutions each, through OpenRouter | Mac |
| A | `src.stage_a.a3_filter` | checks answers (PRM800K normalisation, `src/common/answers.py`), keeps questions with at least three correct solutions | Mac |
| A | `src.stage_a.a4_score_quality` | quality score: rule half (LIMO criteria) and judge half (Gemini 2.5 Flash-Lite, prompt in `src/common/prompts.py`) | Mac |
| B | `src.stage_b.b1_fit` | student-side signals in one forward pass: familiarity (GRAPE), RSR, LARK | Windows, RTX 3060 |
| B | `src.tools.segment_steps`, `src.stage_b.b1_localnat` | step segmentation by GLM-4.5-Air and Local Naturalness | Mac, then Windows |
| B | `src.stage_b.b2_select` | runs a selection rule, writes the training set | either |
| B | `src.stage_b.b3_train` | LoRA fine-tuning (rank 16, alpha 32, learning rate 1e-4, cosine, 3 epochs, batch 16) | Windows |
| C | `src.stage_c.c0_prepare_eval`, `src.stage_c.c1_evaluate` | MATH-500, AMC 2023, AIME 2024/2025, GSM8K, BBH subset; 4 samples, temperature 0.6, top-p 0.95, 3,072 tokens | Windows |
| C | `src.tools.eval_table` | Acc@4, Pass@4 and paired bootstrap intervals (2,000 resamples) | either |
| C | `src.stage_c.c2_deploy` | merge LoRA, convert to GGUF with llama.cpp, quantise, serve and evaluate through Ollama | Mac |
| | `src.tools.make_figures` | every figure of the paper, from the data | either |

All parameters are in `configs/`: `base.yaml` (shared settings), `data.yaml`, `models.yaml` (teachers and judge,
API keys read from environment variables), `student/` and one file per rule in `method/` and `ablation/`.
Every table joins on one key, `tid = "<qid>|<teacher>|<sample_idx>"`.

## Rule names

| Paper | Config | Paper | Config |
| --- | --- | --- | --- |
| Random (Correct-Only) | `method/correct_only` | FWT | `method/familiarity_within_teacher` |
| GRAPE | `method/grape` | QWT | `method/quality_within_teacher` |
| FQ-Select v1 | `method/fq_select` | FQ-Select v2 within teacher | `method/fq_veto_judge_within_teacher` |
| FQ-Select v2 | `method/fq_veto_judge` | Random@GRAPE | `method/random_at_grape` |
| Judge only, Removal only (Table 4) | `method/fq_judge`, `method/fq_veto` | Prefer Qwen, Avoid Qwen | `method/teacher_qwen_first`, `method/teacher_qwen_last` |
| Local Naturalness | `method/local_naturalness` | Pool without Qwen | `method/*_noqwen` |
| RSR, LARK | `method/rsr`, `method/lark` | No-Filter, Token-Length | `method/no_filter`, `method/token_length` |
| QD-RSR | `ablation/fit_quality` | Quality-only | `ablation/quality_only` |

## Setup

Python 3.11 or later (we used 3.13). The main packages are PyTorch, Transformers 5.5, PEFT, Datasets, Hugging Face Hub 1.x (below
2.0, which Transformers 5.5 requires), NumPy, PyYAML, tenacity, the OpenAI client (for OpenRouter), Jinja2, and
cairosvg for the figures. The exact versions we used are in `requirements.txt`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests -q          # offline: no network, no GPU, no API key
```

## Check the paper's numbers without any API call

Download the dataset and point the evaluation table at its `eval/` folder:

```bash
hf download VietNguyenHoang/fq-select-math-distillation --repo-type dataset --local-dir fqdata
python -m src.tools.eval_table --root fqdata/eval --versus correct_only.k3     # Table 2 and paired differences
```

## Run the full pipeline

Stage A calls paid APIs. Set `OPENROUTER_API_KEY` first; `--max-cost` stops a run at a spending limit in US dollars.

```bash
python -m src.stage_a.a1_prepare
python -m src.stage_a.a2_generate --workdir data/stage_a --max-cost 12
python -m src.stage_a.a3_filter --workdir data/stage_a
python -m src.stage_a.a4_score_quality --workdir data/stage_a

python -m src.stage_b.b1_fit --workdir data/stage_a --student qwen1_5b --base
python -m src.stage_b.b2_select --student qwen1_5b --all
python -m src.stage_b.b3_train --student qwen1_5b --method grape --seed 42

python -m src.stage_c.c0_prepare_eval
python -m src.stage_c.c1_evaluate --student qwen1_5b --method grape --seed 42
python -m src.tools.eval_table --versus correct_only.k3
python -m src.tools.make_figures results
```

`scripts/exp_random_at_grape.sh` runs the Random@GRAPE control end to end. `scripts/compare_rsr_official.py` checks
our RSR against the official implementation, and `scripts/pack_hf_dataset.py` with `scripts/pack_hf_generations.py`
build the released dataset. The teachers and the judge are served through APIs that may change, so regenerated
solutions and judge scores will not match exactly; the released ones are those used in the paper.

## Running the released models locally

See the model cards on Hugging Face. They give the Ollama `Modelfile`, the settings for LM Studio, and the MATH-500
accuracy of each file.

## Licence and attribution

The code is released under the MIT licence (`LICENSE`). The released data and models are subject to the terms of
the models that wrote or underlie them; see their cards. Part of the training data was written by
Llama-3.3-70B-Instruct (**Built with Llama**) and by Qwen2.5-VL-72B-Instruct (**Built with Qwen**).

## Citation

Until the paper is published, please cite the code through `CITATION.cff` or https://doi.org/10.5281/zenodo.24126099
(always the latest version).
