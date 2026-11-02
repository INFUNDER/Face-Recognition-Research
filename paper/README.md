# Paper draft: lightweight thermal-to-visible FR on the edge (IEEE Access)

Build: `latexmk -pdf main.tex` (in this directory).

`main.tex` uses the stock `IEEEtran` journal class so it compiles anywhere. For submission,
download the IEEE Access LaTeX template from the IEEE Author Center and move the content over:
replace `\documentclass[journal]{IEEEtran}` by the template's class, and add the template-only
fields (`\history`, `\doi`, `\tfootnote`, `\corresp`, author biographies and photos).

Before submitting:
- Replace the placeholder authors, affiliation and acknowledgment.
- Check every entry of `refs.bib` against the publisher's page.

## Where every number comes from

| Paper item | Source |
|---|---|
| Table I, II, III, Fig. 1 | `results/hfr/cv/*` via `python scripts/hfr/aggregate.py results/hfr/cv --precisions fp32 int8 int8_s3 int8_s012` (saved in `results/hfr/cv_summary.md`) |
| Paired differences, frontal and VIS→VIS numbers | per-run `eval_*.json` files in `results/hfr/cv/` |
| Table IV (latency, size) | `results/hfr/latency_x86.json` from `scripts/hfr/benchmark_latency.pbs` |
| Learning-rate selection, negative QAT/distillation results | `results/hfr/dev/` (config `scripts/hfr/configs/dev_sweep.txt`) |

## Reproducing

1. Download the frames: `python scripts/hfr/download_speakingfaces.py`
2. Align them: `python scripts/hfr/align_speakingfaces.py`
3. Train and evaluate the cross-validation runs:
   `python scripts/hfr/run_experiments.py scripts/hfr/configs/cv.txt --out_root results/hfr/cv --lfw --int8_stages 3 0,1,2`
   On the PBS cluster, use `scripts/hfr/run_experiments_multi.pbs` instead.

Stage numbering: the code's `model.stages.0`–`3` are stages 1–4 in the paper.
