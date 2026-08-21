# AHC015

PPOで学習した方策をPythonで評価・量子化し、Rustで推論する実装である。主モデルは144 channel、
9 residual blockのactor（368,209 parameter）で、提出時は4方向のafterstateを一括評価する。
解法は `docs/solution.md`、過去を含む実験結果は `docs/experiments.md` にまとめている。

## 学習

MPSとW&B online loggingを使い、約10時間学習する。wall-clock上限に到達したiterationと最後の
固定評価・exportに必要な時間は10時間を少し超えることがある。

```bash
caffeinate -i env PYTHONPATH=python .venv/bin/python \
  -m examples.ahc015.python.train \
  --config examples/ahc015/config.toml \
  --device mps \
  --wandb-mode online \
  --max-hours 10
```

`best.pt` はactor単体で評価・提出用、`best-training.pt` はbest時点の完全な学習状態、`last.pt` は
終了時点の完全な学習状態である。基本学習のbestを`2.5e-4`で2時間fine-tuningする場合は次を使う。

```bash
caffeinate -i env PYTHONPATH=python .venv/bin/python \
  -m examples.ahc015.python.train \
  --config examples/ahc015/config_finetune.toml \
  --device mps \
  --wandb-mode online \
  --max-hours 2 \
  --resume outputs/ahc015/<run-name>/best-training.pt
```

## 評価と提出モデル生成

```bash
PYTHONPATH=python .venv/bin/python -m examples.ahc015.python.evaluate \
  --checkpoint outputs/ahc015/<run-name>/best.pt \
  --episodes 2000 \
  --seed 20260825 \
  --device mps

PYTHONPATH=python .venv/bin/python -m examples.ahc015.python.export \
  --checkpoint outputs/ahc015/<run-name>/best.pt \
  --output outputs/ahc015/<run-name>/model.bin \
  --quantized-output outputs/ahc015/<run-name>/model.q8.bin \
  --rust-output examples/ahc015/rust/src/generated_model.rs

cargo build --release -p ahc015-inference
PYTHONPATH=python .venv/bin/python -m examples.ahc015.python.benchmark
scripts/build_submit.sh examples/ahc015/rust/src/main.rs dist/ahc015.rs
```

AtCoder環境の速度測定入力は次で生成する。

```bash
PYTHONPATH=python .venv/bin/python -m examples.ahc015.python.generate_case \
  --seed 15015 > /tmp/ahc015-benchmark.txt
```

## 検証

```bash
PYTHONPATH=python .venv/bin/python -m pytest python/tests
cargo test --workspace
```
