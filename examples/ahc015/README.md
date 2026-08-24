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

現行global-averageモデルが飽和するか確認するため、最新bestを同じ構造のまま10時間継続する場合は
次を使う。rolloutには未使用seed `15020`を使う。

```bash
caffeinate -i env PYTHONPATH=python .venv/bin/python \
  -m examples.ahc015.python.train \
  --config examples/ahc015/config_continue.toml \
  --device mps \
  --wandb-mode online \
  --max-hours 10 \
  --resume outputs/ahc015/ppo-20260821-103346/best-training.pt
```

最新global-average bestから、初期盤面の多様性を増やした`rollout 1024 / minibatch 1024 /
2 epochs`で8時間継続する場合は次を使う。

```bash
caffeinate -i env PYTHONPATH=python .venv/bin/python \
  -m examples.ahc015.python.train \
  --config examples/ahc015/config_rollout1024.toml \
  --device mps \
  --wandb-mode online \
  --max-hours 8 \
  --resume outputs/ahc015/ppo-20260821-232137/best-training.pt
```

採用したrollout 1024版bestから、learning rateを`2.5e-4`から`3e-4`へ上げて10時間継続する
実験は次を使う。rolloutには未使用seed `15022`を使う。

```bash
caffeinate -i env PYTHONPATH=python .venv/bin/python \
  -m examples.ahc015.python.train \
  --config examples/ahc015/config_rollout1024_lr3e4.toml \
  --device mps \
  --wandb-mode online \
  --max-hours 10 \
  --resume outputs/ahc015/ppo-20260822-125821/best-training.pt
```

この実験時点の採用済みfloat actorは`outputs/ahc015/ppo-20260822-234223/best.pt`である。

採用済み`3e-4`モデルを同じ学習設定のままさらに10時間継続する場合は次を使う。rolloutには
未使用seed `15023`を使う。

```bash
caffeinate -i env PYTHONPATH=python .venv/bin/python \
  -m examples.ahc015.python.train \
  --config examples/ahc015/config_rollout1024_continue_lr3e4.toml \
  --device mps \
  --wandb-mode online \
  --max-hours 10 \
  --resume outputs/ahc015/ppo-20260822-234223/best-training.pt
```

追加10時間後、この時点のfloat actorは`outputs/ahc015/ppo-20260823-225410/best.pt`となった。

同じ設定でさらに10時間継続した最新float actorは`outputs/ahc015/ppo-20260824-095458/best.pt`である。
独立評価の改善は小幅となり、現行構成は概ね飽和した。現在の提出用Rustは引き続き
`ppo-20260822-234223/best.pt`を埋め込んでいる。

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

現在の`dist/ahc015.rs`は`ppo-20260822-234223/best.pt`を量子化した提出用単一ファイルである。

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
