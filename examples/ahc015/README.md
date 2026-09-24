# AHC015

現在の提出候補は、未来列を使わない afterstate PPO の 128 channel・10 residual block モデルである。
学習は Kaggle T4 x2 上で行い、rollout・評価を GPU ごとの process に分け、更新は 2-process DDP/NCCL
で同期する。

## Notebook

- `ahc015-distillation.ipynb`: ランダム初期化した 128 channel student を、64 channel 教師から3時間蒸留する。
- `ahc015-ppo.ipynb`: 蒸留済み checkpoint、または直近の `last-training-checkpoint` から PPO を継続する。

どちらも Save & Run All 向けで、Kaggle の Internet On と `GITHUB_TOKEN`、`WANDB_API_KEY` の Secret を
必要とする。本番前には notebook 内の短い smoke test を完走させる。

## 設定

- `config_distill.toml`: 蒸留段階。PPO の potential shaping と policy `alpha=0` を維持し、
  actor の `KL(teacher || student)` 係数を実時間3時間で `1` から `0` に線形減衰する。critic は蒸留しない。
- `config_ppo.toml`: PPO 継続段階。future mode は `none`、global microbatch は512、
  学習率は `1.5e-4`。現在は`large-20260922-133843`のepoch 892のlast checkpointから
  9時間30分継続する。

`best.pt` は actor 単体、`best-training.pt` と `last.pt` は actor・critic・optimizer を含む再開用
checkpoint である。継続学習には、最良性能を使うときは `best-training.pt`、学習軌跡をつなぐときは
`last-training-checkpoint` を使う。

## 評価と提出

Python で checkpoint を評価する。

```bash
PYTHONPATH=python .venv/bin/python -m examples.ahc015.python.evaluate \
  --checkpoint outputs/ahc015/<run-name>/best.pt --episodes 2000 --seed 20260826
```

提出器は最良の actor を int8 量子化して Rust に埋め込む。終盤は80〜92手目にMCTS、
93手目以降に7手の厳密 expectimax を使う。詳細と検証結果は `docs/solution.md` を参照する。
MCTSは盤面hashで同一局面をDAGへ統合し、直後2配置のrank組を層化する。展開時は24ルールでplayoutし、
手のpriorには局所連結度の順位を使う。playout後は末尾2方向を後ろから局所的に修正する。
構造特徴を128 channel actorから蒸留する小型NN priorも実験可能だが、評価では改善しなかったため提出設定では
使用しない。

```bash
PYTHONPATH=python .venv/bin/python -m examples.ahc015.python.export \
  --checkpoint outputs/ahc015/<run-name>/best-training.pt \
  --output outputs/ahc015/model.bin \
  --quantized-output outputs/ahc015/model.q8.bin \
  --rust-output examples/ahc015/rust/src/generated_model.rs
cargo build --release -p ahc015-inference
scripts/build_submit.sh examples/ahc015/rust/src/main.rs outputs/ahc015/submit.rs
```

## 検証

```bash
PYTHONPATH=python .venv/bin/python -m pytest python/tests
cargo test --workspace
```
