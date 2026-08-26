# AHC015

PPOで大型教師モデルを学習し、後で提出用の小型生徒モデルへ蒸留する構成である。教師は256 channel・
10 residual block（986,113 parameter）、生徒は128 channel・8 residual block（244,481 parameter）を使う。
盤面入力は`(12, 10, 10)`、既知の未来列は`(3, 100)`で、未来列はFiLMによる上流の条件付けだけに使う。
解法の詳細は `docs/solution.md`、過去の実験結果は `docs/experiments.md` にまとめている。

## 教師モデルの学習

`config.toml`はゼロから教師を学習する標準設定である。rollout 4,096局、PPO 1 epoch、minibatch 1,024、
AdamW learning rate `3e-4`を使用する。macOSでスリープを防ぎながら10時間学習するコマンドは次のとおり。

```bash
caffeinate -i env PYTHONPATH=python .venv/bin/python \
  -m examples.ahc015.python.train \
  --config examples/ahc015/config.toml \
  --device mps \
  --wandb-mode online \
  --max-hours 10
```

ゼロから始めるときは`--resume`を付けない。`best.pt`は教師actor単体、`best-training.pt`はbest時点の
actor・critic・optimizerを含む継続学習用checkpoint、`last.pt`は終了時点の完全な学習状態である。
wall-clock上限に達したiterationと最後の評価・exportは完了させるため、実行時間は10時間を少し超える
場合がある。

## 評価

checkpoint内のchannel数とblock数は自動判定される。

```bash
PYTHONPATH=python .venv/bin/python -m examples.ahc015.python.evaluate \
  --checkpoint outputs/ahc015/<run-name>/best.pt \
  --episodes 2000 \
  --seed 20260826 \
  --device mps
```

## 生徒モデルのexport

蒸留済み生徒checkpointをfloatおよびint8形式へ変換し、Rustの埋め込みデータを生成する。

```bash
PYTHONPATH=python .venv/bin/python -m examples.ahc015.python.export \
  --checkpoint outputs/ahc015/<distillation-run>/best.pt \
  --output outputs/ahc015/<distillation-run>/model.bin \
  --quantized-output outputs/ahc015/<distillation-run>/model.q8.bin \
  --rust-output examples/ahc015/rust/src/generated_model.rs

cargo build --release -p ahc015-inference
PYTHONPATH=python .venv/bin/python -m examples.ahc015.python.benchmark
scripts/build_submit.sh examples/ahc015/rust/src/main.rs dist/ahc015.rs
```

## 検証

```bash
PYTHONPATH=python .venv/bin/python -m pytest python/tests
cargo test --workspace
```
