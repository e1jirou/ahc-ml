# AHC015

特徴量・モデル容量の過剰性を調べるための小型PPO実験である。各ターンの傾斜前盤面を1回だけ入力し、
64 channel・10 residual blockのactorが4方向を同時に出力する。入力はcanonical味3面と空きマス1面の
`(4, 10, 10)`だけで、未来列や派生特徴は使わない。味の置換と盤面の回転・反転は正規化する。

## ローカル学習

`config.toml`はこの小型モデルをMPSでゼロから学習する設定である。rollout 4,096局、PPO 1 epoch、minibatch 1,024、
AdamW learning rate `3e-4`を使用する。GPUメモリ上では128件ずつ処理して勾配を蓄積し、1,024件ごとに
1回だけoptimizerを更新する。macOSでスリープを防ぎながら10時間学習するコマンドは次のとおり。

```bash
caffeinate -i env PYTHONPATH=python .venv/bin/python \
  -m examples.ahc015.python.train \
  --config examples/ahc015/config.toml \
  --device mps \
  --wandb-mode online \
  --max-hours 10
```

ゼロから始めるときは`--resume`を付けない。`best.pt`はactor単体、`best-training.pt`はbest時点の
actor・critic・optimizerを含む継続学習用checkpoint、`last.pt`は終了時点の完全な学習状態である。
wall-clock上限に達したiterationと最後の評価・checkpoint保存は完了させるため、実行時間は10時間を少し超える
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

この実験はPython学習・評価までを対象とし、既存のafterstateモデル用Rust推論器へのexportは行わない。

## 検証

```bash
PYTHONPATH=python .venv/bin/python -m pytest python/tests
cargo test --workspace
```
