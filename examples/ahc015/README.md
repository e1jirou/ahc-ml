# AHC015

特徴量・モデル容量の過剰性を調べるための小型PPO実験である。入力はcanonical味3面と空きマス1面の
`(4, 10, 10)`だけ、モデルは64 channel・10 residual blockとし、未来列や派生特徴は使わない。
味の置換と盤面の回転・反転は正規化する。傾斜前盤面を1回入力するpre-tilt版と、傾斜後の4候補を
共有モデルへ1つずつ入力するafterstate版を約15時間ずつ比較した結果、性能の高かったafterstate版を
採用構成とする。pre-tilt版は入力方式の比較基準として残す。

## ローカル学習

`config.toml`はpre-tilt比較基準、`config_afterstate.toml`は採用したafterstate版をMPSでゼロから学習する
設定である。rollout 4,096局、PPO 1 epoch、minibatch 1,024、
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

## 採用構成: Afterstate

`config_afterstate.toml`は、傾斜後の4候補を共有モデルへ1つずつ入力する設定である。pre-tilt版と
同じcanonical味3面＋空きマス1面、64 channel・10 blockを使い、未来列や派生特徴は入力しない。
約15時間ずつの比較では、未使用の同一2,000ケースでpre-tilt版の平均748,286.8に対しafterstate版は
764,943.6となり、afterstate版が16,656.8点上回った。ローカルMPS学習は次のコマンドで実行する。

```bash
caffeinate -i env PYTHONPATH=python .venv/bin/python \
  -m examples.ahc015.python.train \
  --config examples/ahc015/config_afterstate.toml
```

W&Bのrun名は`small-<時刻>`となり、run configの`model.input_mode`で入力方式を識別できる。

## 検証

```bash
PYTHONPATH=python .venv/bin/python -m pytest python/tests
cargo test --workspace
```
