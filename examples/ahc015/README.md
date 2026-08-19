# AHC015

`docs/solution.md` のafterstate価値学習を実装した例である。Python側でシミュレーション、学習、
評価、量子化exportを行い、Rust側で4候補の特徴生成と推論を行う。
現在の主モデルは144 channel、9 residual block、368,209 parameterである。

## 検証

```bash
.venv/bin/python -m pytest python/tests/test_ahc015.py
cargo test -p ahc015-inference -p ahc-ml
```

## 学習前の速度測定

出力層が0のbootstrapモデルをexportする。このモデルも全層の計算を実行するが、方策は厳密に
`Phi` 貪欲法になる。

```bash
PYTHONPATH=python .venv/bin/python -m examples.ahc015.python.export \
  --output outputs/ahc015/bootstrap/model.bin \
  --quantized-output outputs/ahc015/bootstrap/model.q8.bin \
  --rust-output examples/ahc015/rust/src/generated_model.rs

cargo build --release -p ahc015-inference
PYTHONPATH=python .venv/bin/python -m examples.ahc015.python.benchmark
scripts/build_submit.sh examples/ahc015/rust/src/main.rs dist/ahc015.rs
```

AtCoder環境では `dist/ahc015.rs` をコードテストへ入れ、次で生成した入力を使う。通常の提出は
interactive judgeだが、この入力は100個の配置番号も先に含むため、時間測定用の通常実行でも同じ
99回の推論を通る。

```bash
PYTHONPATH=python .venv/bin/python -m examples.ahc015.python.generate_case \
  --seed 15015 > /tmp/ahc015-benchmark.txt
```

測定値は `docs/experiments.md` に環境、重み、提出ソースサイズとともに追記する。

## 学習と評価

AtCoder上の速度を確認してから学習を開始する。次のコマンドはMPSとW&B online loggingを使い、
約10時間で新しいrollout開始またはgradient updateを止める。最後の固定評価、checkpoint保存、exportに
必要な時間はこの10時間に追加される。target networkは1,000 updateごとにhard updateする。

```bash
caffeinate -i env PYTHONPATH=python .venv/bin/python \
  -m examples.ahc015.python.train \
  --device mps \
  --wandb-mode online \
  --max-hours 10 \
  --iterations 300 \
  --target-update-interval 1000

PYTHONPATH=python .venv/bin/python -m examples.ahc015.python.evaluate \
  --checkpoint outputs/ahc015/<run-name>/best.pt \
  --episodes 1000
```

学習runはcheckpoint、float/int8重み、JSONL metricsを `outputs/ahc015/<run-name>/` に保存する。
開始・完了時の要約は `docs/experiments.md` に追記される。
