# Fashion-MNIST example

MNISTと共通の小型CNN・学習コードを使って、Fashion-MNISTの10クラスを分類します。
入力と出力のshapeがMNISTと同じなので、Rust推論器も共通化できます。

リポジトリルートから学習を実行します。

```bash
uv run python examples/mnist/python/train.py \
  --config examples/fashion_mnist/config.toml
```

W&Bを使わない試運転は次の通りです。

```bash
uv run python examples/mnist/python/train.py \
  --config examples/fashion_mnist/config.toml \
  --wandb-mode disabled
```

結果は`outputs/fashion_mnist/<run-name>/`に保存されます。評価と再exportにも同じ設定を
指定してください。

```bash
uv run python examples/mnist/python/evaluate.py \
  outputs/fashion_mnist/<run-name>/best.pt \
  --config examples/fashion_mnist/config.toml

uv run python examples/mnist/python/export.py \
  outputs/fashion_mnist/<run-name>/best.pt \
  outputs/fashion_mnist/<run-name>/model.bin \
  --config examples/fashion_mnist/config.toml
```

量子化済みモデルをRustで評価します。

```bash
cargo run --release -p mnist-inference -- \
  --model outputs/fashion_mnist/<run-name>/model.q8.bin \
  --images data/fashion_mnist/FashionMNIST/raw/t10k-images-idx3-ubyte \
  --labels data/fashion_mnist/FashionMNIST/raw/t10k-labels-idx1-ubyte
```
