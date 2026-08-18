# MNIST example

PyTorchで小型CNNを学習し、Rust推論用の重みを生成する最初のexampleです。

モデル構造は次の通りです。

```text
Input: 1 x 28 x 28
Conv2d: 1 -> 32, kernel 3, padding 1
ReLU
MaxPool2d: kernel 2
Conv2d: 32 -> 64, kernel 3, padding 1
ReLU
MaxPool2d: kernel 2
Flatten
Linear: 3136 -> 128
ReLU
Linear: 128 -> 10
Output: logits
```

分類時はlogitsの`argmax`を使うため、Rust推論器にSoftmaxは必要ありません。

Rust推論器は、float32の`model.bin`とint8量子化・圧縮済みの`model.q8.bin`を読み込めます。
実行方法と重みの埋め込み方法はリポジトリルートの[README](../../README.md)を参照してください。

設定は[config.toml](config.toml)にあります。実行方法はリポジトリルートの
[README](../../README.md)を参照してください。

学習時には各層の入出力shapeを含むモデル構造図をSVGとPNGで生成します。W&Bを有効にすると、
PNGがrunの`model/architecture`画像パネルへ記録されます。
