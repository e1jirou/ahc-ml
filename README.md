# ahc-ml

AtCoder Heuristic Contest向けに、PyTorchでモデルを学習し、Rustで高速に推論するためのツール群です。

現在は最初の題材としてMNISTのPython学習パイプラインとRust推論器を実装しています。
学習済みモデルはfloat32形式に加え、提出へ埋め込むためのint8量子化・圧縮形式へexportできます。

## 必要なもの

- [uv](https://docs.astral.sh/uv/)
- Python 3.13（`uv`が自動的に用意できます）
- Rust 1.89以降
- W&Bをonlineで使う場合はWeights & Biasesのアカウント

依存関係をインストールします。

```bash
uv sync
```

`dev`と`train`を既定のdependency groupにしているため、通常の`uv sync`と`uv run`では
torchvision、W&B、テスト・整形ツールを含む開発環境が同期されます。

AtCoderのCPython提出環境に合わせ、推論に使うNumPyは2.2.6、PyTorchは2.8.0へ
固定しています。データ取得に使うtorchvision 0.23.0とW&Bは
学習専用dependency groupです。AtCoderにはtorchvisionとW&Bがないため、提出コードからは
importしません。

Rust推論器では`ndarray 0.16.1`を使用し、Rustのversionと同様にAtCoder環境へ合わせています。
読み込んだtensorは`ArrayD<f32>`、CNN内部のweightとactivationは次元が固定された
`Array1`〜`Array4`として保持します。

## MNISTを学習する

W&Bへログインします。

```bash
uv run wandb login
```

学習を開始します。

```bash
uv run python examples/mnist/python/train.py
```

既定の`device = "auto"`は、CUDA、MPS、CPUの順に利用可能なdeviceを選びます。このMacでは
Apple Silicon GPUを使うMPSが選択されます。deviceや設定の一部はCLIから上書きできます。

```bash
uv run python examples/mnist/python/train.py \
  --device mps \
  --epochs 10 \
  --batch-size 256
```

W&Bへ送信せずに試す場合は次のようにします。

```bash
uv run python examples/mnist/python/train.py --wandb-mode disabled
```

offline runを保存する場合は`--wandb-mode offline`を指定します。

学習開始時に、各層とtensor shapeを示すモデル構造図を`model-graph.svg`と
`model-graph.png`へ生成します。onlineまたはofflineのW&B runでは、PNGを
`model/architecture`という画像パネルとして記録します。offlineの場合は、学習終了時に
表示される`wandb sync <offline-run-directory>`を実行してからW&B上で確認します。
`disabled`ではW&Bへ記録しませんが、ローカルの画像は生成します。

図の生成にはGraphvizの`dot`コマンドが必要です。macOSで未導入の場合は
`brew install graphviz`でインストールできます。

## 出力

各学習runは`outputs/mnist/<run-name>/`へ次を保存します。

| ファイル | 内容 |
| --- | --- |
| `best.pt` | 検証精度が最高だったPyTorch checkpoint |
| `last.pt` | 最終epochのPyTorch checkpoint |
| `model.bin` | Rust推論用のfloat32重み |
| `model.bin.json` | 重みのshapeとモデルmetadata |
| `model.q8.bin` | int8量子化・Huffman圧縮したRust推論用重み |
| `model.q8.bin.json` | 量子化誤差、圧縮率、shapeのmanifest |
| `model_data.rs` | 圧縮済み重みをBase93で埋め込んだRust定数 |
| `config.json` | device情報を含む実効設定 |
| `model-graph.svg` | 拡大可能なモデル構造図 |
| `model-graph.png` | W&Bにも記録するモデル構造図 |

checkpointにはモデル、optimizer、epoch、設定、metricsを含みます。`model.bin`と
`model.q8.bin`には推論に必要なtensorだけを含みます。

保存済みcheckpointを評価できます。

```bash
uv run python examples/mnist/python/evaluate.py \
  outputs/mnist/<run-name>/best.pt
```

checkpointを再度Rust形式へexportする場合は次を実行します。

```bash
uv run python examples/mnist/python/export.py \
  outputs/mnist/<run-name>/best.pt \
  outputs/mnist/<run-name>/model.bin
```

このコマンドは同じディレクトリへ`model.q8.bin`と`model_data.rs`も生成します。出力先は
`--quantized-output`と`--rust-output`で変更できます。

## Rustで推論する

MNISTのIDX test dataを、量子化・圧縮モデルで評価します。

```bash
cargo run --release -p mnist-inference -- \
  --model outputs/mnist/<run-name>/model.q8.bin \
  --images data/mnist/MNIST/raw/t10k-images-idx3-ubyte \
  --labels data/mnist/MNIST/raw/t10k-labels-idx1-ubyte
```

1件だけlogitsを確認する場合は`--index 0`、先頭N件だけ評価する場合は`--limit N`を追加します。
`model.bin`も同じ`--model`引数で読み込めます。

圧縮済み重みをRustへ埋め込む場合は、生成先にexampleのmoduleを指定します。

```bash
uv run python examples/mnist/python/export.py \
  outputs/mnist/<run-name>/best.pt \
  outputs/mnist/<run-name>/model.bin \
  --rust-output examples/mnist/rust/src/generated_model.rs

cargo run --release -p mnist-inference -- \
  --embedded \
  --images data/mnist/MNIST/raw/t10k-images-idx3-ubyte \
  --labels data/mnist/MNIST/raw/t10k-labels-idx1-ubyte
```

`generated_model.rs`の`MODEL_DATA_BASE93`が実行binaryへ静的に埋め込まれます。AHCへ提出するときは、
このmoduleと`rust/ahc-ml`の必要部分を問題固有の`main.rs`へ統合します。復号・展開は起動時に1回だけで、
各推論では展開済みfloat32重みを再利用します。

## AtCoder提出用に1ファイル化する

`scripts/bundle_submit.rs`は、提出用entry pointから参照される自作の`mod`と、文字列リテラルで
指定された`include!`、`include_str!`、`include_bytes!`を1ファイルへ展開します。
`scripts/minify_submit.rs`は、展開後のコメント、不要な空白、debug系macroなどを除去します。

問題固有のentry pointを指定して、bundle済みファイルとminify済み提出ファイルを生成します。

```bash
scripts/build_submit.sh path/to/main.rs
# dist/submit.bundle.rs: 確認用の展開済みソース
# dist/submit.rs:        AtCoderへ提出する圧縮済みソース
```

出力先も指定できます。

```bash
scripts/build_submit.sh path/to/main.rs dist/ahcXXX.rs
```

外部crateは展開されません。提出コードで使う外部crateはAtCoderのRust環境にあるものに限ります。
ビルド時には、展開前のソースがAtCoderの提出上限である512 KiBを超えていないことも確認します。
minifyは識別子名を変更しないため、外部crateを含むRustの名前解決を壊しません。

## 開発

```bash
uv run ruff check .
uv run ruff format --check .
uv run python -m pytest
cargo test --workspace
cargo fmt --all --check
```

テストでは、同じ量子化済み重みと入力に対するPython/Rustのlogitsを比較します。重み形式の詳細は
[docs/model-format.md](docs/model-format.md)を参照してください。

## 構成

```text
python/ahc_ml/              問題に依存しないPythonライブラリ
examples/mnist/python/     MNIST固有のモデルと学習コード
examples/mnist/rust/       MNIST用のRust推論CLI
python/tests/              Python単体テスト
rust/ahc-ml/               問題に依存しないRust reader・推論部品
docs/                      Python/Rust間の形式仕様
data/                      dataset cache（Git管理外）
outputs/                   checkpointとexport結果（Git管理外）
```
