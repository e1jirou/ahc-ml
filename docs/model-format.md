# Rust用重み形式

`ahc-ml-weights`は、PyTorchの名前付きtensorをRustへ渡すための単純なbinary形式です。
整数はすべてlittle-endianです。現在のformat versionは`1`です。

## Header

| 型 | 内容 |
| --- | --- |
| `[u8; 8]` | ASCII magic `AHCRLWT1` |
| `u32` | format version (`1`) |
| `u32` | tensor数 |

## Tensor entry

各tensorを名前の辞書順で格納します。

| 型 | 内容 |
| --- | --- |
| `u16` | UTF-8 tensor名のbyte長 |
| `[u8; name_length]` | tensor名 |
| `u8` | dtype code。`1`はfloat32 |
| `u8` | rank |
| `[u32; rank]` | shape |
| `u64` | raw dataのbyte長 |
| `[u8; byte_length]` | C-contiguousなraw data |

PyTorch tensorはexport時にCPU、contiguous、float32へ変換されます。Linearのweight shapeなどは
PyTorchの`state_dict`と同じで、暗黙のtransposeは行いません。

同名の`.bin.json`ファイルには、人が確認するためのshape一覧とモデルmetadataを保存します。
Rust推論器はbinaryだけで重みを読み込めますが、モデル構造はRustコード側で定義する必要があります。

formatを変更するときはmagicを流用せず、versionを増やし、PythonとRustのreaderで旧versionの扱いを
明示します。

## 量子化形式

提出への埋め込みには、`model.q8.bin`を使用します。全tensorをint8へ対称量子化します。
rankが2以上のweightは出力channel（axis 0）ごと、それ以外のbiasなどはtensor全体でscaleを
1つ持ちます。

```text
dequantized_value = int8_value * scale
```

量子化binaryのmagicは`AHCRLQ81`、format versionは`1`です。

### Quantized header

| 型 | 内容 |
| --- | --- |
| `[u8; 8]` | ASCII magic `AHCRLQ81` |
| `u32` | format version (`1`) |
| `u32` | tensor数 |

### Quantized tensor entry

| 型 | 内容 |
| --- | --- |
| `u16` | UTF-8 tensor名のbyte長 |
| `[u8; name_length]` | tensor名 |
| `u8` | rank |
| `i8` | quantization axis。`0`は出力channelごと、`-1`はtensor全体 |
| `[u32; rank]` | shape |
| `u32` | scale数 |
| `[f32; scale_count]` | little-endian scale |
| `u64` | int8 dataのbyte長 |
| `[i8; byte_length]` | C-contiguousな量子化済みdata |

## 圧縮container

量子化binary全体をcanonical Huffman符号で圧縮します。重みによって圧縮後のほうが大きくなる
場合は、自動的に非圧縮格納へ切り替えます。Rust側はどちらも同じAPIで読み込めます。

| 型 | 内容 |
| --- | --- |
| `[u8; 8]` | ASCII magic `AHCRLZ01` |
| `u32` | container version (`1`) |
| `u64` | 展開後のbyte長 |
| `u8` | codec。`0`は非圧縮、`1`はcanonical Huffman |
| payload | codec固有data |

Huffman payloadは、最初の256 bytesに各byte値のcode lengthを格納し、その後にMSB-firstの
bitstreamを格納します。復号は期待する展開後byte長に達した時点で終了し、末尾byteのpadding bitは
無視します。

## Rust sourceへの埋め込み

`model_data.rs`では圧縮containerをBase93にし、`MODEL_DATA_BASE93`定数として保存します。
Base93は表示可能ASCIIからRust文字列でescapeが必要な`"`と`\\`を除いた93文字を使います。
提出ソースのサイズは元dataの約1.226倍で、約1.333倍になるBase64より約8%短くなります。Rust側は
Base93復号、Huffman展開、int8からfloat32への復元を起動時に一度だけ行います。

復元後のtensorはC-contiguousな`ndarray::ArrayD<f32>`として保持します。MNIST CNNの構築時に
Conv2d weightを`Array4`、Linear weightを`Array2`、biasを`Array1`へ変換するため、rankやshapeの
不一致は推論前に検出されます。
