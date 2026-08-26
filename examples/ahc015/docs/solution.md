# AHC015 PPO解法

## 方針

各ターンに4方向の傾斜結果を正確に生成し、PPOで学習したactorで選ぶ。actorは方向を直接表す4出力を
持たず、4個のafterstateを共有ネットワークで個別に採点する。これにより盤面回転・反転と味番号の
対称性を特徴生成側で処理でき、学習後のactorを小さいRust推論器へそのまま移せる。

教師モデルをPPOでゼロから学習し、十分に性能が出た後で提出用の小型生徒モデルへ蒸留する。過去の方式と
棄却した実験は `experiments.md` に記録として残す。

## 問題の定式化

最終盤面にある同種連結成分の大きさを `n_1, ..., n_k`、味 `i` の総数を `d_i` とする。

```text
C(B)   = sum_j n_j(B)^2
D      = sum_i d_i^2
Phi(B) = C(B) / D
score  = round(1,000,000 * Phi(B))
```

`D` は最初に与えられる100個の味列だけで決まり、`Phi` は `[0, 1]` に収まる。大きさ `x`, `y` の
同種成分を結合すると分子は `2xy` 増えるため、大きな塊同士を接続することが重要である。

`B_t` を `t` 個目の配置直後かつ傾斜前の盤面、`T_a` を方向 `a in {F, B, L, R}` への傾斜、
`W_t^a = T_a(B_t)` をafterstateとする。100個目の配置後は盤面が埋まり傾斜で変化しないため、
意味のある意思決定は `t = 1, ..., 99` の99回である。

## 方策

actorはafterstate特徴 `x(W_t^a)` から残差 `G_theta(W_t^a)` を1個出力する。4方向のlogitは

```text
logit_t(a) = temperature * (Phi(W_t^a) + G_theta(W_t^a))
temperature = 12
```

とする。学習時はCategorical分布からsampleし、提出時は最大logitを決定的に選ぶ。temperatureは
argmaxを変えない。actorの最終層をゼロ初期化するので、学習開始時は厳密に`Phi`貪欲方策になる。
これにより完全ランダムな初期方策で悪い状態ばかり集めることを避ける。

criticはactorと同じ構造の別ネットワークで4個のafterstateを評価し、その平均を状態価値とする。

```text
V_psi(B_t) = 1/4 * sum_a Q_psi(x(W_t^a))
```

平均は実際にsampleした行動に依存せず、候補順の置換にも不変なので、policy gradientのbaselineとして
使える。criticは学習時だけ必要で、提出物には含めない。

## 報酬とGAE

最終報酬だけでは99手前へ信号が届きにくい。次の配置直後の盤面を `B_{t+1}` として、密な報酬

```text
r_t = Phi(B_{t+1}) - Phi(B_t)
```

を用いる。最後の意思決定では `B_100` を終端盤面とする。`gamma = 1` なら

```text
sum_{t=1}^{99} r_t = Phi(B_100) - Phi(B_1)
```

となる。`Phi(B_1)` は行動に依存しないので、期待累積報酬の最大化は公式スコアの最大化と同値である。
この性質を保つため `gamma` は必ず1にする。

advantageには `lambda = 0.95` のGAEを使う。

```text
delta_t = r_t + V(B_{t+1}) - V(B_t)
A_t     = delta_t + lambda * A_{t+1}
R_t     = A_t + V(B_t)
```

エピソード境界では次状態価値を0にする。batch全体でadvantageを平均0・標準偏差1へ正規化する。

## PPO更新

rollout時の旧方策を `pi_old`、更新中の方策を `pi_theta` とする。

```text
ratio_t = pi_theta(a_t|s_t) / pi_old(a_t|s_t)
L_policy = -mean(min(
    ratio_t * A_t,
    clip(ratio_t, 1-epsilon, 1+epsilon) * A_t
))
```

criticにも旧valueからのclipを適用し、全lossを

```text
L = L_policy + value_coefficient * L_value - entropy_coefficient * entropy
```

とする。主設定は次の通りである。

| 項目 | 値 |
| --- | ---: |
| rollout episodes / iteration | 4,096 |
| transitions / iteration | 405,504 |
| PPO epochs | 1 |
| minibatch size | 1,024 |
| AdamW learning rate | `3e-4` |
| weight decay | `1e-4` |
| policy clip | `0.2` |
| value clip | `0.2` |
| value coefficient | `0.5` |
| entropy coefficient | `0.01` |
| target KL | `0.03` |
| gradient clip | `1.0` |

各epoch後の平均近似KLがtargetを超えたら、そのiterationの残りepochを打ち切る。rolloutはon-policy
なのでreplay bufferは使わず、1回のPPO更新後に破棄する。

## 対称性と入力特徴

味番号には `3!`、盤面には正方形の回転・反転対称性がある。各候補について次を行う。

1. 候補を生成した傾斜方向が上向きになるよう盤面を回転する。
2. 左右反転前後を辞書順比較し、小さい方を採用する。
3. 次に出現する時刻、最終個数、初出時刻により味をcanonical IDへ写す。

盤面入力は `(12, 10, 10)` の`f32`、未来列入力は `(3, 100)` の`f32`である。未来列の列 `k`
（0始まり）は、配置順 `k + 1` のキャンディーのcanonical味をone-hotで表す。すでに配置済みの列は
すべて0とする。したがって、モデルは各味がこの後いつ現れるかを100手分の順序付き列として参照できる。

| channel | 個数 | 内容 |
| --- | ---: | --- |
| 種類別占有 | 3 | canonical味のone-hot |
| 空きマス | 1 | 空きなら1 |
| 連結成分サイズ | 3 | 所属成分サイズ / 100 |
| ターン | 1 | `t / 100` |
| 残り個数 | 3 | 未配置個数 / 100 |
| 現在のポテンシャル | 1 | `Phi(W)` |

未来の味列は最初から入力で与えられるので利用してよいが、未知の将来配置位置は入力してはならない。

## ネットワーク

教師と生徒は同じ基本構造を使う。提出用の生徒は従来の144 channel・9 block構成をわずかに小さくした
128 channel・8 blockとする。教師は256 channel・10 blockとし、1ネットワーク当たりのparameter数を
生徒のおよそ4倍にする。actorとcriticはそれぞれ別ネットワークである。

未来列は盤面特徴をFiLMで条件付けするためだけに用いる。global pooling後の特徴と結合するfusion層は持たない。

```text
width, blocks = (256, 10) (teacher) / (128, 8) (student)

board input (12, 10, 10):
    Conv 3x3, 12 -> width, ReLU
    [DepthwiseConv 3x3, ReLU, Conv 1x1, skip, ReLU] x blocks
    Global average pooling -> width

future input (3, 100):
    Flatten 300
    Linear 300 -> width, ReLU
    Linear width -> width, ReLU

future conditioning:
    Linear width -> 2 * width
    Split into gamma, beta (width each)
    Stem output h <- (1 + gamma) * h + beta

head:
    Linear width -> 1
```

Batch NormalizationとDropoutは使わず、学習時とRust推論時の差を避ける。

## 蒸留の予定

教師のPPO学習後、教師が出す4方向のsoftな方策分布と候補間の順位を教師信号として、生徒actorを学習する。
まずは教師のrolloutで得た状態を用い、必要なら生徒自身が訪問した状態にも教師を適用するDAgger型の蒸留を
追加する。独立評価で教師が生徒を明確に上回ることを、蒸留を開始する条件とする。

## 提出

学習時は確率的に行動するが、提出時は4候補の `Phi + G_theta` が最大の方向を選ぶ。100ターン目は
盤面が埋まっているのでモデルを呼ばず`F`を返す。actorをper-tensor int8量子化してRustソースへ
埋め込み、起動時にf32へ展開する。量子化後はPython float版との候補一致率と公式スコア差を独立ケースで
確認する。

提出物には蒸留済みの生徒actorだけを含める。量子化binaryとRustへ埋め込むデータを含めて提出サイズ制限を
満たすこと、2秒の実行時間制限内であることを独立ケースで確認する。

計測の結果、実行時間に余裕がある場合は、その時間を終盤の追加探索に使うことを検討する。候補は、未知の
配置位置をsampleして各方向の将来スコアを比較するモンテカルロ法と、残り手数が十分に少ない最終盤に、
起こり得る配置位置を列挙して各方向の厳密な期待値を計算する方法の2つとする。適用するターン数と探索量は、
2秒の制限を安定して満たす範囲で決める。

教師の性能を優先して学習し、その後に生徒への蒸留と提出用量子化を行う。

## 実装上の確認事項

- 配置番号は直前の傾斜後盤面の空きマスを行優先に数える。
- 傾斜は非空セルの順序を保って指定側へ詰める。
- 報酬差分の次状態は「次のキャンディーを配置した直後」である。
- `gamma = 1`を維持し、99手目の次状態価値を0にする。
- rolloutの旧log probabilityと旧valueを更新前に固定する。
- Python float、量子化復元、Rust forwardの候補順位を同一fixtureで比較する。
- interactive出力は毎ターン改行してflushする。
