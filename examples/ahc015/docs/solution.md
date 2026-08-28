# AHC015 PPO解法

## 方針

各ターンの傾斜前盤面を1回だけactorへ入力し、4方向の残差を同時に出力する。特徴量とモデル容量の
過剰性を調べるため、入力は種類別占有と空きマスだけ、モデル幅は64 channelとする。

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

actorは傾斜前特徴 `x(B_t)` から4方向の残差 `G_theta(B_t, a)` を同時に出力する。4方向のlogitは

```text
logit_t(a) = temperature * (Phi(W_t^a) + G_theta(B_t, a))
temperature = 12
```

とする。学習時はCategorical分布からsampleし、提出時は最大logitを決定的に選ぶ。temperatureは
argmaxを変えない。actorの最終層をゼロ初期化するので、学習開始時は厳密に`Phi`貪欲方策になる。
これにより完全ランダムな初期方策で悪い状態ばかり集めることを避ける。

criticはactorと同じbackboneの別ネットワークで、傾斜前盤面から状態価値を1個出力する。

```text
V_psi(B_t) = V_psi(x(B_t))
```

criticは学習時のpolicy gradientのbaselineとして使う。

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
| GPU microbatch size | 128（8回の勾配蓄積でminibatch 1,024を維持） |
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

味番号には `3!`、盤面には正方形の回転・反転対称性がある。各傾斜前盤面について次を行う。

1. 次に出現する時刻、最終個数、初出時刻により味をcanonical IDへ写す。
2. 回転・反転8通りを辞書順比較し、小さい盤面を採用する。
3. canonical盤面での4方向出力を、選択した変換の逆写像で元盤面の方向へ戻す。

盤面入力は `(4, 10, 10)` の`f32`である。未来列は入力しない。

| channel | 個数 | 内容 |
| --- | ---: | --- |
| 種類別占有 | 3 | canonical味のone-hot |
| 空きマス | 1 | 空きなら1 |

## ネットワーク

actorとcriticはそれぞれ別ネットワークで、同じ64 channel・10 blockのbackboneを使う。

```text
width, blocks = (64, 10)

board input (4, 10, 10):
    Conv 3x3, 4 -> width, ReLU
    [DepthwiseConv 3x3, ReLU, Conv 1x1, skip, ReLU] x blocks
    Global average pooling -> width

actor head: Linear width -> 4
critic head: Linear width -> 1
```

Batch NormalizationとDropoutは使わない。

## 実装上の確認事項

- 配置番号は直前の傾斜後盤面の空きマスを行優先に数える。
- 傾斜は非空セルの順序を保って指定側へ詰める。
- 報酬差分の次状態は「次のキャンディーを配置した直後」である。
- `gamma = 1`を維持し、99手目の次状態価値を0にする。
- rolloutの旧log probabilityと旧valueを更新前に固定する。
- canonical盤面のactionを元盤面のactionへ戻してから傾斜する。
