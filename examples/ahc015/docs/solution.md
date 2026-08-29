# AHC015 PPO解法

## 方針

特徴量とモデル容量の過剰性を調べるため、入力は種類別占有と空きマスだけ、モデル幅は64 channelとする。
傾斜前盤面を1回入力するpre-tilt版と、4方向の傾斜後盤面を1つずつ入力するafterstate版を同条件で比較し、
約15時間の学習後に性能の高かったafterstate版を採用する。pre-tilt版は比較基準として実装を残す。

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

採用するafterstate版のactorは、方向を正規化した各傾斜後盤面の特徴 `x(W_t^a, a)` から残差を1個ずつ
出力する。4方向のlogitは

```text
logit_t(a) = temperature * (Phi(W_t^a) + G_theta(x(W_t^a, a)))
temperature = 12
```

とする。学習時はCategorical分布からsampleし、提出時は最大logitを決定的に選ぶ。temperatureは
argmaxを変えない。actorの最終層をゼロ初期化するので、学習開始時は厳密に`Phi`貪欲方策になる。
これにより完全ランダムな初期方策で悪い状態ばかり集めることを避ける。

criticはactorと同じbackboneの別ネットワークで4候補を評価し、その平均を傾斜前状態の価値とする。

```text
V_psi(B_t) = mean_a V_psi(x(W_t^a, a))
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

味番号には `3!`、盤面には正方形の回転・反転対称性がある。採用するafterstate版では各候補について
次を行う。

1. 次に出現する時刻、最終個数、初出時刻により味をcanonical IDへ写す。
2. 候補を生成した傾斜方向が上になるよう盤面を回転する。
3. 左右反転前後を辞書順比較し、小さい盤面を採用する。

pre-tilt版では傾斜前盤面の回転・反転8通りから辞書順最小を採用し、canonical盤面の4方向出力を
逆写像で元盤面の方向へ戻す。

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

afterstate actor/critic head: Linear width -> 1
pre-tilt actor head: Linear width -> 4
pre-tilt critic head: Linear width -> 1
```

Batch NormalizationとDropoutは使わない。

## 入力方式比較と採用判断

入力方式だけのablationとして、4方向の傾斜後盤面を共有ネットワークへ1つずつ入力する構成を追加した。
各afterstateは対応する傾斜方向が上になるよう回転し、左右反転の辞書順最小を採用する。特徴量、味正規化、
64 channel・10 block、`Phi` baseline、PPO設定はpre-tilt版と揃える。actorは各候補から残差を1個出力し、
criticは4候補の出力平均を状態価値とする。これにより入力方式以外の差を抑える。

両方式をローカルMPSで約15時間ずつ学習し、学習・best選択に使っていない同一2,000ケース
（seed `20260828`）で評価した。

| 入力方式 | 平均スコア | `Phi`貪欲比 | 平均スコアSE |
| --- | ---: | ---: | ---: |
| pre-tilt | 748,286.815 | +402,740.656 | 1,891.610 |
| afterstate | 764,943.646 | +419,397.488 | 1,815.051 |

afterstate版は16,656.832点高く、固定512ケースのbestでもpre-tilt版の`+401,973.766`に対して
`+422,974.883`だった。afterstate版は4候補を評価するため計算量が大きく、累計transitionもpre-tilt版の
99,348,480に対して44,199,936に留まるが、それでも同じwall-clock予算で性能が上回った。以上から、
性能を優先する本解法ではafterstate版を採用し、pre-tilt版は高速な比較基準として残す。

## 実装上の確認事項

- 配置番号は直前の傾斜後盤面の空きマスを行優先に数える。
- 傾斜は非空セルの順序を保って指定側へ詰める。
- 報酬差分の次状態は「次のキャンディーを配置した直後」である。
- `gamma = 1`を維持し、99手目の次状態価値を0にする。
- rolloutの旧log probabilityと旧valueを更新前に固定する。
- canonical盤面のactionを元盤面のactionへ戻してから傾斜する。
