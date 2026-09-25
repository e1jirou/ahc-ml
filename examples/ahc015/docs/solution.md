# AHC015 PPO解法

## 方針

入力は種類別占有と空きマスだけとし、提出モデルの幅は128 channelとする。
傾斜前盤面を1回入力するpre-tilt版と、4方向の傾斜後盤面を1つずつ入力するafterstate版を同条件で比較し、
約15時間の比較で性能の高かったafterstate版を採用する。

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

採用するafterstate版のactorは、方向を正規化した各傾斜後盤面の特徴 `x(W_t^a, a)` から値を1個ずつ
出力する。4方向のlogitは

```text
logit_t(a) = temperature * G_theta(x(W_t^a, a))
temperature = 12
```

とする。学習時はCategorical分布からsampleし、提出時は最大logitを決定的に選ぶ。temperatureは
argmaxを変えない。方策は`G_theta`だけで行動を選ぶ。
したがって、候補4方向のPhiは方策・推論では計算しない。

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

potential shapingを廃止し、終端公式スコアだけを報酬として`lambda = 1`で10時間継続するablationも
実施した。しかし固定評価は学習開始時の778,415を一度も上回らず、未使用5,000ケースでも学習後モデルは
基準より18,189.576点低かった。局内の99行動へ同じ終端結果を割り当てるだけではcredit assignmentの
分散が大きいため、方策でのPhiは除く一方、学習報酬のpotential shapingは維持する。

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
| GPU microbatch size | 512（2 GPUで各256、global microbatch 512） |
| AdamW learning rate | `3e-4`（学習状況を見て減衰） |
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

actorとcriticはそれぞれ別ネットワークで、同じ128 channel・10 blockのbackboneを使う。

```text
width, blocks = (128, 10)

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
比較時は64 channel・10 block、`Phi` baseline、PPO設定をpre-tilt版と揃えた。actorは各候補から
残差を1個出力し、
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
性能を優先する本解法ではafterstate版を採用する。

## 提出用の追加探索

提出器には、最良の128 channel afterstate actorをint8量子化して埋め込む。通常手はactorのargmaxを選び、
終盤だけ追加探索で置き換える。

- 93手目以降は、残り7手をexpectimaxで厳密評価する。将来の配置位置は全列挙し、各配置後は最大の方向を選ぶ。
  transposition tableはターン間でも再利用し、盤面専用の高速ハッシュを使う。8手探索は単独実行でも2秒を
  超えるため採用しない。
- 86〜92手目は、確率的な配置を含むMCTSで初手4方向を比較する。配置後盤面をhash keyにして、異なる経路から
  同じ盤面へ到達した場合はvisit数と価値を共有するDAGとする。未訪問手は局所連結度の順位で展開し、その後は
  PUCTで選択する。leafは24通りのルール方策（味の役割6 permutation × 盤面4 rotation）でplayoutする。
  各初手には同じ将来配置列と同じルールを適用し、モデルの第一候補を覆すには連結度分子で20以上の推定改善を
  要求する。直後2配置のrank組は一巡まで重複しないよう層化する。playout完了後は、そのsimulationで判明した
  配置rank列を固定し、末尾3方向を後ろから各4通り試して終端連結度が改善する方向へ1回だけ置き換える。
  4手以上の修正や複数回の反復は、simulation内の未来配置への過適合が強くなり実スコアを下げた。
  修正長と反復回数は`--mcts-tail-repair-turns`と`--mcts-tail-repair-passes`で再実験できる。
- `--mcts-rollout-depth N`でleaf playoutをN手後に打ち切り、部分盤面の連結度を味別配置数で正規化して
  終局尺度へ射影できる。実験では終局までplayoutする方が良かったため、デフォルトは`0`（打ち切りなし）。
  `--mcts-rollout-cutoff-until`、`--mcts-early-simulations`、`--mcts-early-min-gain`を併用すると、指定手数
  より前だけ短期playoutと専用の探索量・上書き閾値を使える。
- `--mcts-prior tiny-nn`では、連結成分、最大成分、成分数、空きマスとの接触辺、外周を含む露出辺を入力する
  289 parameterの共有MLPをpriorに使える。128 channel actorから蒸留した実験モデルは改善しなかったため、
  提出時のデフォルトは`connectivity`を維持する。
- 固定sample上限に加え、残時間を残りのモンテカルロ手数で割ったdeadlineで打ち切る。`--mc-turns 0` と
  `--exact-turns 0`でそれぞれ無効化できる。
- future rank列とルールの行動列を事前計算し、終端連結度は3味の`u128` bitboard flood fillで求める。
  旧モンテカルロ法は比較用に`--endgame-search mc`で残す。

## 実装上の確認事項

- 配置番号は直前の傾斜後盤面の空きマスを行優先に数える。
- 傾斜は非空セルの順序を保って指定側へ詰める。
- 報酬差分の次状態は「次のキャンディーを配置した直後」である。
- `gamma = 1`を維持し、99手目の次状態価値を0にする。
- rolloutの旧log probabilityと旧valueを更新前に固定する。
- canonical盤面のactionを元盤面のactionへ戻してから傾斜する。
