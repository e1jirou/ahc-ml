# AHC015 PPO解法

## 方針

各ターンに4方向の傾斜結果を正確に生成し、PPOで学習したactorで選ぶ。actorは方向を直接表す4出力を
持たず、4個のafterstateを共有ネットワークで個別に採点する。これにより盤面回転・反転と味番号の
対称性を特徴生成側で処理でき、学習後のactorを小さいRust推論器へそのまま移せる。

10時間学習したPPO bestは、学習やcheckpoint選択に未使用の2,000ケースで平均698,013点だった。
従来のBellman型afterstate学習は同じケースで607,950点であり、PPOが平均90,063点上回ったため、
主方式をPPOへ変更する。過去の方式と棄却した実験は `experiments.md` に記録として残す。

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
| rollout episodes / iteration | 32 |
| transitions / iteration | 3,168 |
| PPO epochs | 4 |
| minibatch size | 256 |
| AdamW learning rate | `1e-4` |
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

モデル入力は `(18, 10, 10)` の`f32`である。

| channel | 個数 | 内容 |
| --- | ---: | --- |
| 種類別占有 | 3 | canonical味のone-hot |
| 空きマス | 1 | 空きなら1 |
| 連結成分サイズ | 3 | 所属成分サイズ / 100 |
| ターン | 1 | `t / 100` |
| 最終個数 | 3 | `d_i / 100` |
| 残り個数 | 3 | 未配置個数 / 100 |
| 現在のポテンシャル | 1 | `Phi(W)` |
| 全未来列 | 3 | 残り味列のone-hot |

未来の味列は最初から入力で与えられるので利用してよいが、未知の将来配置位置は入力してはならない。

## ネットワーク

actorとcriticはそれぞれ144 channel、9個のdepthwise-separable residual blockを持つ。actorは
368,209 parameter、actorとcriticを合わせた学習モデルは736,418 parameterである。

```text
board channels 0..14:
    Conv 3x3, 15 -> 144, ReLU
    [DepthwiseConv 3x3, ReLU, Conv 1x1, skip, ReLU] x 9
    Global average pooling -> 144

future channels 15..17:
    Flatten 300
    Linear 300 -> 144, ReLU
    Linear 144 -> 144, ReLU

fusion:
    Concatenate -> 288
    Linear 288 -> 288, ReLU
    Linear 288 -> 1
```

Batch NormalizationとDropoutは使わず、学習時とRust推論時の差を避ける。

## 学習結果

主run `ppo-20260820-010211` はMPSで10.009時間、835 iteration、43,069 gradient updateを行った。
固定512ケースのbestはiteration 789で平均704,182点、独立2,000ケースでは次の結果だった。

| 方策 | 平均スコア | `Phi`貪欲比 | 勝率 |
| --- | ---: | ---: | ---: |
| `Phi`貪欲 | 348,470 | 0 | - |
| 従来Bellman版 | 607,950 | +259,480 | 97.45% |
| PPO best | 698,013 | +349,543 | 99.55% |
| PPO Rust int8 | 693,493 | +345,023 | 99.30% |

最後20回の固定評価は平均685,794、標準偏差7,766、範囲35,614だった。従来方式より安定したが、
目標の平均80万には約10.2万点届かない。

## 200 iteration以降の停滞

固定評価はiteration 49で621,150、99で661,221、199で665,252まで急速に伸びた。一方bestの
704,182はiteration 789であり、その後約7時間で約3.9万点しか伸びていない。200以降の平均entropyは
0.395、clip fractionは13.5%、KL early stop率は54.5%だった。方策が完全に決定的になったわけではないが、
32 episodeだけで4 epoch更新するため、同じ小さなon-policy batchへ早く適合してKL制約に当たり、
新しい状態分布を得る速度が律速になっている可能性が高い。

停滞原因を切り分けずに学習時間だけ延ばす優先度は低い。次の順で、各2時間程度のrunを固定seedと
独立seedの両方で比較する。

1. **rolloutを大きくする。** `rollout_episodes = 128`、minibatch `512`または`1024`とし、1 iterationの
   状態多様性を4倍にする。まずPPO epochは4のままにし、sample再利用回数、KL、clip率を比較する。
   wall-clock比較だけでなく、環境step数に対する改善も見る。最優先候補である。
2. **更新圧を下げる。** 大きいrolloutでなおKL early stopが多ければ、epochを`4 -> 2`、または
   learning rateを`1e-4 -> 5e-5`へ下げる。単独でLRだけを下げるより、rollout拡大後に調整する。
3. **entropyをscheduleする。** 現在は係数`0.01`固定である。序盤`0.01`、iteration 200以降`0.02`〜`0.03`
   の再加熱、または目標entropyに基づく自動調整を比較し、局所方策からの脱出を試す。
4. **bestからfine-tuningする。** 新しいrunでは`best-training.pt`にactor、対応critic、optimizerを同時保存する。
   これをresumeし、大batch・低更新圧で継続する。最初の10時間runにはこの完全checkpointがないため、
   そのrunだけはbestからcriticを厳密には復元できない。
5. **criticを改善する。** 現在は4候補値の単純平均である。候補特徴をpoolする専用state-value head、
   value loss係数、GAE lambdaを比較する。explained varianceは終盤0.96前後なので優先度は上記より低い。
6. **複数seed評価をbest選択に使う。** 固定512ケース1組だけでなく、複数seedを交互に評価し、単一集合への
   過適合とcheckpoint偶然差を抑える。
7. **大型teacherから提出モデルへ蒸留する。** 優先度は低いが、提出サイズや推論時間の制約を外した
   大型PPOモデルまたはensembleが十分強くなれば、144 channelのactorへ蒸留する。同一局面の4候補に
   対するteacherのsoftな行動分布と候補間rankingを学習し、student自身の訪問状態にもteacherを適用する
   DAgger型を候補とする。teacherが現行studentを独立評価で明確に上回ることを実施条件とする。

大batch化では、1 iterationが長くなって評価回数が減る点に注意する。W&Bにはscoreだけでなく、
累積environment transitions、gradient update、KL early stop率、entropy、clip fraction、
score/environment-stepを記録し、何が改善したかを判別できるようにする。

## 提出

学習時は確率的に行動するが、提出時は4候補の `Phi + G_theta` が最大の方向を選ぶ。100ターン目は
盤面が埋まっているのでモデルを呼ばず`F`を返す。actorをper-tensor int8量子化してRustソースへ
埋め込み、起動時にf32へ展開する。量子化後はPython float版との候補一致率と公式スコア差を独立ケースで
確認する。採用モデルの独立2,000ケースではRust int8版がfloat版より平均4,520点低く、公式スコアの
完全一致率は27.6%だった。改善余地はあるものの、従来モデルを大幅に上回るため現時点では採用する。
今後はper-tensor量子化からper-output-channel量子化への変更を低優先度で比較する。

144 channelモデルのAtCoder実測は約0.9秒、PPO重みでのローカル実測は平均0.575秒で、2秒制限には
余裕がある。PPO単体で平均80万へ近づけることを優先し、その後に最後5手程度のexpectimaxや、
不確実な局面だけのplayoutを追加する。

## 実装上の確認事項

- 配置番号は直前の傾斜後盤面の空きマスを行優先に数える。
- 傾斜は非空セルの順序を保って指定側へ詰める。
- 報酬差分の次状態は「次のキャンディーを配置した直後」である。
- `gamma = 1`を維持し、99手目の次状態価値を0にする。
- rolloutの旧log probabilityと旧valueを更新前に固定する。
- Python float、量子化復元、Rust forwardの候補順位を同一fixtureで比較する。
- interactive出力は毎ターン改行してflushする。
