# AHC015 PPO解法

## 方針

各ターンに4方向の傾斜結果を正確に生成し、PPOで学習したactorで選ぶ。actorは方向を直接表す4出力を
持たず、4個のafterstateを共有ネットワークで個別に採点する。これにより盤面回転・反転と味番号の
対称性を特徴生成側で処理でき、学習後のactorを小さいRust推論器へそのまま移せる。

最初に10時間学習したPPO bestは、学習やcheckpoint選択に未使用の2,000ケースで平均698,013点だった。
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
| rollout episodes / iteration | 128（基本学習）、4,096（採用済み継続設定） |
| transitions / iteration | 12,672（基本学習）、405,504（採用済み継続設定） |
| PPO epochs | 4（基本学習）、1（採用済み継続設定） |
| minibatch size | 1,024 |
| AdamW learning rate | `2e-4`（基本学習）、`2.5e-4`（fine-tuning）、`3e-4`（成熟モデルの継続） |
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

actorとcriticはそれぞれ144 channel、9個のdepthwise-separable residual blockを持つ。未来列による
FiLMを含む実験モデルのactorは409,969 parameter、actorとcriticを合わせると819,938 parameterである。

```text
board channels 0..14:
    Conv 3x3, 15 -> 144, ReLU
    [DepthwiseConv 3x3, ReLU, Conv 1x1, skip, ReLU] x 9
    Global average pooling -> 144

future channels 15..17:
    Flatten 300
    Linear 300 -> 144, ReLU
    Linear 144 -> 144, ReLU

future conditioning:
    Linear 144 -> 288
    Split into gamma, beta (144 each)
    Stem output h <- (1 + gamma) * h + beta

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

その後、rollout 128局、batch 1,024、learning rate `2e-4`へ変更し、4時間runのbestからさらに
10時間継続した。新bestは固定512ケースで725,449点、未使用の独立2,000ケースで723,675点となった。
同じ独立ケース上で最初の10時間モデルは693,826点であり、新モデルが29,849 ±2,810点上回った。

さらに上記bestからlearning rateだけを`2.5e-4`へ上げて2時間継続した。固定512ケースbestは
734,300点（+8,851）、別の未使用2,000ケースでは728,636点となり、同じケース上の継続元best
722,045点を+6,590点上回った。56 iteration中24回はKL判定により3 epochで終了し、平均最適化時間も
113.1秒から100.0秒へ短縮した。KL平均0.0222、clip fraction 13.8%で破綻は見られないため、
初期状態からは`rollout 128 / batch 1,024 / learning rate 2e-4`で学習し、成熟したbestを`2.5e-4`で
fine-tuningする2段階を標準手順とする。`2.5e-4`を初期状態から使う実験はしていない。このfloat actorも
提出モデルの更新候補となったが、この時点ではRustへ埋め込まれていたのは上表の初期PPOモデルだった。

このbestを現行global-average構造のまま`2.5e-4`でさらに10時間継続したところ、固定512ケースbestは
経過7.64時間で745,845点、未使用の独立2,000ケースでは745,457点となった。同じ独立ケース上の
継続元best 729,784点を+15,672 ±2,510点上回ったため、新しいglobal-average float actorとして採用する。
best更新は7時間台まで続いたので継続開始時点では飽和していなかった。一方、最後2.4時間はbest更新が
なく、最後10評価の平均は733,351点だったため、終盤は飽和へ近づいた兆候があるが完全な飽和とは
断定しない。新しい構造・特徴量は、この10時間runと同じ開始checkpoint・seed・時間で比較する。

続いて初期盤面の多様性を増やすため、`rollout 1024 / minibatch 1024 / 2 epochs`へ変更して8時間
継続した。固定512ケースbestは773,498点、未使用の独立2,000ケースでは763,408点となり、同じケース上の
継続元best 739,780点を+23,628 ±2,365点上回った。固定評価の後半5回平均も前半より6,657点高く、
単なる序盤の上振れではないため採用する。10時間換算のoptimizer update数をほぼ維持しつつ、1回の
on-policy rolloutで見る初期盤面を8倍に増やしたことが有効だったと考えられる。最新float actorは
`outputs/ahc015/ppo-20260822-125821/best.pt`となった。

このbestからlearning rateだけを`2.5e-4`から`3e-4`へ上げて10時間継続した。固定512ケースbestは
最後の固定評価となる経過9.71時間で783,207点、未使用の独立2,000ケースでは780,098点となり、同じ
ケース上の継続元best 764,590点を+15,509 ±2,198点上回ったため採用する。新しい最新float actorは
`outputs/ahc015/ppo-20260822-234223/best.pt`である。ただし同じ開始checkpointを`2.5e-4`で継続する
対照runはなく、平均KLも0.0116から0.0114へわずかに下がったため、改善をlearning rate変更だけの効果とは
断定できない。

さらに同じ`3e-4`設定で10時間継続した。固定512ケースbestは時間上限後の最終評価で788,932点、未使用の
独立2,000ケースでは787,369点となり、同じケース上の継続元best 780,157点を+7,212 ±2,196点上回ったため
採用する。最新float actorは`outputs/ahc015/ppo-20260823-225410/best.pt`である。改善幅は前回10時間の
+15,509点から縮小したが、後半5回の固定評価平均は前半より7,047点高く、完全な飽和にはまだ達していない。

同じ設定でもう10時間継続すると、固定512ケースbestはiteration 759で795,693点、未使用の独立2,000ケース
では788,335点となった。同じケース上の継続元best 786,175点との差は+2,160 ±2,046点であり、改善方向では
あるため最新float actorを`outputs/ahc015/ppo-20260824-095458/best.pt`へ更新する。ただし独立評価差は
標準誤差と同程度で、固定評価の後半5回平均も前半より2,717点低く、残り約5時間はbestを更新しなかった。
このため、現行global-average構造を同じ設定で継続するだけの学習は概ね飽和したと判断する。

このbestへ未来列からCNNを条件付けする直接FiLMを追加して10時間継続した。固定512ケースbestは
797,602点、未使用の独立2,000ケースでは791,702点となり、同じケース上の継続元best 789,324点との差は
+2,377 ±2,036点だった。単純継続との差は明確でないが絶対スコアは改善方向なので、
`outputs/ahc015/ppo-20260824-230503/best.pt`を最新float actorとして仮採用する。

このFiLM bestから、rolloutを1,024局・2 epochsから4,096局・1 epochへ変更して10時間継続した。
全特徴をfloat32で保持すると約10.88 GiBに加えて巨大な一時copyが必要になるため、1/100刻みの特徴面を
uint8で保存し、potential面だけlosslessなfloat32配列からminibatch時に復元した。bufferは2.72 GiBとなり、
rollout推論も1,024局ずつに分割した。24 iterationで約973万transitionを収集し、固定512ケースbestは
最終iterationで798,724点となった。未使用の独立2,000ケースでは797,558点で、同じケース上の継続元
791,310点を+6,248 ±2,067点上回った。改善が明確で後半も悪化していないため、rollout 4,096・1 epochを
採用し、最新float actorを`outputs/ahc015/ppo-20260825-105935/best.pt`へ更新する。

global average版bestから`2x2` spatial poolingへ拡張する4時間実験も行った。独立2,000ケースでは
global average版を+5,784 ±2,522点上回ったが、bestは開始0.32時間時点で、その後4時間まで改善しなかった。
また、別seedで継続したglobal average版の対照runがなく、改善をspatial headの効果と分離できない。
新しい表現を利用した学習が成功した根拠として不十分なため、この変更は棄却した。

## 停滞対策の検証と現在の課題

旧設定`rollout 32 / batch 256 / learning rate 1e-4`では、固定評価がiteration 49で621,150、99で
661,221、199で665,252まで伸びた後、best 704,182へ到達するまで約7時間を要した。小さいon-policy
batchでは状態分布の更新が遅いことを主因候補として、次の順に検証した。

1. rolloutを128局、batchを1,024へ拡大すると、`lr=1e-4`では平均KLが0.0065まで下がり、同一時間・
   同一transition数とも旧設定を下回った。大batch化だけでは更新圧が不足した。
2. `lr=2e-4`へ上げると平均KLは0.0147となり、4時間時点で旧設定を約7,500点上回った。さらに10時間
   継続して独立評価723,675点に到達し、rollout拡大とlearning rate調整の組合せを採用した。
3. `lr=2.5e-4`で2時間継続すると独立評価がさらに6,590点改善した。56 iteration中24回は3 epochで
   KL判定に達し、平均最適化時間も11.6%短縮した。平均KL 0.0222、clip fraction 13.8%であり、
   target KL 0.03に対して適度な更新圧になっている。

旧small-batch由来の停滞は改善された。追加で最新bestからentropy係数だけを`0.01`から`0.02`へ上げた
2時間実験では、学習entropyは平均0.343から0.402へ上昇したが、固定評価bestは4,637点、独立2,000局は
4,708点悪化した。このためentropy係数は`0.01`を維持し、今後は次の順で検証する。

1. **大型teacherをrollout 4,096で学習する。** 現行モデルでは独立評価が+6,248 ±2,067点改善し、
   unique transition throughputもrollout 1,024版より約59%高くなった。次は提出制約を外してchannel数や
   residual block数を数倍規模へ拡大し、現行モデルを明確に上回るteacherを作る。
2. **大型teacherから提出モデルへ蒸留する。** teacherのsoftな4方向分布と候補間rankingを144 channel以下の
   studentへ学習させ、student自身の訪問状態にもteacherを適用するDAgger型を候補とする。teacherが現行
   studentを独立評価で明確に上回ることを蒸留開始条件とする。
3. **criticを改善する。** 現在は4候補値の単純平均である。候補特徴をpoolする専用state-value head、
   value loss係数、GAE lambdaを比較する。explained varianceは最新runで0.98前後なので優先度は低い。

W&Bにはscoreだけでなく、累積environment transitions、gradient update、KL early stop率、entropy、
clip fraction、score/environment-stepを記録し、何が改善したかを判別できるようにする。

## 提出

学習時は確率的に行動するが、提出時は4候補の `Phi + G_theta` が最大の方向を選ぶ。100ターン目は
盤面が埋まっているのでモデルを呼ばず`F`を返す。actorをper-tensor int8量子化してRustソースへ
埋め込み、起動時にf32へ展開する。量子化後はPython float版との候補一致率と公式スコア差を独立ケースで
確認する。現在の提出モデルの未使用2,000ケースではfloat版779,902点に対しRust int8版781,754点で、差は
+1,853 ±1,050点、公式スコアの完全一致率は72.3%だった。量子化による有意な劣化はないため採用する。
今後のper-output-channel量子化は、明確な量子化劣化が再び観測された場合のみ検討する。

採用した最新float actor `ppo-20260825-105935/best.pt`は、量子化binaryが402,138 bytesである一方、
base93化したRust model dataは532,310 bytesとなり、それだけで提出サイズ制限を超える。提出モデルは
まだ更新せず、大型teacherから提出可能なstudentへ蒸留した後に差し替える。

144 channelモデルのAtCoder実測は約0.9秒、最新提出ファイルのローカル実測は平均0.564秒で、2秒制限には
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
