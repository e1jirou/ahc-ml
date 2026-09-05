# AHC015 実験記録

学習・評価・提出速度の測定を時系列で追記する。学習runは設定と成果物を
`outputs/ahc015/<run-name>/` に保存し、このファイルには比較に必要な要点だけを残す。

## 2026-08-18 bootstrap推論

- モデル: 120 channel、8 residual block、250,801 parameter
- 重み: He初期化、出力層のみ0（方策は厳密に `Phi` 貪欲）
- 量子化モデル: 248,534 bytes
- 提出ソース: bundle 382,404 bytes、minify後350,027 bytes
- ローカル環境: Apple Silicon Mac、release build、標準入力に固定1ケース
- 1ケース全体: 10回平均0.4023秒、中央値0.4022秒、範囲0.3996〜0.4074秒
  （起動、復号、特徴量、99回の4候補推論、I/Oを含む）
- 備考: AtCoder実機の値ではない。学習開始前にAtCoder上で別途計測する。

## 2026-08-18 144 channelモデル

- 変更: 144 channel、9 residual block、368,209 parameter
- 重み: He初期化、出力層のみ0（方策は厳密に `Phi` 貪欲）
- 量子化モデル: 361,632 bytes
- 提出ソース: bundle 532,004 bytes、minify後492,700 bytes
- 提出上限までの余裕: 31,588 bytes
- ローカル1ケース全体: 10回平均0.5628秒、中央値0.5631秒、範囲0.5607〜0.5658秒
- 120 channel版に対するローカル時間比: 約1.40倍
- AtCoder実測: 約0.9秒
- 判断: 実行時間と提出サイズの双方が制限内なので、この144 channelモデルを学習に採用する

## afterstate-20260818-161824

- status: started
- output: `outputs/ahc015/afterstate-20260818-161824`
- device: mps (Apple Metal Performance Shaders)
- seed: 15015
- wall-clock limit: 2.000 hours
- W&B: online, run ID `xb7lmzz9`
- status: time limit reached
- elapsed: 2.020 hours
- updates: 4237
- best paired gain: 162690.404

### best checkpointの独立評価とRust量子化確認

- checkpoint: update 4000の `best.pt`
- 評価ケース: 2,000件、seed `20260819`（学習およびbest選択には未使用）
- `Phi` 貪欲: 平均345,143.223
- floatモデル: 平均513,877.538、貪欲比+168,734.315 ±2,632.424（標準誤差）、勝率92.00%
- Rust int8モデル: 平均514,671.027、貪欲比+169,527.804 ±2,623.770（標準誤差）、勝率92.65%
- Rust int8とfloatのpaired差: +793.489 ±1,508.629（標準誤差）
- 完全に同じ最終スコアとなった割合: 50.85%
- 判断: 量子化で選択手順は変わるが、平均スコアの有意な劣化は見られないためRustモデルを採用する
- 量子化モデル: 362,224 bytes
- 提出ソース: bundle 533,046 bytes、minify後493,692 bytes
- 提出上限までの余裕: 30,596 bytes
- ローカル1ケース全体: 10回平均0.5647秒、中央値0.5634秒、範囲0.5610〜0.5776秒
- 備考: AtCoder上の学習前144 channelモデル約0.9秒に対し、学習後も演算量は不変

## afterstate-20260818-185203

- status: started
- output: `outputs/ahc015/afterstate-20260818-185203`
- device: mps (Apple Metal Performance Shaders)
- seed: 15015
- wall-clock limit: 2.000 hours
- W&B: online, run ID `p575nb7p`
- status: time limit reached
- elapsed: 2.020 hours
- updates: 4251
- best paired gain: 125129.848

### Polyak soft updateの評価

- 変更: 1,000 updateごとのhard updateを、各updateのPolyak更新（`tau = 0.002`）へ変更
- 固定512ケースのbest: update 3000、貪欲比+125,129.848、勝率85.55%
- 固定512ケースの最終値: update 4251、貪欲比+118,921.219、勝率81.64%
- 独立評価: 2,000件、seed `20260819`（学習およびbest選択には未使用）
- soft版best: 平均470,782.086、貪欲比+125,638.863 ±2,598.921（標準誤差）、勝率85.35%
- hard版best（同じ独立ケース）: 平均513,877.538、貪欲比+168,734.315
- hard版との差: -43,095.452
- 判断: 損失は低下したが方策性能は明確に悪化したため、`tau = 0.002` のsoft updateは不採用

## afterstate-20260818-224843

- status: started
- output: `outputs/ahc015/afterstate-20260818-224843`
- device: mps (Apple Metal Performance Shaders)
- seed: 15015
- wall-clock limit: 10.000 hours
- W&B: online, run ID `wpp4rji5`
- status: time limit reached
- elapsed: 10.021 hours
- updates: 22566
- best paired gain: 257718.711

### 10時間hard update版の評価

- target network: 1,000 updateごとのhard update
- 固定512ケースのbest: update 22,000（iteration 219）、平均606,131.740、
  貪欲比+257,718.711 ±5,101.367（標準誤差）、勝率98.63%
- 学習終了時の現行モデル: update 22,566、平均570,085.854、
  貪欲比+221,672.824 ±5,252.433（標準誤差）、勝率95.51%
- 判断: 評価値には大きな振動があるため、終了時の重みではなく保存済みの `best.pt` を採用する
- 独立評価: 2,000件、seed `20260820`（学習、best選択、過去の独立評価には未使用）
- `Phi` 貪欲: 平均348,713.057
- floatモデル: 平均607,344.258、貪欲比+258,631.202 ±2,718.464（標準誤差）、勝率97.35%
- Rust int8モデル: 平均612,504.886、貪欲比+263,791.830 ±2,734.652（標準誤差）、勝率97.50%
- Rust int8とfloatのpaired差: +5,160.628 ±2,283.744（標準誤差）
- 完全に同じ最終スコアとなった割合: 19.85%
- 2時間hard版float（独立2,000件）の平均513,877.538に対し、10時間版は+93,466.720
- 判断: 未使用ケースでも固定評価の改善を再現し、量子化による有意な劣化も見られないため、
  10時間版のbest checkpointとRust int8モデルを採用する
- W&B: run ID `wpp4rji5`、model graphを `model/architecture` に記録済み
- 量子化モデル: 361,788 bytes
- 提出ソース: bundle 532,442 bytes、minify後493,118 bytes
- 提出ソースSHA-256: `cba5f4bff3ff57cde0bef3e41df250db46152b70d433e50dffea569af9b5628b`
- 提出上限までの余裕: 31,170 bytes
- ローカル1ケース全体: 10回平均0.5645秒、中央値0.5644秒、範囲0.5631〜0.5665秒
- 備考: 旧学習済みモデルのローカル平均0.5647秒と同等であり、学習による実行時間の変化はない

## 2026-08-19 large-batch MPS smoke test

- config: `examples/ahc015/config_large_batch.toml`（実験不採用後に削除）
- 変更: batch 512、rollout 128 episode、25 updates/iteration、replay 100万、hard update 250回ごと
- 比較条件: 1 iterationの学習sample数は現行の `128 x 100` とほぼ同じ
- device: MPS、W&B disabled、3分制限
- 結果: OOMなし、1 iterationで24 updates、12,288 sampleを更新
- rollout: 13.62秒（現行32 episodeでは約3.8秒）
- Bellman target生成: 165.99秒（現行の約12,800 sampleあたり約135秒に対して約23%低速）
- optimizer: 5.95秒
- 判断: MPS上で実行可能。速度上の利得はないが、軌跡数とgradientの多様性を増やす比較実験として
  採用可能。まず2時間runで現行hard版の同時刻性能と比較してから長時間学習を判断する

## afterstate-20260819-104515

- status: started
- output: `outputs/ahc015/afterstate-20260819-104515`
- device: mps (Apple Metal Performance Shaders)
- seed: 15015
- wall-clock limit: 2.000 hours
- W&B: online, run ID `ur3lx9lc`
- status: time limit reached
- elapsed: 2.004 hours
- updates: 1000
- best paired gain: 102545.314

### large-batch 2時間比較

- 設定: batch 512、rollout 128 episode、25 updates/iteration、replay 100万、hard update 250回ごと
- 固定512ケースのbest: update 875（iteration 34）、平均450,958.344、
  貪欲比+102,545.314 ±5,021.516（標準誤差）、勝率81.05%
- 固定512ケースの終了時: update 1,000、平均446,366.953、貪欲比+97,953.924、勝率81.45%
- 独立評価: 2,000件、seed `20260821`（学習および過去の評価には未使用）
- `Phi` 貪欲: 平均349,505.633
- large-batch best: 平均456,867.504、貪欲比+107,361.872 ±2,510.213、勝率82.85%
- 元の2時間hard版: 平均513,096.201、貪欲比+163,590.568 ±2,513.179、勝率92.75%
- 10時間hard版: 平均607,328.975、貪欲比+257,823.343 ±2,731.084、勝率97.40%
- 同時間の元2時間版との差: -56,228.697
- 10時間版との差: -150,461.471
- W&B: run ID `ur3lx9lc`、model graphを `model/architecture` に記録済み
- 判断: 固定評価と独立評価が一致して悪化したため、このlarge-batch組合せは不採用。提出モデルは
  10時間hard版のままとし、このrunを10時間へ延長しない
- 備考: batch、rollout数、replay容量を同時に変更した実験なので、batch拡大単独の因果とは断定しない

## 2026-08-19 終盤exact expectimax教師（実験準備）

- 初期checkpoint: `outputs/ahc015/afterstate-20260818-224843/best.pt`
- 通常学習設定: batch 128、rollout 32、100 updates/iteration、hard target 1,000 updateごと
- 教師: `placed >= 96`（残り4個以下）のafterstateから、配置を一様平均、行動を最大化して終端まで厳密評価
- 教師target: 厳密な期待最終potentialと現在potentialの差
- sampling: batchの50%を教師付き終盤afterstateにする。残り50%は通常replayから抽出する
- 時間: 2時間。固定512ケースのbestに加え、終了後に独立2,000ケースで判断する
- config: `examples/ahc015/config_endgame_teacher.toml`（実験不採用後に削除）
- smoke test: MPS、1 episode、1 updateでcheckpointのupdate 22,000から22,001へ正常に再開
- smoke timing: 教師生成0.109秒、教師16 afterstate、教師loss 0.000202
- 実装上、教師付き50%には破棄されるBellman targetを計算せず、通常標本50%だけにtarget networkを使う
- status: 下記2時間runで検証し、この設定は不採用

## afterstate-20260819-134648

- status: started
- output: `/tmp/ahc015-endgame-smoke/afterstate-20260819-134648`
- device: mps (Apple Metal Performance Shaders)
- seed: 15015
- wall-clock limit: 0.050 hours
- W&B: disabled, run ID `sxzb106b`
- status: completed
- elapsed: 0.001 hours
- updates: 22001
- best paired gain: 403244.000

## afterstate-20260819-134955

- status: started
- output: `outputs/ahc015/afterstate-20260819-134955`
- device: mps (Apple Metal Performance Shaders)
- seed: 15015
- wall-clock limit: 2.000 hours
- W&B: online, run ID `4o4bu0yt`
- status: time limit reached
- elapsed: 2.007 hours
- updates: 29500
- best paired gain: 221438.182

### 終盤exact expectimax教師の2時間比較

- W&B: run ID `4o4bu0yt`、model graphを `model/architecture` に記録済み
- 学習: 10時間hard版bestのupdate 22,000から再開し、2.007時間でupdate 29,500まで実行
- 教師生成: 毎iteration 512 afterstate、約3.8秒。計算量は律速にならなかった
- 教師loss: 初回iterationの0.000114から、best時0.000061、終了時0.000063へ低下
- 固定512ケースのbest: update 26,500（iteration 264）、平均569,851.211、
  `Phi`貪欲比+221,438.182 ±5,619.258（標準誤差）、勝率94.73%
- 元の10時間hard版best（同じ固定ケース）: 平均606,131.740、`Phi`貪欲比+257,718.711
- 固定評価での差: -36,280.529
- 独立評価: 2,000件、seed `20260822`（学習、best選択、過去の独立評価には未使用）
- 終盤教師版best: 平均571,390.328、`Phi`貪欲比+221,940.779 ±2,719.001、勝率96.00%
- 元の10時間hard版best: 平均610,238.004、`Phi`貪欲比+260,788.455 ±2,691.314、勝率97.90%
- 独立評価での差: -38,847.676
- 判断: 固定評価と独立評価が一致して明確に悪化したため、教師50%・learning rate `3e-4`の
  fine-tuning設定は不採用。関連する学習コードと専用configを削除し、提出モデルは10時間hard版bestの
  ままとする
- 解釈: exact教師へのlossは下がった一方、共有networkを終盤標本へ強く偏らせたことで全turnの表現を
  崩した可能性が高い。これはexact expectimax教師自体の否定ではなく、次に試すなら教師比率を
  5〜10%へ下げ、learning rateも`1e-4`以下にして破壊的忘却を抑える

## 2026-08-19 playout policy improvement（実験準備）

- 初期checkpoint: `outputs/ahc015/afterstate-20260818-224843/best.pt`
- config: `examples/ahc015/config_playout_teacher.toml`（実験不採用後に削除）
- 固定教師方策: 上記10時間hard版best（fine-tuning中には更新しない）
- 教師局面: 現行rolloutの`placed = 20..80`から毎iteration 8局面
- 教師評価: 4方向それぞれ4 playout。4方向には同じ未来rank列を使う
- 教師loss: 4方向の期待終局potentialを局面内で中心化した相対action-value回帰
- 通常学習: batch 128、100 updates/iteration、hard target 1,000 updateごとを維持
- fine-tuning: learning rate `1e-4`、補助loss weight `0.025`、2時間
- W&B metrics: 教師loss、元方策との一致率、元方策regret、1位と2位のgap、教師生成時間
- MPS smoke test: 8局面×4方向×4 playout、`placed = 78`で教師生成3.71秒。1 updateを含む
  iteration全体5.98秒。中心化後の教師lossは0.000481、通常lossは0.000051で、weight `0.025`なら
  初回のloss寄与は通常lossの約24%となる
- status: 下記2時間runで検証し、この設定は不採用

## afterstate-20260819-162231

- status: started
- output: `outputs/ahc015/afterstate-20260819-162231`
- device: mps (Apple Metal Performance Shaders)
- seed: 15015
- wall-clock limit: 2.000 hours
- learning rate: 0.0001
- playout teacher: True
- W&B: online, run ID `fc9sl138`
- status: time limit reached
- elapsed: 2.020 hours
- updates: 26367
- best paired gain: 244208.264

### playout policy improvementの2時間比較

- W&B: run ID `fc9sl138`、model graphを `model/architecture` に記録済み
- 学習: 10時間hard版bestのupdate 22,000から再開し、2.020時間でupdate 26,367まで実行
- 教師生成: 毎iteration 8局面×4方向×4 playout、平均7.98秒。iteration平均163.64秒の約4.9%
- 教師統計: 固定教師方策とplayout最良手の平均一致率28.13%、平均regret 34,363点、
  1位と2位の平均gap 25,675点
- 教師loss: 初回0.000140、以降は概ね0.00043〜0.00053。教師bufferの分布が広がるにつれて上昇し、
  明確な低下傾向は見られなかった
- 固定512ケースのbest: update 23,500（iteration 234）、平均592,621.293、
  `Phi`貪欲比+244,208.264 ±5,400.854（標準誤差）、勝率96.48%
- 元の10時間hard版best（同じ固定ケース）: 平均606,131.740、`Phi`貪欲比+257,718.711
- 固定評価での差: -13,510.447
- 独立評価: 2,000件、seed `20260823`（学習、best選択、過去の独立評価には未使用）
- playout版best: 平均592,409.304、`Phi`貪欲比+243,574.877 ±2,774.508、勝率96.70%
- 元の10時間hard版best: 平均612,739.590、`Phi`貪欲比+263,905.164 ±2,653.781、勝率97.95%
- 同一ケース上のplayout版と元モデルの差: -20,330.287 ±2,887.349（標準誤差）
- playout版が元モデルに勝った割合43.80%、同点0.20%
- 判断: 固定評価と独立評価が一致して有意に悪化したため、この4 playout・固定方策・相対価値lossの
  設定は不採用。10時間へ延長せず、関連する学習コードと専用configを削除し、提出モデルは
  10時間hard版bestのままとする
- 解釈: 教師最良手と元方策の一致率がほぼrandom水準であり、各方向4本では局面ごとの行動差に対して
  playout分散が大きすぎる可能性が高い。再挑戦するなら教師本数を大幅に増やしたoffline datasetを
  先に生成して再現性を測る必要があり、現時点では優先しない

## afterstate-20260819-184622

- status: started
- output: `outputs/ahc015/afterstate-20260819-184622`
- device: mps (Apple Metal Performance Shaders)
- seed: 15015
- wall-clock limit: 2.000 hours
- learning rate: 0.0001
- resume checkpoint: `outputs/ahc015/afterstate-20260818-224843/best.pt`
- optimizer reset on resume: True
- W&B: online, run ID `cuzi0wr2`
- status: time limit reached
- elapsed: 2.020 hours
- updates: 26622
- best paired gain: 259506.193

### low learning-rate fine-tuningの2時間比較

- W&B: run ID `cuzi0wr2`、model graphを `model/architecture` に記録済み
- 目的: 10時間hard版後半の評価値の振動が、後半には大きすぎるlearning rateに起因するか確認する
- 学習: 10時間hard版bestのupdate 22,000からoptimizerをリセットして再開し、2.020時間で
  update 26,622まで実行
- 設定: learning rate `1e-4`。144 channelモデル、batch 128、32 rollout、100 updates/iteration、
  hard target 1,000 updateごと、replay 20万件、固定評価512件は元runと同じ
- 固定512ケースのbest: update 25,000（iteration 249）、平均607,919.223、
  `Phi`貪欲比+259,506.193 ±5,094.602（標準誤差）、勝率98.05%
- 元の10時間hard版best（同じ固定ケース）: 平均606,131.740、`Phi`貪欲比+257,718.711
- 固定評価でのbest差: +1,787.482。標準誤差より小さく、明確な改善ではない
- 9回の固定評価: 平均542,372〜607,919、標準偏差19,637、range 65,547、隣接評価間の
  平均絶対変動18,440。元runのupdate 18,500以降10回ではそれぞれ標準偏差51,702、
  range 173,088、平均絶対変動63,043だった
- update 25,000から25,500では52,731点低下しており、低LRでもcheckpoint間の大きな振動は残る
- 独立評価: 2,000件、seed `20260824`（学習、best選択、過去の独立評価には未使用）
- low-LR版best: 平均608,203.610、`Phi`貪欲比+260,734.608 ±2,708.259、勝率97.30%
- 元の10時間hard版best: 平均608,214.948、`Phi`貪欲比+260,745.946 ±2,743.159、勝率97.65%
- 同一ケース上の平均差: low-LR版 − 元モデル = -11.338点
- 判断: 評価変動は従来後半より小さい兆候があるが、依然大きな振動があり、独立性能は元モデルと
  実質同一だった。提出モデルは変更しない。今後強いcheckpointを追加学習する場合の破壊抑制策として
  `1e-4`は候補に残すが、平均80万へ向けた性能改善策としてこのまま延長はしない
- 後片付け: 専用configとoptimizer reset用の実験コードは削除。再検証時はcheckpointからoptimizerを
  復元するとparam groupのlearning rateも`3e-4`へ戻るため、modelだけを読み込んでoptimizerを作り直す

## ppo-20260820-010211

- algorithm: PPO
- status: started
- output: `outputs/ahc015/ppo-20260820-010211`
- device: mps (Apple Metal Performance Shaders)
- seed: 15015
- wall-clock limit: 10.000 hours
- W&B: online, run ID `bg1dyojk`
- status: time limit reached
- elapsed: 10.009 hours
- updates: 43069
- best paired gain: 355768.855
- 完走確認: 835 iteration、43,069 update、実学習10.009時間。MPSとW&B onlineは正常
- 固定512ケースbest: iteration 789（経過9.468時間）、平均704,181.885、`Phi`貪欲比
  +355,768.855 ±5,086.411（標準誤差）、勝率99.61%
- 最終固定評価: iteration 834、平均689,164.775、`Phi`貪欲比+340,751.746 ±5,550.082、
  勝率98.44%
- 最後20回の固定評価: 平均685,794.324、標準偏差7,766、range 35,614、隣接評価間の
  平均絶対変動8,563（最大21,981）。従来のafterstate学習より評価推移はかなり安定している
- 独立評価: 2,000件、seed `20260825`（学習、best選択、過去の独立評価には未使用）
- PPO best: 平均698,013.299、`Phi`貪欲比+349,543.198 ±2,777.774、勝率99.55%
- 従来採用hard版best: 平均607,949.975、`Phi`貪欲比+259,479.873 ±2,629.162、勝率97.45%
- 同一ケース上の平均差: PPO − 従来採用モデル = +90,063.325点
- 判断: best選択seed外でも約9万点改善しており、PPOは明確に有効。現時点の学習済みモデルとして
  PPO bestを優先候補とする。ただし目標の平均80万には約10.2万点届かず、Rust int8版の一致と性能を
  確認してから提出モデルを置き換える
- 採用: PPOを主方式とし、旧Bellman/replay学習コードは削除。提出埋め込み重みもこのPPO bestへ更新する
- 停滞分析: iteration 199の固定平均665,252に対しbestはiteration 789の704,182。iteration 200以降は
  entropy平均0.395、clip fraction平均13.5%、KL early stop率54.5%。次はrollout episodeを32から128へ
  増やす大batch PPOを最優先で比較する
- Rust int8独立評価: 同じ2,000件、平均693,493.287、`Phi`貪欲比+345,023.186 ±2,794.422、
  勝率99.30%。floatとの差は-4,520.012 ±2,140.017、公式スコア完全一致率27.60%。量子化損失は
  小さいが有意な可能性があるため、今後per-channel量子化を検討する
- Rustローカル速度: seed `15015`、warmup 2回後10回。平均0.575秒、median 0.561秒、
  min 0.557秒、max 0.694秒
- 提出ソース: 493,655 bytes、SHA-256
  `958e5ff76707efa90360f5c3a263da9e3bf8063fcbae713c0a8d4982fe34d233`

## rollout 128・batch 1024 事前速度計測

- 目的: PPO停滞対策の2時間比較前に、MPS上の速度とメモリ適合性を確認する
- 新設定: rollout 128局（12,672 transition）、batch 1,024、4 epoch。1 iterationだけ実行
- 新設定の初回実測: rollout 16.882秒、PPO更新110.875秒、合計127.758秒、52 optimizer update
- 現行10時間run平均: rollout 4.380秒、PPO更新25.518秒、非評価iteration合計29.898秒
- 現行runの初回33.125秒との比較では3.86倍、現行平均との比較では4.27倍遅い。新設定の実測には
  MPS初回実行の影響があるため、定常時は約115〜128秒/iterationと見積もる。扱うtransitionも4倍で、
  評価を除くthroughput低下は大きくても約106.0から99.2 transition/秒への6.4%である
- 固定512件評価の追加時間は現行実測で平均66.2秒。5 iterationごとの評価を維持すると、2時間で
  約51〜56 iteration、約64.6万〜71.0万transition、10〜11回の固定評価を見込む。現行runの最初の2時間は
  166 iteration、52.6万transition、8,632 optimizer updateだった
- 新設定の2時間ではoptimizer updateは約2,650〜2,900回と見込む。各batchは4倍大きいため全sampleを
  4 epoch見る点は同じだが、Adamのparameter update回数は約69%減る。勾配分散低下と更新回数低下を
  合わせた変更として評価する
- メモリ不足は発生せず、batch 1,024で完走した

## ppo-20260820-133549

- algorithm: PPO
- status: started
- output: `outputs/ahc015/ppo-20260820-133549`
- device: mps (Apple Metal Performance Shaders)
- seed: 15015
- wall-clock limit: 2.000 hours
- W&B: online, run ID `ch8u6hiu`
- status: time limit reached
- elapsed: 2.044 hours
- updates: 2704
- best paired gain: 310430.637
- 完走確認: 52 iteration、2,704 optimizer update、658,944 transition。MPSとW&B onlineは正常
- 固定512ケースbest: iteration 39（経過1.563時間、506,880 transition）、平均658,843.666、
  `Phi`貪欲比+310,430.637 ±5,582.356（標準誤差）、勝率98.44%
- 現行設定の最初の2時間best: iteration 114、平均667,430.334。新設定bestは8,586.668点低い
- ほぼ同じtransition数での比較: 新設定50 iteration（633,600 transition）の平均655,553に対し、
  現行設定200 iteration（633,600 transition）は665,252で、新設定が約9,699点低い。途中10点の
  比較でも新設定が上回ったのは506,880 transition時点の1回だけだった
- 更新指標: 新設定52 iterationの平均KL 0.00648、clip fraction 7.61%、KL early stop 0%。現行設定の
  最初の166 iterationはそれぞれ0.01258、12.09%、0.60%。大batch化で更新は明確に穏やかになった
- 速度: rollout平均17.623秒、PPO更新平均109.789秒、評価込みiteration平均140.091秒。事前見積もり内
- 独立評価: 2,000件、seed `20260826`（学習、best選択、過去の独立評価には未使用）
- 新2時間best: 平均644,988.626。現行10時間best: 平均698,858.613。同一ケース上の差は
  -53,869.987 ±3,105.691、新モデルが上回った割合34.60%、同点0.45%
- 判断: rollout 128・batch 1,024・learning rate `1e-4`は安定化した一方、2時間および同一transition数で
  スコア改善がなく、このまま10時間へ延長しない。平均KLがtarget 0.03に対して低くearly stopも0%なので、
  batch 1,024のまま再挑戦するならlearning rate `2e-4`を次の短時間比較候補とする

## ppo-20260820-172434

- algorithm: PPO
- status: started
- output: `outputs/ahc015/ppo-20260820-172434`
- device: mps (Apple Metal Performance Shaders)
- seed: 15015
- wall-clock limit: 4.000 hours
- W&B: online, run ID `0mtfaxub`
- status: time limit reached
- elapsed: 4.039 hours
- updates: 5304
- best paired gain: 342745.279
- 完走確認: 102 iteration、5,304 optimizer update、1,292,544 transition。MPSとW&B onlineは正常
- 固定512ケースbest: iteration 94（経過3.749時間、1,203,840 transition）、平均691,158.309、
  `Phi`貪欲比+342,745.279。最終固定評価はiteration 99の684,732.158
- 2時間時点best: 673,193.746。現行小batch設定の最初の2時間best 667,430.334より+5,763.412、
  大batch `1e-4`版best 658,843.666より+14,350.080
- 4時間時点best: 現行小batch設定の683,673.547に対して+7,484.762
- 同一transition数比較: 50 iteration（633,600 transition）以降の固定評価11点中10点で現行設定を
  上回った。iteration 94相当では+33,141点、iteration 99相当では+18,608点
- 更新指標: 全102 iterationの平均KL 0.01473、clip fraction 13.75%、KL early stop 0%。後半52 iterationは
  KL 0.01612、entropy 0.4266で、target KL 0.03を超えずに更新圧を回復できた
- 速度: rollout平均17.671秒、PPO更新平均111.040秒。評価込みiteration平均は約140秒
- 独立評価: 2,000件、seed `20260827`（学習、best選択、過去の独立評価には未使用）
- 今回4時間best: 平均680,058.510、`Phi`貪欲比+329,967.978、勝率98.80%
- 現行10時間best: 平均697,582.639。同一ケース上の差は-17,524.129 ±3,057.898、今回モデルが
  上回った割合45.40%、同点0.25%
- 判断: 4時間モデル単体は現行10時間bestをまだ下回るため提出には採用しない。一方、wall-clockと
  transitionの両方で現行設定の同時点を上回り、`1e-4`大batch版も明確に上回ったため、rollout 128・
  batch 1,024・learning rate `2e-4`を次の長時間学習候補として採用する

## ppo-20260821-001327

- algorithm: PPO
- status: started
- output: `outputs/ahc015/ppo-20260821-001327`
- device: mps (Apple Metal Performance Shaders)
- seed: 15016
- wall-clock limit: 10.000 hours
- W&B: online, run ID `sob80xus`
- status: time limit reached
- elapsed: 10.005 hours
- updates: 17927
- best paired gain: 377035.801
- 継続元: `ppo-20260820-172434/best-training.pt`、iteration 94、固定平均691,158.309
- 完走確認: iteration 95から344まで250 iterationを追加し、累積17,927 optimizer update、
  4,371,840 transition。追加学習10.005時間、累積学習時間はbestまでを含め約13.75時間
- 固定512ケースbest: iteration 339（追加学習9.804時間）、平均725,448.830、`Phi`貪欲比
  +377,035.801。最終iteration 344も721,459.510で、旧best 704,181.885を上回った
- 最後20回の固定評価: 平均716,536.652、標準偏差5,549、range 22,689、隣接評価間の平均絶対変動
  6,307。最初10回平均689,349に対し最後10回平均718,446で、後半にも改善が続いた
- 更新指標: 平均KL 0.01924、clip fraction 13.80%、entropy 0.3749、explained variance 0.9668。
  KL early stop表示は全体46%、最後100 iterationで75%だが、249/250 iterationは4 epochすべての
  52 updateを実行しており、ほぼ常に最終epoch終了時に閾値を超えただけである
- 独立評価: 2,000件、seed `20260828`（学習、best選択、過去の独立評価には未使用）
- 継続run best: 平均723,674.946、`Phi`貪欲比+378,128.788 ±2,584.676、勝率99.70%
- 現行10時間best: 平均693,826.054。同一ケース上の差は+29,848.892 ±2,809.720、新モデルが
  上回った割合59.65%、同点0.40%
- 判断: 固定評価と独立評価の両方で現行モデルを明確に上回った。新float actorを採用候補とし、
  Rust int8量子化後の独立性能と速度を確認してから提出埋め込みモデルを更新する

## ppo-20260821-103346

- algorithm: PPO
- status: started
- output: `outputs/ahc015/ppo-20260821-103346`
- device: mps (Apple Metal Performance Shaders)
- seed: 15017
- wall-clock limit: 2.000 hours
- W&B: online, run ID `xwdnk9zw`
- status: time limit reached
- elapsed: 2.052 hours
- updates: 20267
- best paired gain: 385886.920
- 継続元: `ppo-20260821-001327/best-training.pt`、iteration 339、固定平均725,448.830
- 設定変更: rollout 128、batch 1,024、4 epochを維持し、learning rateのみ`2e-4`から
  `2.5e-4`へ変更。再開後のoptimizerにもconfigのlearning rateを再適用した
- 完走確認: iteration 340から395まで56 iteration、2,600 optimizer update、709,632 transition。
  learning rateは全iterationで`2.5e-4`と記録された
- 固定512ケースbest: iteration 364（経過0.914時間）、平均734,299.949、`Phi`貪欲比
  +385,886.920。継続元bestより+8,851.119。11回の固定評価平均は726,672.381、標準偏差5,262
- 更新指標: 平均KL 0.02217、clip fraction 13.77%、entropy 0.3434。56 iteration中24回は
  3 epoch（39 update）で停止し、残り32回は4 epoch（52 update）を実行した。最適化時間は平均
  100.0秒で、`2e-4`継続runの平均113.1秒から11.6%短縮。transition速度は約349,390件/時
- 独立評価: 2,000件、seed `20260829`（学習、best選択、過去の独立評価には未使用）
- 今回best: 平均728,635.858、`Phi`貪欲比+378,357.950、勝率99.75%
- 継続元best: 平均722,045.497。同一ケース平均では今回bestが+6,590.362点上回り、固定評価と
  独立評価で改善方向が一致した
- 判断: learning rate `2.5e-4`を採用する。KLとclip率に破綻はなく、不要な4 epoch目を一部省いて
  データ収集速度も改善した。新float actorを採用候補とし、提出更新前にRust int8量子化後を評価する

## ppo-20260821-131630

- algorithm: PPO
- status: started
- output: `outputs/ahc015/ppo-20260821-131630`
- device: mps (Apple Metal Performance Shaders)
- seed: 15018
- wall-clock limit: 2.000 hours
- W&B: online, run ID `n2xsclrn`
- status: time limit reached
- elapsed: 2.031 hours
- updates: 21463
- best paired gain: 381250.174
- 継続元: `ppo-20260821-103346/best-training.pt`、iteration 364、固定平均734,299.949
- 設定変更: rollout 128、batch 1,024、4 epoch、learning rate `2.5e-4`を維持し、entropy係数だけを
  `0.01`から`0.02`へ変更した
- 完走確認: iteration 365から417まで53 iteration、2,626 optimizer update、671,616 transition。
  W&B online run IDは`n2xsclrn`で、checkpointとfloat/int8 exportも正常に生成された
- 固定512ケースbest: iteration 409（経過1.735時間）、平均729,663.203、`Phi`貪欲比
  +381,250.174。継続元bestより-4,636.746。10回の固定評価平均は720,882.241、標準偏差5,000
- 更新指標: 学習entropy平均0.4021、rollout entropy平均0.4117で、前runの0.3434、0.3517から
  上昇した。平均KL 0.02222、clip fraction 15.30%、explained variance 0.9685。53 iteration中
  10回は3 epoch（39 update）で停止し、43回は4 epoch（52 update）を実行した
- 独立評価: 2,000件、seed `20260830`（学習、best選択、過去の独立評価には未使用）
- 今回best: 平均722,712.550。継続元bestは平均727,420.534。同一ケース平均では今回bestが
  -4,707.984 ±2,526.370点、今回bestの勝率49.75%、同率0%だった
- 判断: entropyは意図どおり増えたが、固定評価と独立評価がともに悪化したため`0.02`は不採用とし、
  entropy係数を`0.01`へ戻す

## ppo-20260821-170807

- algorithm: PPO
- status: started
- output: `outputs/ahc015/ppo-20260821-170807`
- device: mps (Apple Metal Performance Shaders)
- seed: 15019
- wall-clock limit: 4.000 hours
- W&B: online, run ID `k8phfpfc`
- status: time limit reached
- elapsed: 4.018 hours
- updates: 23803
- best paired gain: 387864.820
- 継続元: `ppo-20260821-103346/best-training.pt`、iteration 364、固定平均734,299.949
- 設定変更: actorとcriticのglobal average poolingを`2x2` adaptive average poolingと線形射影へ
  拡張した。新headは旧global averageと完全に同じ出力で初期化し、既存parameterの重みとAdam momentsを
  引き継いだ。actorは368,209から451,297 parameterへ増加した
- 完走確認: iteration 365から479まで115 iteration、4,966 optimizer update、1,457,280 transition。
  W&B online run IDは`k8phfpfc`で、checkpointと新形式のfloat/int8 exportも正常に生成された
- 固定512ケースbest: iteration 374（経過0.320時間）、平均736,277.850、`Phi`貪欲比
  +387,864.820。継続元bestより+1,977.900。23回の固定評価平均は727,052.287、標準偏差6,155。
  2時間以降はbestを更新せず、最終iteration 479は733,107.428だった
- 更新指標: 平均KL 0.02226、clip fraction 13.75%、entropy 0.3408、explained variance 0.9720で、
  global average版とほぼ同じ。115 iteration中73回は3 epoch、40回は4 epoch、各1回は1、2 epochで
  停止した。spatial headの初期射影からのweight差L2は0.874で、新しい自由度は実際に学習された
- Rust smoke benchmark: 同一環境3回平均でglobal average版0.569秒、spatial版0.578秒（約1.5%増）。
  畳み込み本体は同じなので、追加headの実行時間への影響は小さい
- 独立評価: 2,000件、seed `20260831`（学習、best選択、過去の独立評価には未使用）
- 継続元best: 平均727,727.150。spatial bestは平均733,510.779、同一ケース差
  +5,783.629 ±2,521.699、勝率53.15%、同率0.25%。終了モデルは平均731,322.264、同一ケース差
  +3,595.114 ±2,543.505だった
- 判断: 棄却。独立評価は改善方向だったが、bestが開始0.32時間時点で、その後4時間まで更新されず、
  空間表現を利用した学習が成功した挙動には見えない。また、学習seed `15019`でglobal average版を
  継続する対照runがないため、+5,784点をspatial headの効果と継続学習の軌跡差に分離できない。
  採用モデルは`ppo-20260821-103346/best.pt`のglobal average版に戻す。再検証する場合は同一checkpoint・
  seed・学習時間の対照runを用意する

## 2026-08-21 現行モデルの10時間継続（実験準備）

- 目的: 現行global-averageモデルを追加学習し、性能改善が継続するか、飽和へ向かうかを確認する
- 継続元: `outputs/ahc015/ppo-20260821-103346/best-training.pt`（iteration 364、固定平均734,299.949）
- architectureとoptimizer momentsはそのまま維持する
- 設定: rollout 128、batch 1,024、4 epoch、learning rate `2.5e-4`、entropy係数`0.01`
- rollout seed: 未使用の`15020`、固定評価seed: `515015`、時間: 10時間
- config: `examples/ahc015/config_continue.toml`
- 判断: 後半の評価平均、best更新時刻、独立2,000ケースのpaired比較から、現行モデルの飽和状況を判定する
- status: time limit reached
- output: `outputs/ahc015/ppo-20260821-232137`
- W&B: online、run ID `pq9878o4`
- 完走確認: 10.032時間、iteration 365から645まで281 iteration、3,560,832 transitionを追加し、
  累積8,186,112 transition、31,408 optimizer update。MPSとW&B onlineは正常
- 固定512ケースbest: iteration 579（経過7.643時間）、平均745,844.586、継続元bestより
  +11,544.637。best更新は経過0.17、1.40、1.59、2.66、4.08、4.79、7.64時間に発生した
- 最後10回の固定評価は平均733,351、最後20回は標準偏差5,334、range 20,813。7.64時間以降は
  bestを更新せず、最終固定評価は737,935だった
- 更新指標: 平均KL 0.02159、clip fraction 13.24%、学習entropy 0.3339、rollout entropy 0.3423、
  explained variance 0.9740。281 iteration中157回は3 epoch、124回は4 epochを実行した
- 独立評価: 2,000件、seed `20260901`（学習、best選択、過去の独立評価には未使用）
- 継続元best: 平均729,784.088。今回best: 平均745,456.504。同一ケース差は
  +15,672.416 ±2,509.918、今回bestの勝率55.60%、同率0.25%
- 判断: 固定評価と独立評価の両方で明確に改善したため、新bestをglobal-average float actorとして採用する。
  7時間台までbest更新が続いたため開始時点では飽和していなかった。一方、時間帯別平均は4時間以降
  730,171〜735,600点の範囲で、最後2.4時間はbest更新がないため、終盤は飽和へ近づいた兆候がある。
  完全な飽和とは断定せず、次の構造変更に対する10時間継続baselineとしてこのrunを用いる

## 2026-08-22 rollout 512 / 1024 MPS速度計測

- 目的: 公開されたAHC015 PPO事例の`batch_size=4096`はPPO minibatchではなく並列episode数なので、
  現行実装の`rollout_episodes`を増やした場合のメモリ適合性と速度を確認する
- checkpoint: `outputs/ahc015/ppo-20260821-232137/best-training.pt`（iteration 579）
- 共通設定: minibatch 1,024、2 epochs、learning rate `2.5e-4`、MPS、各1 iteration。
  固定評価は速度測定から除外した
- rollout 512: 50,688 transition、288.387秒。rollout 67.088秒、最適化221.298秒、100 updates、
  KL 0.02207、clip fraction 14.03%。OOMなし
- rollout 1024: 101,376 transition、565.739秒。rollout 135.088秒、最適化430.650秒、198 updates、
  KL 0.02447、clip fraction 15.13%。OOMなし
- unique transition throughputは現行128局・実効3〜4 epochsの110.1件/秒に対し、512局で175.8件/秒、
  1024局で179.2件/秒。固定512局評価の実測約69.2秒を5 iterationごとに含めると、10時間で
  512局版は約604万、1024局版は約630万transitionを見込む。現行10時間runの356万件から約77%増える
- optimizer update見込みは10時間で現行約12,600回、1024局・2 epochsも約12,300回でほぼ同じ。
  gradient step数を大きく減らさず、1回のon-policy rolloutに含む初期盤面を8倍へ増やせる
- 判断: 1024局はメモリ・更新指標・throughputの全てに問題がなく、512局よりわずかに効率がよい。
  記事の4096局を直接使う前段階として、次の長時間実験には`rollout 1024 / minibatch 1024 / 2 epochs`を
  第一候補とする

## 2026-08-22 rollout 1024・8時間実験

- 継続元: `outputs/ahc015/ppo-20260821-232137/best-training.pt`（iteration 579、固定平均745,844.586）
- 変更: rollout episodeを128から1,024へ増やし、PPO epochsを4から2へ減らす。minibatch 1,024、
  learning rate `2.5e-4`、entropy係数`0.01`、global-average architectureは維持する
- rollout seed: 未使用の`15021`、固定評価seed: `515015`
- 時間: 8時間。1 iteration実測と評価時間から約49 iteration、約497万transition、固定評価約10回を見込む
- config: `examples/ahc015/config_rollout1024.toml`
- 判断: 固定評価のbestと後半平均に加え、終了後に未使用2,000ケースで継続元とpaired比較する
- status: time limit reached
- output: `outputs/ahc015/ppo-20260822-125821`
- W&B: online、run ID `hne9hqzn`

## ppo-20260821-232137

- algorithm: PPO
- status: started
- output: `outputs/ahc015/ppo-20260821-232137`
- device: mps (Apple Metal Performance Shaders)
- seed: 15020
- wall-clock limit: 10.000 hours
- W&B: online, run ID `pq9878o4`
- status: time limit reached
- elapsed: 10.032 hours
- updates: 31408
- best paired gain: 397431.557

## ppo-20260822-125821

- algorithm: PPO
- status: started
- output: `outputs/ahc015/ppo-20260822-125821`
- device: mps (Apple Metal Performance Shaders)
- seed: 15021
- wall-clock limit: 8.000 hours
- W&B: online, run ID `hne9hqzn`
- status: time limit reached
- elapsed: 8.115 hours
- updates: 38318
- best paired gain: 425085.432
- 完走確認: 8.115時間、iteration 580から629まで50 iteration、5,068,800 transitionを追加し、
  累積12,418,560 transition、38,318 optimizer update。MPSとW&B onlineは正常
- 固定512ケースbest: iteration 604（経過4.054時間）、平均773,498.461、継続元bestより
  +27,653.875。最終評価も772,727.082だった
- 固定評価10回の前半5回平均は762,400.782、後半5回平均は769,057.463。後半平均が+6,657点高く、
  開始直後だけの上振れではない。全10回の標準偏差は6,109、rangeは17,903
- 更新指標: 平均KL 0.01157、clip fraction 10.14%、学習entropy 0.2927、rollout entropy 0.2961、
  explained variance 0.9817。全50 iterationが2 epochs、198 updateを完遂し、KL early stopは0回
- 速度: rollout平均134.861秒、最適化平均436.215秒、評価込みiteration平均584.103秒。
  事前計測565.739秒に固定評価の償却分を加えた見積もりと整合する
- 独立評価: 2,000件、seed `20260902`（学習、best選択、過去の独立評価には未使用）
- 継続元best: 平均739,779.871。今回best: 平均763,407.717。同一ケース差は
  +23,627.846 ±2,364.637、今回bestの勝率58.40%、同率0.15%
- 判断: 固定評価と独立評価が一致して明確に改善し、後半評価も高水準なので採用する。現行の標準設定を
  `rollout 1024 / minibatch 1024 / 2 epochs / learning rate 2.5e-4`へ更新する候補とする。
  平均KLはtarget 0.03に対して0.0116まで下がったため、次に最適化設定を試す場合はlearning rateを
  `3e-4`へ上げる短時間比較が候補になる

## 2026-08-22 rollout 1024・learning rate 3e-4・10時間実験

- 継続元: `outputs/ahc015/ppo-20260822-125821/best-training.pt`（iteration 604、固定平均773,498.461）
- 変更: rollout 1,024、minibatch 1,024、2 epochs、entropy係数`0.01`を維持し、learning rateだけを
  `2.5e-4`から`3e-4`へ上げる
- 根拠: 採用runの平均KLは0.01157、clip fractionは10.14%、KL early stopは0回で、target KL
  `0.03`に対して更新圧を上げる余地がある。rollout増加に対する単純なLR線形拡大ではなく、epochsを
  4から2へ減らしたことで低下した更新圧を調整する実験と位置付ける
- rollout seed: 未使用の`15022`、固定評価seed: `515015`
- 時間: 10時間。直近の実測から約60 iteration、固定評価約12回を見込む
- config: `examples/ahc015/config_rollout1024_lr3e4.toml`
- 判断基準: 固定評価bestと後半推移を確認し、終了後に未使用2,000ケースで`2.5e-4`版bestとpaired比較する
- status: time limit reached
- output: `outputs/ahc015/ppo-20260822-234223`
- W&B: online、run ID `r2s940ka`

## ppo-20260822-234223

- algorithm: PPO
- status: started
- output: `outputs/ahc015/ppo-20260822-234223`
- device: mps (Apple Metal Performance Shaders)
- seed: 15022
- wall-clock limit: 10.000 hours
- W&B: online, run ID `r2s940ka`
- status: time limit reached
- elapsed: 10.047 hours
- updates: 45644
- best paired gain: 434793.912
- 完走確認: 10.047時間、iteration 605から666まで62 iteration、6,285,312 transitionを追加し、
  累積16,169,472 transition、45,644 optimizer update。MPSとW&B onlineは正常
- 固定512ケースbest: iteration 664（経過9.710時間）、平均783,206.941、継続元bestより
  +9,708.480。12回の固定評価の前半5回平均は770,981.571、後半5回平均は776,034.951で、
  後半が+5,053.380点高い。全12回の標準偏差は4,257、rangeは17,048
- 更新指標: 平均KL 0.01137、clip fraction 9.77%、学習entropy 0.2777、rollout entropy 0.2813、
  explained variance 0.9835。全62 iterationが2 epochs、198 updateを完遂し、このrun中の
  KL early stopは0回。learning rateを上げてもKLは継続元runの0.01157から増えなかった
- 速度: rollout平均134.925秒、最適化平均434.630秒、評価込みiteration平均582.199秒
- 独立評価: 2,000件、seed `20260903`（学習、best選択、過去の独立評価には未使用）
- 継続元best: 平均764,589.828。今回best: 平均780,098.436。同一ケース差は
  +15,508.608 ±2,198.476、今回bestの勝率56.20%、同率0.45%
- 判断: 固定評価と独立評価が一致して明確に改善し、bestが最後の固定評価で更新されたため採用する。
  一方、同じ開始checkpointから`2.5e-4`で継続した対照runはなく、平均KLも増えていないため、改善を
  learning rate変更だけの効果とは断定しない。現時点では`3e-4`を維持し、さらに上げるより飽和を
  確認する継続学習を優先する

## 2026-08-23 最新モデルのRust量子化・提出ファイル生成

- float checkpoint: `outputs/ahc015/ppo-20260822-234223/best.pt`
- 未使用2,000ケース、seed `20260904`: float平均779,901.665、Rust int8平均781,754.434
- Rust int8とfloatの同一ケース差は+1,852.770 ±1,049.693、公式スコア完全一致率72.30%。
  量子化による有意な劣化は見られない
- Rust unit testは7件成功、release buildと提出用単一ファイルのCargo compileも成功
- ローカル実行時間: package binary 10回平均0.565秒、提出用単一ファイル5回平均0.564秒
- 提出ファイル: `dist/ahc015.rs`、493,919 bytes。524,288 byte制限に対して30,369 bytesの余裕
- 判断: 最新int8モデルを提出用Rustへ採用する

## 2026-08-23 rollout 1024・learning rate 3e-4・追加10時間

- 継続元: `outputs/ahc015/ppo-20260822-234223/best-training.pt`（iteration 664、固定平均783,206.941）
- 設定: rollout 1,024、minibatch 1,024、2 epochs、learning rate `3e-4`、entropy係数`0.01`を維持する
- rollout seed: 未使用の`15023`、固定評価seed: `515015`
- 時間: 10時間。直近の実測から約60 iteration、固定評価約12回を見込む
- config: `examples/ahc015/config_rollout1024_continue_lr3e4.toml`
- 判断基準: 固定評価bestと後半推移を確認し、終了後に未使用2,000ケースで継続元とpaired比較する
- status: time limit reached
- output: `outputs/ahc015/ppo-20260823-225410`
- W&B: online、run ID `pa9fgmc5`

## ppo-20260823-225410

- algorithm: PPO
- status: started
- output: `outputs/ahc015/ppo-20260823-225410`
- device: mps (Apple Metal Performance Shaders)
- seed: 15023
- wall-clock limit: 10.000 hours
- W&B: online, run ID `pa9fgmc5`
- status: time limit reached
- elapsed: 10.064 hours
- updates: 57524
- best paired gain: 440518.488
- 完走確認: 10.064時間、iteration 665から726まで62 iteration、6,285,312 transitionを追加し、
  累積22,252,032 transition、57,524 optimizer update。MPSとW&B onlineは正常
- 固定512ケースbest: 時間上限後の最終評価、iteration 726（経過10.043時間）、平均788,931.518、
  継続元bestより+5,724.576。最終評価を含む13回の前半5回平均は779,043.450、後半5回平均は
  786,090.148で、後半が+7,046.698点高い。標準偏差は4,252、rangeは12,950
- 更新指標: 平均KL 0.01037、clip fraction 9.00%、学習entropy 0.2693、rollout entropy 0.2727、
  explained variance 0.9844。全62 iterationが2 epochs、198 updateを完遂し、このrun中の
  KL early stopは0回
- 速度: rollout平均135.877秒、最適化平均434.550秒、評価込みiteration平均583.141秒
- 独立評価: 2,000件、seed `20260905`（学習、best選択、過去の独立評価には未使用）
- 継続元best: 平均780,156.972。今回best: 平均787,368.504。同一ケース差は
  +7,211.532 ±2,196.178、今回bestの勝率52.80%、同率0.30%
- 判断: 固定評価と独立評価が一致して改善したため採用する。bestが時間上限後の最終評価で更新されており、
  完全な飽和には達していない。一方、独立評価の改善幅は前回10時間の+15,509点から+7,212点へ縮小し、
  KLとentropyも低下しているため、限界効用は小さくなっている

## 2026-08-24 rollout 1024・learning rate 3e-4・2回目の追加10時間（準備）

- 継続元: `outputs/ahc015/ppo-20260823-225410/best-training.pt`（iteration 726、固定平均788,931.518）
- 設定: rollout 1,024、minibatch 1,024、2 epochs、learning rate `3e-4`、entropy係数`0.01`を維持する
- rollout seed: 未使用の`15024`、固定評価seed: `515015`
- 時間: 10時間
- config: `examples/ahc015/config_rollout1024_continue2_lr3e4.toml`
- 判断基準: 独立評価の改善が3,000点未満、または後半の固定評価が横ばいなら、現行構成は概ね飽和と判断する
- status: 実験準備完了

## ppo-20260824-095458

- algorithm: PPO
- status: started
- output: `outputs/ahc015/ppo-20260824-095458`
- device: mps (Apple Metal Performance Shaders)
- seed: 15024
- wall-clock limit: 10.000 hours
- W&B: online, run ID `japvvnnk`
- status: time limit reached
- elapsed: 10.038 hours
- updates: 69800
- best paired gain: 447279.818
- 完走確認: 10.038時間、iteration 727から788まで62 iteration、6,285,312 transitionを追加し、
  累積28,537,344 transition、69,800 optimizer update。MPSとW&B onlineは正常
- 固定512ケースbest: iteration 759、平均795,692.848、継続元bestより+6,761.330。12回の固定評価の
  前半5回平均は788,180.826、後半5回平均は785,463.523で、後半が-2,717.303点低い。標準偏差は
  3,625、rangeは14,486。best後の5回の固定評価では更新されなかった
- 更新指標: 平均KL 0.01038、clip fraction 8.91%、学習entropy 0.2597、rollout entropy 0.2627、
  explained variance 0.9857。全62 iterationが2 epochs、198 updateを完遂し、このrun中の
  KL early stopは0回
- 速度: rollout平均135.236秒、最適化平均433.720秒、評価込みiteration平均581.611秒
- 独立評価: 2,000件、seed `20260906`（学習、best選択、過去の独立評価には未使用）
- 継続元best: 平均786,174.941。今回best: 平均788,335.118。同一ケース差は
  +2,160.178 ±2,045.527、今回bestの勝率51.40%、同率0.15%
- 判断: 固定評価bestと独立評価はいずれも改善方向なのでcheckpointは採用する。ただし独立評価差は
  標準誤差と同程度で、事前基準の3,000点未満である。固定評価の後半平均も前半を下回り、bestは
  経過5.338時間のiteration 759から更新されなかったため、現行モデル・学習設定は概ね飽和したと判断する

## 2026-08-24 未来列FiLM・10時間（準備）

- 継続元: `outputs/ahc015/ppo-20260824-095458/best-training.pt`（iteration 759、固定平均795,692.848）
- 変更: 既存の未来列144次元表現から共有の`gamma/beta`各144次元を直接生成する。
  CNN stem直後へ`h' = (1 + gamma) * h + beta`として一度適用し、既存の最終fusionも維持する
- 初期化: `gamma/beta`生成層をゼロ初期化し、継続開始時のactorとcriticを旧checkpointと一致させる。
  既存parameterのAdamW momentも名前対応で引き継ぎ、新規FiLM parameterだけ状態なしから開始する。
  継続元を新runの初期bestにも保存し、FiLM学習が悪化した場合に親checkpointを失わない
- parameter: actor 368,209から409,969（+41,760、+11.34%）。FiLMの効果を分離して確認するため
  低次元bottleneckは設けず、今回の実験では提出サイズを採否基準にしない
- 設定: rollout 1,024、minibatch 1,024、2 epochs、learning rate `3e-4`、entropy係数`0.01`
- rollout seed: 未使用の`15025`、固定評価seed: `515015`
- 時間: 10時間
- config: `examples/ahc015/config_film.toml`
- 判断基準: 同設定の単純継続が独立評価+2,160点で概ね飽和したため、固定評価bestだけでなく、終了後の
  未使用2,000ケースで継続元を明確に上回るかを重視する

## ppo-20260824-230503

- algorithm: PPO
- status: started
- output: `outputs/ahc015/ppo-20260824-230503`
- device: mps (Apple Metal Performance Shaders)
- seed: 15025
- wall-clock limit: 10.000 hours
- W&B: online, run ID `jvqxedd3`
- status: time limit reached
- elapsed: 10.146 hours
- updates: 76037
- best paired gain: 449189.457
- 完走確認: 10.146時間、iteration 760から820まで61 iteration、6,183,936 transitionを追加し、
  累積31,781,376 transition、76,037 optimizer update。MPSとW&B onlineは正常
- 固定512ケースbest: iteration 774（経過2.445時間）、平均797,602.486、継続元bestより+1,909.639。
  12回の固定評価の前半5回平均は788,501.507、後半5回平均は793,949.114で、後半が
  +5,447.607点高い。標準偏差は4,049、rangeは12,560。ただしbest後の9回では更新されなかった
- 更新指標: 平均KL 0.01169、clip fraction 9.45%、学習entropy 0.2707、rollout entropy 0.2737、
  explained variance 0.9853。60 iterationは2 epochs・198 updateを完遂し、KL early stopは初回の
  1 iterationだけ。actorのFiLM weight normは初期0からbest時2.875、終了時6.296へ増加した
- 速度: rollout平均139.795秒、最適化平均444.514秒、評価込みiteration平均597.495秒。
  前runの581.611秒から約2.7%低下したが、10時間実験の運用上は問題ない
- 独立評価: 2,000件、seed `20260908`（学習、best選択、過去の正式な独立評価には未使用）
- 継続元best: 平均789,324.285。FiLM best: 平均791,701.636。同一ケース差は
  +2,377.352 ±2,035.978、FiLM bestの勝率50.75%、同率0.30%
- 提出サイズ: bestの量子化model dataだけで531,707 bytesとなり、Rustコードを加える前に524,288 byte
  制限を超える。今回は効果判定を優先したため想定内
- 判断: 固定評価と独立評価は改善方向だが、独立差は標準誤差と同程度であり、直前の単純継続の
  +2,160 ±2,046点とほぼ同じである。FiLM固有の改善は確認できないものの、絶対スコアは上がっているため
  `outputs/ahc015/ppo-20260824-230503/best.pt`を仮採用する。量子化model dataだけで提出サイズを超えるため、
  現状のまま提出モデルにはできない

## 2026-08-25 rollout 4096（準備）

- 目的: 大型teacherモデルへ移る前に、1回のon-policy rolloutを1,024局から4,096局へ増やした効果を
  現行FiLMモデルで確認する
- 継続元: `outputs/ahc015/ppo-20260824-230503/best-training.pt`（FiLM仮採用best）
- 問題: 旧実装は全候補の18x10x10特徴をfloat32で保持するため、4,096局では特徴bufferだけで
  約10.88 GiBとなる。turnごとの配列を最後にstackする際は同程度の一時コピーも生じ、18 GB機では
  OOMまたはswapが見込まれる。CNN推論も16,384候補の一括処理になり、MPS活性メモリが4倍になる
- メモリ対策: 14番以外の特徴面はすべて1/100刻みなので、100倍したuint8で事前確保bufferへ直接保存する。
  14番のpotential面は既存のfloat32 candidate potentialからminibatchごとに復元する。このためモデル入力の
  意味を変えず、feature bufferを約2.72 GiBへ削減し、最後の巨大stackもなくす
- MPS対策: rollout推論を`inference_batch_size=4096`候補、すなわち1,024局ずつに分割し、従来の
  1,024局runと同じ推論時peak活性に抑える。PPO更新のminibatchも1,024のまま維持する
- 更新回数対策: `2 epochs`のままでは約792 update/iterationとなり、古い方策のrolloutに対して更新しすぎる。
  初回は`1 epoch`として約396 update/iterationに抑え、unique transitionの利用率を上げる
- 事前速度見積もり: FiLM 1,024局runのrollout 139.8秒、2-epoch更新444.5秒から線形外挿すると、
  4,096局・1 epochはrollout約559秒、更新約889秒、評価を除き約24.1分/iteration
- 1分未満microbenchmark: 4,096局の中央turnを1回、PPO更新を4 minibatch、2.72 GiB bufferの全域書き込みを
  seed `15026`と`15027`で測定した。各runの実時間は25.2秒と22.0秒。rollout外挿は514秒と551秒、
  更新外挿は863秒と873秒、合計は1,378秒と1,425秒だった。平均約23.4分/iterationで事前見積もりと整合する。
  buffer全域書き込みは0.151秒と0.113秒で、MPS modelと同時に2.72 GiBを保持してもOOMやswap兆候はなかった
- evaluationを除く10時間見込み: 約25 iteration、約1,014万unique transition、約9,900
  optimizer update。従来10時間runの約618万transition、
  約12,000 updateに対し、データ多様性は増える一方でupdate数は約2割減る
- evaluation: rollout 1,024時代と同等以上のbest選択粒度を保つため毎iteration実施する。実測約66秒を
  用いると時間コストは約4.5%。10時間では学習約24 iteration、約973万unique transition、約9,500
  optimizer update、evaluation約24回を見込む
- config: `examples/ahc015/config_rollout4096.toml`（seed `15026`、10時間、evaluation interval 1）
- benchmark: `examples/ahc015/python/benchmark_training.py`
- status: 実装、test、1分未満の実機microbenchmarkまで完了。完全な1 iterationと学習は未実行

## ppo-20260825-105935

- algorithm: PPO
- status: started
- output: `outputs/ahc015/ppo-20260825-105935`
- device: mps (Apple Metal Performance Shaders)
- seed: 15026
- wall-clock limit: 10.000 hours
- W&B: online, run ID `3o20qjh2`
- status: time limit reached
- elapsed: 10.022 hours
- updates: 76433
- best paired gain: 450311.150
- 完走確認: 10.022時間、iteration 775から798まで24 iteration、9,732,096 transitionと9,504
  optimizer updateを追加した。MPS、W&B online、2.72 GiBの圧縮rollout bufferは正常に動作した
- 固定512ケース: 24回の平均792,977、標準偏差3,231、range 13,103。前半5回平均790,708に対し
  後半5回平均793,342で+2,635点。bestは最後のiteration 798で798,724となり、継続元bestの
  797,602を+1,122点上回った
- 更新指標: 平均KL 0.00934、clip fraction 8.11%、学習entropy 0.2404、rollout entropy 0.2439、
  explained variance 0.9868。全24 iterationが1 epoch・396 updateを完遂し、更新は安定していた
- 速度: rollout平均548.573秒、最適化平均887.899秒、512局evaluation込みiteration平均1,502.864秒
  （25.05分）。evaluationは平均約66.4秒、全体の4.42%。microbenchmarkの23.4分に対し実学習は
  evaluationを除いて23.94分で、見積もり誤差は約2.5%だった
- rollout収集効率: 1,024局FiLM runの725 transition/秒に対し739 transition/秒で約1.9%向上した。
  evaluation込みのunique transition throughputは約170件/秒から270件/秒へ約59%向上した
- 独立評価: 2,000件、seed `20260909`（学習、best選択、過去の独立評価には未使用）
- 継続元best: 平均791,309.966。今回best: 平均797,557.963。同一ケース差は
  +6,247.997 ±2,067.327、今回bestの勝率52.10%、同率0.15%。改善は約3.0標準誤差で明確
- 提出サイズ: bestの量子化binaryは402,138 bytesだが、提出用base93 Rust model dataは532,310 bytesで
  524,288 byte制限を超える。FiLMモデルは引き続きそのままでは提出できない
- 判断: 独立評価で明確に改善し、固定評価bestも最終iterationで更新されたため、rollout 4,096・1 epochを
  採用する。最新float actorを`outputs/ahc015/ppo-20260825-105935/best.pt`へ更新する

## ppo-20260826-000916

- algorithm: PPO
- status: started
- output: `outputs/ahc015/ppo-20260826-000916`
- device: mps (Apple Metal Performance Shaders)
- seed: 15027
- wall-clock limit: 10.000 hours
- W&B: online, run ID `kw5wd7ua`
- status: time limit reached
- elapsed: 12.718 hours
- updates: 1188
- best paired gain: 150311.695

## small-20260827-232147

- algorithm: PPO
- status: started
- output: `outputs/ahc015/small-20260827-232147`
- device: mps (Apple Metal Performance Shaders)
- seed: 15027
- wall-clock limit: 10.000 hours
- W&B: disabled, run ID `mj8m2zdn`
- status: aborted during setup before the first iteration; W&Bを無効化していたためonline設定へ直して再実行した

## small-20260827-232351

- algorithm: PPO
- status: started
- output: `outputs/ahc015/small-20260827-232351`
- device: mps (Apple Metal Performance Shaders)
- seed: 15027
- wall-clock limit: 10.000 hours
- W&B: online, run ID `hohp3ev1`
- status: time limit reached
- elapsed: 10.004 hours
- updates: 67320
- best paired gain: 392249.254
- 目的: afterstate 4入力・12盤面特徴・未来列FiLM・256 channelの構成が過剰かを調べるため、傾斜前盤面を
  1ターン1入力とし、canonical味3面と空きマス1面だけを使う64 channel・10 blockモデルを学習した。
  actorは4方向の残差を同時出力し、解析的な`Phi`は特徴量ではなくlogitのbaselineとしてのみ加えた
- モデル規模: actor 50,628 parameter。従来teacher 986,113 parameterの約5.1%。rollout feature bufferは
  0.151 GiBで、従来の圧縮buffer 2.72 GiBから約94.4%減少した
- 完走確認: iteration 0から169まで170 iteration、68,935,680 transition、67,320 optimizer update。
  全iterationが4,096局・1 epoch・396 updateを完遂し、early stopは初期の1回だけだった
- 固定512ケース: bestはiteration 161の平均740,662、greedy比+392,249、勝率100%。最終iteration 169は
  平均732,747で、best選択が必要な程度の評価変動は残った
- 終盤の更新指標: 最後5 iterationの平均KL 0.00535、clip fraction 4.89%、rollout entropy 0.2191、
  explained variance 0.9774。学習は安定しており、entropyは初期1.38から低下した
- 速度: 全iteration平均でrollout 126.673秒、最適化63.193秒、evaluation 21.907秒、合計211.800秒
  （3.53分）。従来teacherの約25分/iterationより大幅に高速化した

## small-20260828-093536

- algorithm: PPO
- status: started
- output: `outputs/ahc015/small-20260828-093536`
- device: mps (Apple Metal Performance Shaders)
- seed: 15028
- wall-clock limit: 5.000 hours
- W&B: online, run ID `lwi47t81`
- status: time limit reached
- elapsed: 5.054 hours
- updates: 97020
- best paired gain: 401973.766
- 継続条件: 10時間runの`best-training.pt`（iteration 161）から再開し、rolloutの重複を避けるためseedを
  `15028`へ変更。learning rate `3e-4`など他の条件は維持した
- 完走確認: iteration 162から244まで83 iteration、33,656,832 transitionと32,868 optimizer updateを
  追加した。継続前を含む累計は99,348,480 transition、97,020 update
- 固定512ケース: 最初5回平均736,309に対し最後5回平均744,315で+8,006点。bestは最終iteration 244の
  平均750,387、greedy比+401,974、勝率99.80%で、継続元bestを+9,725点更新した
- 更新指標: 平均KL 0.00539、clip fraction 4.88%、explained variance 0.9783。rollout entropyは最初5回の
  0.2068から最後5回の0.1804へ低下したが、性能改善は継続していた
- 速度: rollout平均125.842秒、最適化平均71.404秒、evaluation平均21.865秒、iteration平均219.117秒
  （3.65分）
- 独立評価: 2,000件、seed `20260828`（学習・best選択には未使用）。10時間bestの平均736,034.997に対し
  継続bestは748,286.815。同一ケース差は+12,251.818 ±2,618.700、継続bestの勝率53.35%、同率0.30%。
  改善は約4.7標準誤差で明確
- 判断: 従来teacherの約5.1%のparameterと4種類の素朴な特徴だけでも高い性能に到達し、15時間時点でも
  改善が続いた。一方、従来teacherの固定評価best 798,724とは約48,000点の差があり、容量・afterstate表現・
  未来列・派生特徴のどれが差を生むかは本実験単独では分離できない。単純モデルを有力baselineとして残し、
  次は変更を1要素ずつ戻すablationで寄与を測る

## small-20260828-151420

- algorithm: PPO
- status: started
- output: `outputs/ahc015/small-20260828-151420`
- input mode: afterstate
- device: mps (Apple Metal Performance Shaders)
- seed: 15029
- wall-clock limit: 5.000 hours
- W&B: online, run ID `ze26mqfe`
- status: time limit reached
- elapsed: 5.046 hours
- updates: 15048
- best paired gain: 378466.434
- 目的: pre-tilt版と入力方式だけを比較するため、傾斜後の4候補を共有モデルへ1つずつ入力した。
  canonical味3面＋空きマス1面、64 channel・10 block、未来列なし、`Phi` baselineとPPO設定は揃えた
- モデル規模: actor 50,433 parameter。rolloutは各transitionにつき4候補を保持するためfeature bufferは
  0.604 GiBとなり、pre-tilt版の0.151 GiBの4倍だった
- 完走確認: iteration 0から37まで38 iteration、15,409,152 transition、15,048 optimizer update。
  early stopはなく、全iterationが4,096局・1 epoch・396 updateを完遂した
- 固定512ケース: bestはiteration 33の平均726,879、greedy比+378,466、勝率100%。評価gainの各期間平均は
  最初の12 iterationが260,064、中間13 iterationが337,729、最後13 iterationが369,551で、学習は継続していた
- 更新指標: rollout entropyは最初5 iteration平均1.3454から最後5 iteration平均0.3609へ低下した。
  平均KL 0.00478、clip fraction 5.47%で更新は安定していた
- 速度: rollout平均198.890秒、最適化平均251.561秒、evaluation平均27.407秒、iteration平均477.909秒
  （7.97分）。同じ4,096局でもpre-tilt版の約3.6分より遅いため、比較はiteration数でなくwall-clockを揃える

## small-20260828-230251

- algorithm: PPO
- status: started
- output: `outputs/ahc015/small-20260828-230251`
- input mode: afterstate
- device: mps (Apple Metal Performance Shaders)
- seed: 15030
- wall-clock limit: 10.000 hours
- W&B: online, run ID `0if54qbx`
- status: time limit reached
- elapsed: 10.026 hours
- updates: 43164
- best paired gain: 422974.883
- 継続条件: 5時間runの`best-training.pt`（iteration 33）から再開し、rolloutの重複を避けるためseedを
  `15030`へ変更。learning rate `3e-4`など他の条件は維持した
- 完走確認: iteration 34から108まで75 iteration、30,412,800 transitionと29,700 optimizer updateを
  追加した。継続前を含む累計は44,199,936 transition、43,164 update。early stopはなかった
- 固定512ケース: 最初5回平均725,708（greedy比+377,295）に対し最後5回平均769,024
  （greedy比+420,611）で+43,316点。bestはiteration 107の平均771,388、greedy比+422,975、
  勝率100%で、継続元bestを+44,508点更新した
- 更新指標: 平均KL 0.00436、clip fraction 4.88%、explained variance 0.9727。rollout entropyは最初5回の
  0.3418から最後5回の0.2144へ低下したが、性能改善は継続していた
- 速度: rollout平均197.480秒、最適化平均256.204秒、evaluation平均27.460秒、iteration平均481.166秒
  （8.02分）
- 独立評価: 学習・best選択に未使用の同一2,000件、seed `20260828`で15時間相当のbest同士を比較した。
  pre-tilt版は平均748,286.815（SE 1,891.610）、greedy比+402,740.656だった。afterstate版は
  平均764,943.646（SE 1,815.051）、greedy比+419,397.488で、平均スコアが+16,656.832点高かった
- 判断: 固定評価と独立評価の両方でafterstate版がpre-tilt版を上回った。afterstate版は4候補の評価により
  計算が遅く、同じ約15時間での累計transitionはpre-tilt版99,348,480の約45%だが、それでも高い性能に
  到達した。性能を優先する今後の標準構成にはafterstate入力を採用し、pre-tilt版は高速な比較基準として残す

## 2026-08-29 policy Phi除去・10時間（準備）

- 目的: 特に序盤では候補盤面の`Phi`が近視眼的なpriorまたはノイズになり得るため、afterstate actorを
  `Phi + G`から`G`単独の行動選択へ移行する
- 継続元: `outputs/ahc015/small-20260828-230251/best-training.pt`（iteration 107、固定平均771,387.912）
- schedule: wall-clockの最初3時間でpolicy Phi係数`alpha`を`1`から`0`へ線形減衰し、残り7時間は
  `alpha=0`で学習する。既存checkpointのactor・critic・optimizerはそのまま引き継ぐ
- 報酬: `Phi(S_{t+1}) - Phi(S_t)`のpotential shapingは維持し、行動選択に使う候補Phiだけを除く。
  `alpha=0`到達後は4候補のPhiを計算・保存しない
- checkpoint: anneal中の一時的な高スコアを最終no-policy-Phiモデルとして採用しないよう、best選択は
  `alpha=0`到達後の固定評価に限定する。schedule累計時間と係数を保存し、再開時にも引き継ぐ
- config: `examples/ahc015/config_afterstate_no_phi.toml`
- seed: `15031`、時間: 10時間、device: MPS、W&B: online

## small-20260829-230746

- algorithm: PPO
- status: started
- output: `outputs/ahc015/small-20260829-230746`
- input mode: afterstate
- device: mps (Apple Metal Performance Shaders)
- seed: 15031
- wall-clock limit: 10.000 hours
- policy Phi coefficient: 1 -> 0 over 3 hours
- W&B: online, run ID `zhauzzxu`
- status: time limit reached
- elapsed: 10.110 hours
- updates: 76824
- best paired gain: 430076.818

## small-20260830-091546

- algorithm: PPO
- status: started
- output: `outputs/ahc015/small-20260830-091546`
- input mode: afterstate
- device: mps (Apple Metal Performance Shaders)
- seed: 15032
- wall-clock limit: 5.000 hours
- policy Phi coefficient: 1 -> 0 over 3 hours
- W&B: online, run ID `bo7kgqlq`
- status: time limit reached
- elapsed: 5.115 hours
- updates: 93852
- best paired gain: 431364.128

## 2026-08-30 終端公式スコア報酬・Phi計算省略（準備）

- 方針: policy Phi係数`alpha=0`を採用し、potential shapingも廃止する。連結成分に基づくPhiは各局の
  最終公式スコアを求める1回だけ計算し、中間状態、4方向候補、Phi-greedy固定評価では計算しない
- 継続元: `outputs/ahc015/small-20260830-091546/best-training.pt`（64 channel・10 block、alpha=0）
- 報酬: 98手目まで0、最後のtransitionだけ正規化済み公式スコア`Phi(final_board)`。終端報酬を
  全手へ伝播するため`gamma=1`、`gae_lambda=1`とする
- 評価: 2 iterationごとに固定2,048ケース。Phi-greedyとのpaired gainではなく平均公式スコアで
  bestを選ぶ。同一固定ケースではPhi-greedy平均が定数なので、bestの順位は従来と同じ
- 計算量: rolloutでは1 iterationあたり4,096回、固定評価では2,048回だけPhiを計算する。
  state Phiとcandidate Phiの評価回数をW&Bへ別々に記録し、いずれも0であることを確認できる
- config: `examples/ahc015/config_afterstate_terminal.toml`
- seed: `15033`、初回時間: 5時間、device: MPS、W&B: online

## small-20260830-230316

- algorithm: PPO
- status: started
- output: `outputs/ahc015/small-20260830-230316`
- input mode: afterstate
- device: mps (Apple Metal Performance Shaders)
- seed: 15033
- wall-clock limit: 10.000 hours
- policy Phi coefficient: 0 -> 0 over 0 hours
- reward mode: terminal
- Phi-greedy evaluation: False
- W&B: online, run ID `2a42u6f2`
- status: time limit reached
- elapsed: 10.034 hours
- updates: 131472
- best mean score: 778415.444
- 完走確認: 10.034時間、iteration 237から331まで95 iteration、38,522,880 transitionと37,620 updateを
  追加した。全iterationでpolicy Phi係数は0、state Phiとcandidate Phiの計算回数も0で、各局の
  final Phi 1回だけを計算した
- 固定2,048ケース: 継続元は平均778,415.444。terminal学習後の最高は最初の評価iteration 237の
  776,401.390で、継続元を一度も上回らなかった。最初5評価平均774,666.291に対し最後5評価平均は
  760,545.621まで低下し、最終評価は759,483.312だった。runの`best-training.pt`は学習後モデルではなく、
  開始時にコピーした継続元checkpointである
- 学習指標: explained varianceは切替直後の最初5 iteration平均-1.199から終盤5 iteration平均0.284まで
  回復したが、potential shaping版の約0.98には遠い。rollout entropyも0.196から0.154へ低下し、性能が
  悪化する方策への確信が強まった。early stopは0回で、PPO更新自体は完遂した
- 速度: rollout平均107.492秒、評価平均35.719秒、iteration平均380.160秒。直前のpotential shaping版の
  125.513秒、72.260秒、416.699秒に対し、それぞれ約14.4%、50.6%、8.8%短縮した
- 独立評価: 学習・best選択に未使用の5,000ケース、seed `20260831`。継続元は平均777,387.834、
  terminal学習後の`last.pt`は759,198.258。同一ケース差は-18,189.576 ±1,560.971、terminal版の
  勝率42.76%、同率0.30%で、約11.7標準誤差の明確な悪化だった
- 判断: 公式最終スコア以外のPhi計算省略には成功したが、終端報酬だけでは99行動へのcredit assignmentが
  不十分で、学習するほど性能が低下したため不採用とする。policy Phi係数`alpha=0`は維持し、報酬には
  `Phi(S_{t+1}) - Phi(S_t)`のpotential shapingを残す。今後の基準は
  `outputs/ahc015/small-20260830-091546/best-training.pt`へ戻し、標準configを
  `examples/ahc015/config_afterstate_alpha0.toml`とする

## 2026-08-31 full未来列＋pooled加算・5時間（準備）

- 目的: 64 channelのafterstateモデルで、既知の未来列を欠落なく与えた場合の上積みを調べる
- 継続元: `outputs/ahc015/small-20260830-091546/best-training.pt`（64 channel・10 block、
  policy `alpha=0`、potential shaping）
- 未来列: 各ターンの未配置部分を、盤面と同じdynamic flavor mappingでcanonical化する。3種類×全100位置の
  one-hotを使用し、配置済み位置だけを0にする。要約統計への置換や切り詰めは行わない
- 結合: `3 x 100`を単一の線形層で64次元へ射影し、residual CNNのglobal average pooling後の
  64次元盤面表現へ加算する。actorとcriticに同じ構成を使う
- 初期化・継続: 加算層のweightとbiasを0初期化し、開始時のactor・critic出力を継続元と完全一致させる。
  既存parameterのAdamW stateは名前対応で引き継ぎ、新規加算層だけstateなしから開始する
- rollout: 同一ターンの4候補で未来列は共通なので、`uint8`の未来列を1 transitionにつき1つ保存する
- 比較条件: rollout 4,096局、1 epoch、固定評価2,048ケースを2 iterationごと、時間5時間。
  policy `alpha=0`とpotential shapingは基準から変更しない
- config: `examples/ahc015/config_afterstate_future_add.toml`
- seed: `15035`、device: MPS、W&B: online

## small-20260831-142105

- algorithm: PPO
- status: started
- output: `outputs/ahc015/small-20260831-142105`
- input mode: afterstate
- future mode: full_add
- device: mps (Apple Metal Performance Shaders)
- seed: 15035
- wall-clock limit: 5.000 hours
- policy Phi coefficient: 0 -> 0 over 0 hours
- reward mode: potential_shaping
- Phi-greedy evaluation: False
- W&B: online, run ID `ns6puj4v`
- status: time limit reached
- elapsed: 5.103 hours
- updates: 111672
- best mean score: 783137.628
- 完走確認: 5.103時間、iteration 237から281まで45 iteration、18,247,680 transitionと17,820 updateを
  追加した。固定2,048ケースbestはiteration 279の783,137.628で、継続元778,415.444より
  +4,722.184点。最初5評価平均778,550.264、最後5評価平均780,121.505だった
- 問題: pooled盤面表現`b`と4候補で共通の未来表現`f`を加算した直後が1出力の線形層なので、
  `w(b + f) = wb + wf`となる。`wf`は全候補logitに共通の定数であり、softmaxとargmaxで相殺される。
  実checkpointでも未来入力を変えて選択行動は変化しなかった。criticは未来列を利用できるが、actorは
  行動選択に利用できない
- 判断: スコア上昇は盤面経路の追加学習または未来対応criticの効果と分離できず、未来列の結合実験としては
  無効なので継続しない

## 2026-08-31 full未来列＋stem加算・10時間（準備）

- 継続元: `outputs/ahc015/small-20260830-091546/best-training.pt`。上記pooled加算runからは継続しない
- 結合: `3 x 100`のfull未来列を単一の線形層`300 -> 64`で射影し、CNN stem直後のfeature mapへ
  broadcast加算してから10個の非線形residual blockへ通す。候補盤面と未来列の相互作用を可能にしつつ、
  FiLMのscaleや中間MLPはまだ導入しない
- 初期化: 射影のweightとbiasを0初期化するため、開始時のactor・criticは継続元と完全一致する。
  既存AdamW stateも名前対応で引き継ぐ
- 比較条件: rollout 4,096局、1 epoch、固定評価2,048ケースを2 iterationごと、時間10時間。
  policy `alpha=0`、potential shaping、64 channel・10 blockは変更しない
- config: `examples/ahc015/config_afterstate_future_add.toml`
- seed: `15036`、device: MPS、W&B: online

## small-20260831-230806

- algorithm: PPO
- status: started
- output: `outputs/ahc015/small-20260831-230806`
- input mode: afterstate
- future mode: full_add
- device: mps (Apple Metal Performance Shaders)
- seed: 15036
- wall-clock limit: 10.000 hours
- policy Phi coefficient: 0 -> 0 over 0 hours
- reward mode: potential_shaping
- Phi-greedy evaluation: False
- W&B: online, run ID `bz8nzt5t`
- status: time limit reached
- elapsed: 10.015 hours
- updates: 128304
- best mean score: 780156.013
- 完走確認: 10.015時間、iteration 237から323まで87 iteration、35,225,344 transitionと34,452 updateを
  追加した。explained variance平均0.9800、KL平均0.00693、early stopは0回で学習は安定していた
- 固定2,048ケース: 継続元778,415.444に対しbestはiteration 307（8.175時間）の780,156.013で
  +1,740.569点。44評価中で継続元を上回ったのは4回、最初5評価平均774,565.044、最後5評価平均
  777,013.928だった。開始後に一度悪化し、8時間付近で基準水準まで回復した
- 独立評価: 学習・best選択に未使用の2,000ケース、seed `20260901`。継続元は776,726.156、
  stem加算bestは780,881.024。同一ケース差は+4,154.869 ±2,531.711、stem加算版の勝率50.55%、
  同率0.40%。改善方向だが約1.64標準誤差で、採用確定には弱い
- 判断: actorの未来射影weight normはbest時4.536まで増え、未来経路は学習された。一方で序盤悪化と
  固定評価の不安定さがあり、stemへの直接注入は盤面表現へ干渉している可能性がある。単純継続より、
  既存盤面scoreを残す残差late fusionと比較する

## 2026-09-01 full未来列＋残差late fusion・10時間（準備）

- 継続元: `outputs/ahc015/small-20260830-091546/best-training.pt`。stem加算runからは継続せず、
  同じ未来列なし基準から分岐する
- 結合: 盤面CNNのglobal average pooling後の64次元`b`と、full未来列を`300 -> 64`とReLUで
  符号化した`f`をconcatする。`128 -> 64 -> 1`とReLUの小さなMLPで補正`delta(b, f)`を生成し、
  既存の`output(b)`へ加算する。盤面と未来の非線形な相互作用を持つため、共通logit相殺は起きない
- 初期化: 補正MLPの最終`64 -> 1`をゼロ初期化し、開始時のactor・critic出力を継続元と完全一致させる。
  既存parameterのAdamW stateは名前対応で引き継ぎ、新規parameterだけstateなしから開始する
- 比較条件: 64 channel・10 block、rollout 4,096局、1 epoch、固定評価2,048ケースを2 iterationごと、
  policy `alpha=0`、potential shaping、時間10時間
- config: `examples/ahc015/config_afterstate_future_late.toml`
- seed: `15037`、device: MPS、W&B: online

## small-20260901-103739

- algorithm: PPO
- status: started
- output: `outputs/ahc015/small-20260901-103739`
- input mode: afterstate
- future mode: full_late
- device: mps (Apple Metal Performance Shaders)
- seed: 15037
- wall-clock limit: 10.000 hours
- policy Phi coefficient: 0 -> 0 over 0 hours
- reward mode: potential_shaping
- Phi-greedy evaluation: False
- W&B: online, run ID `zxf2veg1`
- status: time limit reached
- elapsed: 10.116 hours
- updates: 129096
- best mean score: 784952.930

## small-20260901-233428

- algorithm: PPO
- status: started
- output: `outputs/ahc015/small-20260901-233428`
- input mode: afterstate
- future mode: full_late
- device: mps (Apple Metal Performance Shaders)
- seed: 15038
- wall-clock limit: 10.000 hours
- policy Phi coefficient: 0 -> 0 over 0 hours
- reward mode: potential_shaping
- Phi-greedy evaluation: False
- W&B: online, run ID `2rjfla17`
- status: time limit reached
- elapsed: 10.045 hours
- updates: 160776
- best mean score: 788386.353

## small-20260902-102642

- algorithm: PPO
- status: started
- output: `outputs/ahc015/small-20260902-102642`
- input mode: afterstate
- future mode: full_late
- device: mps (Apple Metal Performance Shaders)
- seed: 15039
- wall-clock limit: 10.000 hours
- policy Phi coefficient: 0 -> 0 over 0 hours
- reward mode: potential_shaping
- Phi-greedy evaluation: False
- W&B: online, run ID `8vwhuzf0`
- status: time limit reached
- elapsed: 10.016 hours
- updates: 191664
- best mean score: 790087.528
- 完走確認: 10.016時間、iteration 397から483まで87 iteration。固定2,048ケースbestは
  iteration 457（7.059時間）の790,087.528で、未来列なし基準778,415.444より+11,672.084点。
  最初5評価平均784,603.328、最後5評価平均786,972.077で、3回の10時間runにおけるbestの増分は
  +6,537、+3,433、+1,701点と縮小したため、概ね飽和と判断した

## 2026-09-03 full未来列ablation・5,000ケース

- 対象: `outputs/ahc015/small-20260902-102642/best.pt`（full未来列＋残差late fusion、iteration 457）
- 基準: `outputs/ahc015/small-20260830-091546/best.pt`（未来列なし）
- ケース: 学習・best選択・過去の独立評価に未使用の5,000ケース、seed `20260903`
- 条件: 正しい未来列、未来補正`delta`完全OFF、episode間で未来列を入れ替え、各局・各ターンの
  残数を保って未来順だけをshuffle。5条件をCPU 5 process・各2 threadで並列評価し、約32分を要した
- 未来列なし基準: 778,954.653
- 正しい未来列: 787,086.924。基準差+8,132.270 ±1,513.728、勝率53.78%、同率0.32%
- 補正OFF: 787,450.627。正しい未来列との差+363.703 ±365.903、同率92.82%。点推定では
  正しい未来列より高いが、約0.99標準誤差で有意ではない
- episode shuffle: 787,226.237。正しい未来列との差+139.314 ±180.659、同率98.42%
- 順序shuffle: 787,119.295。正しい未来列との差+32.372 ±183.067、同率98.14%
- 判断: 学習済みlate fusionモデル自体は未来列なし基準を明確に上回るが、補正を無効化した方が点推定で
  さらに高く、正しい未来列は誤った未来列や順序を壊した未来列を上回らなかった。改善の大半は30時間の
  追加学習を受けた共有CNN・既存headによるもので、現在の補正MLPが未来情報を有益に利用した証拠はない。
  full未来列の表現を否定する結果ではなく、盤面だけでも表現できる補正MLPへ依存した結合方式の問題とみる

## 2026-09-02 未来列なし128 channel提出用モデル・10時間（準備）

- 方針: 未来列は採用せず、afterstate、canonical味3面＋空きマス1面、policy `alpha=0`、potential
  shapingを維持したまま、モデル幅だけを64から128 channelへ増やす。residual block数は10のまま
- 初期化元: `outputs/ahc015/small-20260902-102642/best-training.pt`。5,000ケースablationで平均
  787,450.627だった未来補正OFFの盤面経路を、actor・criticとも教師として使う
- 初期化方法: 各64 channelを2つに複製する。各pointwise convolutionと最終出力層では複製入力への
  weightを55%/45%に分配し、和を元のweightに一致させる。初期関数を保ちつつ、複製channelへ異なる
  勾配を流して対称性を崩す。未来encoder・fusion・correctionは移さない
- optimizer: 教師のAdamW stateや学習カウンタは引き継がず、新規に開始する
- 保全: PPO開始前に固定2,048ケースを評価し、初期128 channelモデルもbest checkpoint候補として保存する
- 比較条件: rollout 4,096局、1 epoch、minibatch 1,024、microbatch 128、評価interval 2、10時間、
  MPS、W&B online
- config: `examples/ahc015/config_afterstate_128.toml`
- seed: `15040`

## small-20260902-233926

- algorithm: PPO
- status: started
- output: `outputs/ahc015/small-20260902-233926`
- input mode: afterstate
- future mode: none
- device: mps (Apple Metal Performance Shaders)
- seed: 15040
- wall-clock limit: 10.000 hours
- policy Phi coefficient: 0 -> 0 over 0 hours
- reward mode: potential_shaping
- Phi-greedy evaluation: False
- W&B: online, run ID `o4nxbt9c`
- initialization: function-preserving 2x channel widening from `outputs/ahc015/small-20260902-102642/best-training.pt`; optimizer state reset
- initial widened mean score: 790296.413
- status: time limit reached
- elapsed: 10.132 hours
- updates: 17424
- best mean score: 790296.413

## 2026-09-03 旧144 channelモデルの未来列ablation・2,000ケース

- 目的: 64 channel化以前のモデルが、入力していた未来列を実際に有益な情報として利用していたか確認する
- 対象: `outputs/ahc015/ppo-20260825-105935/best.pt`（144 channel・9 block、iteration 798）。
  当時の15盤面特徴、左詰めfull未来列、`300 -> 144 -> 144`の未来MLP、stem直後のFiLM、盤面と
  未来表現の非線形fusionを再現した
- ケース: 学習・best選択・過去の独立評価に未使用の2,000ケース、seed `20260910`
- 条件: 正しい未来列、全ゼロ入力、episode間shuffle、各局・各ターンの残数を保つ順序shuffle。
  同一ケースをMPSで逐次評価し、各条件約111.4～111.8秒、合計約7分27秒だった
- 正しい未来列: 平均795,978.756
- episode shuffle: 平均795,826.552。正しい未来列との差-152.204 ±1,244.870、勝率27.40%、
  同点率44.30%、軌跡上の行動一致率84.64%
- 順序shuffle: 平均795,679.869。正しい未来列との差-298.887 ±1,227.743、勝率26.50%、
  同点率45.20%、軌跡上の行動一致率84.21%
- 全ゼロ入力: 平均794,672.343。正しい未来列との差-1,306.413 ±1,373.679、勝率30.45%、
  同点率37.15%、軌跡上の行動一致率80.98%
- 解釈: 入力を変えると軌跡上の行動は変化するため、未来経路が計算上完全に無視されていたわけではない。
  ただし行動一致率は、最初の行動差から盤面自体が分岐した後も含むため、同一盤面での直接感度ではない
- 判断: 正しい未来列はepisode shuffleや順序shuffleを上回らず、全ゼロとの差も約0.95標準誤差に
  とどまった。旧大型モデルでも正しい未来順をスコア改善へ利用した証拠はなく、現在の128 channel実験では
  未来列なしを維持する。数百点程度の小さな効果までは否定しない

## 2026-09-03 未来列なし128 channel・ランダム初期化＋方策蒸留（準備）

- 方針: 今後、特別な指定がない限り未来列を入力特徴へ含めない
- student: afterstate、canonical味3面＋空きマス1面、128 channel・10 block。完全ランダム初期化
  （既存モデル実装どおりactor/critic最終headのみ0初期化）し、wideningやoptimizer stateは使わない
- teacher: `small-20260902-102642/best-training.pt`の64 channel actorからfuture encoder・fusion・
  correctionを捨てた盤面経路（iteration 457、5,000ケースの補正OFF平均787,450.627）。teacherにも
  future入力を渡さず、criticは教師にしない
- 補助損失: 4候補に対する`KL(teacher || student)`。係数を実時間3時間で`1 -> 0`へ線形減衰する
- PPO: potential shapingを維持し、policy Phi係数は`alpha=0`。学習率`1e-4`、entropy係数`0.003`
- 実行: Kaggle T4 x2。rollout・評価はGPU別の2 process、更新は2-process DDP/NCCL。rollout
  4,096局、minibatch 1,024、microbatch 512、固定評価
  2,048ケース/2 iteration、W&B online、seed `15041`、3時間
- Notebook: `examples/ahc015/ahc015-128-distillation.ipynb`
- W&B run名: `distill-<時刻>`
- 次段階: このrunの`best-training.pt`から蒸留係数0で17時間PPOを継続し、状況確認後にbestから
  さらに10時間継続する
- KaggleのPyTorch 2.10.0+cu128では単一process・multi-threadのDataParallel rolloutがCUDA
  illegal memory accessで2回失敗した（run `9fjtihka`、`yw8h7le6`。ともに更新前）。pre-tilt時代と
  同様にrollout・評価をGPU別processへ分離し、更新はPyTorch推奨のDDPへ置き換える

## 2026-09-03 未来列なし128 channel・ランダム初期化＋方策蒸留（結果）

- W&B: `distill-20260903-045216`（run ID `v1fy4rlz`）、正常終了
- 実行時間: 3.084時間、24 iteration、9,732,096 environment transitions
- 固定2,048ケース評価: 677,112.189（iteration 1）から768,329.384（iteration 23）まで改善。
  最終評価がrun内best、標準誤差1,935.455
- 蒸留KL: 0.727から0.143へ低下。蒸留係数は最後のiteration開始時点で0.0167まで減衰した。
  2.95時間時点で開始したiterationが3時間をまたいだため厳密な0ではないが、次runでは設定値を0にする
- 実装確認: rollout・評価はGPU別2 process、更新は2-process DDP/NCCLで完走。actor用`best.pt`と
  学習再開用`best-training.pt`をW&B artifact `v0`として保存済み
- 判断: 3時間の初期化段階として学習は正常。教師の盤面のみ平均787,450.627には未到達だが、予定どおり
  best checkpointから蒸留なしPPOへ移行する

## 2026-09-03 未来列なし128 channel・蒸留なしPPO 5時間（準備）

- 開始点: W&B run `distill-20260903-045216`のtraining checkpoint `v0`、iteration 23、
  固定評価768,329.384
- 条件: afterstate、未来列なし、128 channel・10 block、potential shaping、policy `alpha=0`、
  蒸留係数0、学習率`3e-4`、entropy係数`0.01`。蒸留中の`1e-4`・`0.003`から、学習初期の探索を
  重視した通常PPO設定へ戻す。飽和が近づいた後にentropy係数の低下を再検討する
- 実行: Kaggle T4 x2、rollout・評価はGPU別2 process、更新は2-process DDP/NCCL、rollout 4,096局、
  固定評価2,048ケース/2 iteration、seed `15042`、5時間
- 設定: `examples/ahc015/config_afterstate_128_continue.toml`
- Notebook: `examples/ahc015/ahc015-128-ppo.ipynb`。本番前に同じresume経路を8局・1 iterationで検証する
- W&B run名: `large-<時刻>`
- microbatch benchmark: `ahc015-128-microbatch-benchmark.ipynb`でglobal 256/512/1,024
  （GPUあたり128/256/512）を同一256-episode rollout・各2回比較した。更新25回の中央値は順に
  21.860秒、21.716秒、21.459秒。1,024は512比1.012倍だったが、条件内の反復差が最大1.610秒あり、
  条件間の最大差0.400秒を上回った
- microbatch判断: 1,024がT4 x2のメモリに収まることは確認できたが、速度差は測定誤差内で追加実験の
  価値は低い。メモリ余裕を残すglobal 512（GPUあたり256）を5時間学習に採用する

## 2026-09-03 未来列なし128 channel・蒸留なしPPO 5時間（結果）

- W&B: `large-20260903-083526`（run ID `bbsoay8q`）、正常終了
- 実行時間: 5.084時間、epoch 23から63、追加16,220,160 environment transitions、15,840 update
- 固定2,048ケース評価: 最初の更新後に754,845.812、最低742,215.057まで低下した後、終盤は
  761,000前後まで回復。最終761,782.361 ±1,902.726で、開始点のrun内best 768,329.384は更新しなかった
- PPO状態: 初期のapproximate KL 0.0223から最終0.00749、clip fraction 0.1234から0.0645へ安定。
  explained varianceは最終0.9752で、数値的な発散はない
- checkpoint: 終了時点の128 channel actor・critic・optimizerを含む`last.pt`（epoch 63、SHA-256
  `aca6a0ee2b87ab7f22f13bda97db745b556dc3e82092fa309d3d37ad8314c67e`）を回収し、W&B artifact
  `large-20260903-083526-last-training-checkpoint:v0`へ補完uploadした
- 次段階: 上記lastから、学習率`3e-4`・entropy係数`0.01`などを維持し、rollout seedだけ`15043`へ
  変更して10時間継続する。Notebookは`ahc015-128-ppo.ipynb`
- 保存改善: 今後はrun終了時にbest actor、best training checkpointに加えてlast training checkpointも
  W&B artifactへ自動uploadし、`last.pt`内部にも終了時の固定評価値を保存する

## 2026-09-04 未来列なし128 channel・蒸留なしPPO 10時間（結果）

- W&B: `large-20260903-141739`（run ID `bvok9tnw`）、正常終了
- 実行時間: 10.048時間、epoch 63から145、追加33,251,328 environment transitions、32,472 update
- 固定2,048ケース評価: 開始時761,782.361から、序盤は755,721.217まで低下したが、その後回復した。
  6.87時間で773,502.399、7.36時間で774,921.207となり、最終評価がrun内bestの
  779,071.038 ±1,761.048だった。開始点からの改善は+17,288.677
- PPO状態: 最終approximate KL 0.00615、clip fraction 0.0539、explained variance 0.9797。
  数値的な発散はなく、最終評価でbestを更新しているため、追加学習を止める根拠はまだない
- checkpoint: `last.pt`はepoch 145、optimizer state 88 parameter分を保持し、SHA-256は
  `4f7bc5520d4eed7dddb6b970f8c5c6edeb00fe151fe0979d99b9016b90913c36`。W&B artifact
  `large-20260903-141739-last-training-checkpoint:v0`へupload済み。最終評価がbestのため、bestとlastは
  同じepochのモデル状態である
- 次段階: 上記lastから学習率`3e-4`・entropy係数`0.01`などを維持し、rollout seedを`15044`へ
  変更して11時間30分継続する。Notebookは`ahc015-128-ppo.ipynb`

## 2026-09-05 未来列なし128 channel・蒸留なしPPO 11時間30分（結果）

- W&B: `large-20260904-005451`（run ID `uqqgn8q4`）、正常終了
- 実行時間: 11.540時間、epoch 145から233、追加35,684,352 environment transitions、34,848 update
- 固定2,048ケース評価: 開始checkpointの779,071.038に対し、序盤の最低値は775,078.431。
  終盤は概ね780,000前後まで回復し、epoch 231でrun内bestの784,700.457 ±1,815.212、最終epoch 233で
  782,029.730 ±1,820.645だった。開始点に対してbestは+5,629.418、lastは+2,958.692
- PPO状態: 最終approximate KL 0.00624、clip fraction 0.0531、explained variance 0.9818、
  rollout entropy 0.1870。数値的な発散はなく、改善幅は縮小しているが追加学習は可能と判断する
- checkpoint: W&Bのbest training checkpointはepoch 231。継続に用いる`last.pt`はepoch 233、
  optimizer state 88 parameter分を保持し、SHA-256は
  `df2a8f4a48e1b0fd61548fcb635d75908b078e9253e77f73059b453d088e2fda`。artifactは
  `large-20260904-005451-last-training-checkpoint:v0`
- 次段階: 上記lastから学習率`3e-4`・entropy係数`0.01`などを維持し、rollout seedを`15045`へ
  変更して10時間継続する。Notebookは`ahc015-128-ppo.ipynb`
