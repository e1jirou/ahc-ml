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
