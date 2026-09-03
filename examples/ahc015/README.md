# AHC015

特徴量・モデル容量の過剰性を調べるための小型PPO実験である。入力はcanonical味3面と空きマス1面の
`(4, 10, 10)`だけ、モデルは64 channel・10 residual blockとし、未来列や派生特徴は使わない。
味の置換と盤面の回転・反転は正規化する。傾斜前盤面を1回入力するpre-tilt版と、傾斜後の4候補を
共有モデルへ1つずつ入力するafterstate版を約15時間ずつ比較した結果、性能の高かったafterstate版を
採用構成とする。pre-tilt版は入力方式の比較基準として残す。

## ローカル学習

`config.toml`はpre-tilt比較基準、`config_afterstate.toml`は採用したafterstate版をMPSでゼロから学習する
設定である。rollout 4,096局、PPO 1 epoch、minibatch 1,024、
AdamW learning rate `3e-4`を使用する。GPUメモリ上では128件ずつ処理して勾配を蓄積し、1,024件ごとに
1回だけoptimizerを更新する。macOSでスリープを防ぎながら10時間学習するコマンドは次のとおり。

```bash
caffeinate -i env PYTHONPATH=python .venv/bin/python \
  -m examples.ahc015.python.train \
  --config examples/ahc015/config.toml \
  --device mps \
  --wandb-mode online \
  --max-hours 10
```

ゼロから始めるときは`--resume`を付けない。`best.pt`はactor単体、`best-training.pt`はbest時点の
actor・critic・optimizerを含む継続学習用checkpoint、`last.pt`は終了時点の完全な学習状態である。
W&Bを有効にしたrunでは、終了時にbestだけでなく`last.pt`もlast training checkpoint artifactとして
uploadする。
wall-clock上限に達したiterationと最後の評価・checkpoint保存は完了させるため、実行時間は10時間を少し超える
場合がある。

## 評価

checkpoint内のchannel数とblock数は自動判定される。

```bash
PYTHONPATH=python .venv/bin/python -m examples.ahc015.python.evaluate \
  --checkpoint outputs/ahc015/<run-name>/best.pt \
  --episodes 2000 \
  --seed 20260826 \
  --device mps
```

この実験はPython学習・評価までを対象とし、既存のafterstateモデル用Rust推論器へのexportは行わない。

## 採用構成: Afterstate

`config_afterstate.toml`は、傾斜後の4候補を共有モデルへ1つずつ入力する設定である。pre-tilt版と
同じcanonical味3面＋空きマス1面、64 channel・10 blockを使い、未来列や派生特徴は入力しない。
約15時間ずつの比較では、未使用の同一2,000ケースでpre-tilt版の平均748,286.8に対しafterstate版は
764,943.6となり、afterstate版が16,656.8点上回った。ローカルMPS学習は次のコマンドで実行する。

```bash
caffeinate -i env PYTHONPATH=python .venv/bin/python \
  -m examples.ahc015.python.train \
  --config examples/ahc015/config_afterstate.toml
```

W&Bのrun名は`small-<時刻>`となり、run configの`model.input_mode`で入力方式を識別できる。

## Policy Phi除去実験

`config_afterstate_no_phi.toml`は、既存afterstate checkpointから10時間継続し、行動logitに加える
`Phi`の係数を最初の3時間で`1`から`0`へ線形に下げ、残り7時間を`Phi`なしで学習する設定である。
PPOの密な報酬`Phi(S_{t+1}) - Phi(S_t)`は維持する。係数が`0`に到達すると、4方向の候補Phiは
rollout・PPO更新・learned policy評価で計算も保存もしない。`best.pt`と`best-training.pt`は
係数が最終値`0`に到達した評価だけから選ぶ。

```bash
caffeinate -i env PYTHONPATH=python .venv/bin/python \
  -m examples.ahc015.python.train \
  --config examples/ahc015/config_afterstate_no_phi.toml \
  --resume outputs/ahc015/<parent-run>/best-training.pt
```

checkpointには係数とanneal累計時間を保存するため、中断後の再開では`1`からやり直さず、保存時点から
scheduleを継続する。独立評価もcheckpointに保存された係数を自動的に使用する。

今後の基準は`config_afterstate_alpha0.toml`である。方策は`alpha=0`を固定し、報酬には従来の
`Phi(S_{t+1}) - Phi(S_t)`を残す。固定評価では不要なPhi-greedyを実行せず、同じ2,048ケースの
平均公式スコアでbestを選ぶ。基準checkpointは
`outputs/ahc015/small-20260830-091546/best-training.pt`である。

```bash
caffeinate -i env PYTHONPATH=python .venv/bin/python \
  -m examples.ahc015.python.train \
  --config examples/ahc015/config_afterstate_alpha0.toml \
  --resume outputs/ahc015/small-20260830-091546/best-training.pt
```

`config_afterstate_terminal.toml`はpotential shapingも除いた不採用ablationを再現するために残す。
10時間学習後は未使用5,000ケースで基準より18,189.576点低く、学習するほど固定評価も低下したため、
このrunの学習後checkpointは採用しない。

full未来列＋加算の実験には`config_afterstate_future_add.toml`を使う。未配置のcanonical味列を
`(3, 100)`のone-hotで欠落なく入力し、線形射影した64次元をCNN stem直後へbroadcast加算する。
後続の非線形residual blockにより、未来列と各候補盤面の相互作用を表現できる。
加算層はゼロ初期化されるため、上記基準checkpointからの開始時点では方策と価値が一致する。

full未来列＋残差late fusionの実験には`config_afterstate_future_late.toml`を使う。盤面CNNの
pooled 64次元と`300 -> 64`で符号化した未来列をconcatし、`128 -> 64 -> 1`のMLPが既存scoreへの
補正値だけを生成する。補正の最終層はゼロ初期化され、基準の盤面経路を開始時に変更しない。

## 128 channel提出用モデル

`config_afterstate_128.toml`は、未来列なしafterstateモデルを128 channel・10 blockで10時間学習する
設定である。初期値には最新の64 channel late-fusion checkpointから未来補正を除いた盤面経路を使い、
各channelを2つに複製する。pointwise convolutionと出力層の入力重みを55%/45%に分配するため、開始時の
actor・critic関数を保ちながら、最初の更新から複製channelの対称性を崩せる。AdamWの状態と学習回数は
引き継がず、新しいrunとして学習する。

```bash
caffeinate -i env PYTHONPATH=python .venv/bin/python \
  -m examples.ahc015.python.train \
  --config examples/ahc015/config_afterstate_128.toml \
  --initialize-from outputs/ahc015/small-20260902-102642/best-training.pt
```

開始時に固定2,048ケースを評価して`best.pt`を保存するため、PPO更新によって一時的に性能が落ちても
64 channel教師相当の初期方策は失われない。`--initialize-from`と、学習状態を丸ごと再開する
`--resume`は同時には指定できない。

### ランダム初期化＋方策蒸留

今後は特別な指定がない限り未来列を入力しない。提出用128 channelモデルを0から作る第一段階には
`ahc015-128-distillation.ipynb`を使う。studentはランダム初期化し、64 channel・未来列なし教師の
actor分布に対する`KL(teacher || student)`だけを補助損失へ加える。criticは蒸留しない。係数は
実時間3時間で`1`から`0`へ線形減衰し、PPOのpotential shapingとpolicy `alpha=0`は維持する。

NotebookはKaggle T4 x2、W&B online用であり、`config_afterstate_128_distill.toml`を実行する。
rolloutと評価は2つの独立processへ半分ずつ分配し、PPO＋蒸留更新はNCCLによる2-process DDPで
勾配を同期する。本番runの前に8局・1 iterationの同じmulti-GPU経路をW&B disabledで検証する。
この段階のW&B run名と出力ディレクトリ名は`distill-<時刻>`とする。
教師checkpointはW&B run `small-20260902-102642`の64 channel bestからfuture encoder・fusion・
correctionを除いた盤面経路である。future入力は生成も使用もしない。この経路の5,000ケースablation
平均は787,450.627だった。3時間終了後の`best-training.pt`を次の蒸留なし17時間PPOの開始点に使う。

### 蒸留後のPPO継続

最初の蒸留なしPPOはW&B run `distill-20260903-045216`のbestから5時間実行し、run
`large-20260903-083526`の終了時点（epoch 63）まで進めた。`ahc015-128-ppo.ipynb`は、このrunの
`last-training-checkpoint:v0`からさらに10時間継続する。future modeは`none`のまま、potential shaping、
policy `alpha=0`、学習率`3e-4`、entropy係数`0.01`を維持する。Kaggle T4 x2ではrollout・評価を
GPU別process、PPO更新を2-process DDP/NCCLで実行し、本番前に同じresume経路を短いsmoke testで
確認する。設定は`config_afterstate_128_continue.toml`、seedは`15043`、W&B run名は`large-<時刻>`とする。
本学習前のmicrobatch比較には`ahc015-128-microbatch-benchmark.ipynb`を使う。同一の128 channel
checkpointとrolloutに対しglobal microbatch 256、512、1,024をT4 x2 DDPで測定し、学習runは開始しない。
実測中央値はそれぞれ21.860秒、21.716秒、21.459秒で、条件内の反復差より小さかった。明確な高速化は
確認できず、GPUあたり256となるglobal microbatch 512を維持する。

## 旧モデルの未来列ablation

64 channel化以前の144 channel・9 blockモデルが未来列を実際に利用していたかは、当時の15盤面特徴、
左詰め未来列、未来MLP、FiLM、非線形fusionを再現した専用CLIで確認する。正しい未来列、ゼロ入力、
episode間shuffle、残数を保った順序shuffleを、同一ケース上でMPSにより逐次評価する。

```bash
PYTHONPATH=python .venv/bin/python \
  -m examples.ahc015.python.evaluate_legacy_future_ablation \
  --checkpoint outputs/ahc015/ppo-20260825-105935/best.pt \
  --episodes 2000 \
  --seed 20260910 \
  --device mps
```

## 検証

```bash
PYTHONPATH=python .venv/bin/python -m pytest python/tests
cargo test --workspace
```
