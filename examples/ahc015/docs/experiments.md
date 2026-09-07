# AHC015 実験の要点

## 採用した構成

- 方策は未来列なしの afterstate PPO、128 channel・10 residual block、potential shaping、policy `alpha=0`。
- 128 channel student はランダム初期化し、64 channel の盤面-only教師から actor 方策を3時間蒸留して開始する。
  critic は蒸留しない。以後は蒸留なし PPO を継続する。
- Kaggle T4 x2 では rollout・評価を GPU 別 process、更新を 2-process DDP/NCCL とする。global microbatch
  512を採用した。256/512/1,024 の速度差は測定誤差内だったため、512でメモリ余裕を確保する。
- PPO 継続時の学習率は `2.5e-4`。学習状況を見て減衰させる。

## 比較から得た知見

- 同条件の約15時間比較で afterstate は pre-tilt より独立2,000ケース平均で16,657点高く、afterstate を採用した。
- 未来列の正しい順序は、未来列を無効化・shuffleした条件を明確に上回らなかった。未来列は入力しない。
- 方策から Phi を除いても、potential shaping 報酬は必要だった。終端報酬だけの学習は基準を下回った。
- 128 channel の PPO 継続では、固定2,048ケース評価の最高値は 796,200.133（epoch 371）。終盤は改善幅が
  小さいため、`best-training.pt` と `last-training-checkpoint` の双方を保存し、評価を見て継続可否を判断する。

## 提出用探索

- 6手の厳密 expectimax は、未使用ケースで探索なしより大幅に改善し、個別レイテンシも2秒制限内だった。
  7手は tail latency が危険なため使わない。
- 88〜93手目では、初手4方向・24ルール・各128 sampleのモンテカルロ探索を使う。6手厳密探索との未使用
  1,000ケース比較で平均 +5,319点だった。
- 固定 sample 上限に加え、残時間を残りのモンテカルロ手数で割る deadline で打ち切る。
- architecture非依存の高速化として、共通rank列、24ルールの全行動列、初回配置後盤面を再利用し、同じ
  suffixのルールを統合した。終端連結度は`u128` bitboard flood fillへ置換した。ランダム1,000盤面で
  旧DFSと一致し、固定100ケースのスコア配列SHA-256も変更前後で一致した。
- 同一300ケース・512 sampleの10並列ジャッジ相当時間は1.339秒から1.139秒へ14.9%短縮した。
  一方、未使用1,000ケースで512 sampleは128 sampleより平均525点低く、計算量増加による改善はなかった。
  128 sampleを維持し、現時点ではAVX専用実装を追加しない。
