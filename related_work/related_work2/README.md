# Predicting Entity Relations across Different Security Databases by Using Graph Attention Network

## モデルアーキテクチャ

$ G = (E, R, S) $:  $E$, $R$, $S$ represent Entity, Relation and Triples 

$S = <h, r, t>$

$r$ : ChildOf, ParentOf, CanPrecede, CanFollow, PeerOf, Semantic, BelongOf, AttackOf, TragetOf

### Structure-Embedding Generation.
TransEを使用して、グラフ構造Embeddingを取得する. \
次元は100

### Description-Embedding Generation
CAPEC, CVE, CWEの記述のEmbeddingを取得する \
次元は100

上記の二つのEmbeddingを結合する
### Graph attention Layerと線形層によって結合ベクトルを獲得する。
2hopの隣接ノードを使用する