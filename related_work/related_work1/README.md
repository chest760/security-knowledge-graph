# Embedding and Predicting Software Security Entity Relationships: A Knowledge Graph Based Approach
https://link.springer.com/chapter/10.1007/978-3-030-36718-3_5#Tab3


## 1 CAPEC, CWE, CVEの単語埋め込みを追加学習

## 2 Word2Vecで得られた単語ベクトルをCNNを通して、文章ベクトルを獲得する
入力層 + 畳み込み層 * 2 + プーリング層 * 2 \
最初のプーリング層は max pooling \
2番目のプーリング層は　mean pooling \
2-gram \
スライディングウィンドウ: 100

## 3 TransHモデルによるナレッジグラフ埋め込み

### $$ f{r}(h,t) = ||\boldsymbol{h}_{\perp} + \boldsymbol{r} - \boldsymbol{t}_{\perp}|| $$
### $$Basic:  f_{r}(h,t) = f^{ss}_{r}(h,t) + f^{dd}_{r}(h,t)$$ 
### $$ f^{ss}_{r}(h,t) = ||\boldsymbol{h}_{s} + \boldsymbol{r} - \boldsymbol{t}_{s}|| $$
### $$ f^{dd}_{r}(h,t) = ||\boldsymbol{h}_{d} + \boldsymbol{r} - \boldsymbol{t}_{d}|| $$

### $$Enhance:  f_{r}(h,t) = f^{ss}_{r}(h,t) + f^{dd}_{r}(h,t) + f^{sd}_{r}(h,t) + f^{ds}_{r}(h,t)$$ 
### $$ f^{sd}_{r}(h,t) = ||\boldsymbol{h}_{s} + \boldsymbol{r} - \boldsymbol{t}_{d}|| $$
### $$ f^{ds}_{r}(h,t) = ||\boldsymbol{h}_{d} + \boldsymbol{r} - \boldsymbol{t}_{s}|| $$

## 4 損失関数

### $$ L =\sum_{<h,r,t,>\in S} \sum_{<h',r',t',>\in S'} [\gamma +f_{r}(h,t) - f_{r'}(h,t)] $$