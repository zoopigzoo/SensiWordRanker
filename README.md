# SensiWordRanker

rank sensitive words given surrounding texts

基于周围的上下文从指定词的列表中为词排序。

## 用法

```python
python3 sensiWordRanker.py --wordfile sensi.txt --device mps --input "陈风一刻也不想在这地底洞**久呆，领着罗妮，迅速从通道中钻出，随后往明泽城的方向急飞而去。"
```

执行完成后会返回按照rank排序的词表

```
穴里 [4.555016040802002]
床上 [4.840875625610352]
裙下 [4.868013858795166]
穴 [4.879256725311279]
胯下 [4.8985209465026855]
```

```python
python3 sensiWordRanker.py --wordfile sensi.txt --device mps --input "远处深山的轮廓显得格*暗，周边大地上的植被花草渐渐的沉入地面"

output: 

外阴 [3.564953327178955]
外流 [4.027756214141846]
色色 [4.237478733062744]
色逼 [4.318686485290527]
调教 [4.329565525054932]

```

## 参数

--input: 输入的字符串，中间用\*表示需要恢复的词。仅支持一个\*。

--wordfile: 输入词表，每行一个词。

--modelpath: GPT2模型path或名称，默认用的uer/gpt2-chinese-cluecorpussmall。

--device: cpu/mps/cuda/cuda:0...，默认cpu。

--batchsize: 默认16。

--contextlen: inference时候使用context的长度，越长越准确，但是会慢。默认20。

--ppltklen: 候选词最长考虑的token长度，由于不等长建议选2。

## 依赖

transformers, torch

## 原理

利用GPT2模型对句子合理性的估计（ppl），将候选词填入，ppl越低的排名越高。

## 优缺点

优点：词的列表很好扩充。

缺点：慢（由于每个词都要做，所以跟词的列表数据成正比）。

