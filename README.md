# AEC

[Paper URL](https://arxiv.org/pdf/2010.14337.pdf) [GitHub](https://github.com/breizhn/DTLN-aec)

[DTLN GitHub URL](https://github.com/breizhn/DTLN)

项目中的 DTLN AEC Model 是参考 Parper URL 中的论文与 DTLN 源码复现的

## Repitation Nots

1. ConCat 应该在 Channel 的维度上进行计算，而不是在 Time 维度上进行计算, 所以 `tf.concat` 的第二个参数应该是 `0`
2. 在第二个分离核心中应该输入的是加窗之后的信号数据

## Prepare Data


## Results
