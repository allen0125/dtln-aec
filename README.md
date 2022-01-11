# AEC

[Paper URL](https://arxiv.org/pdf/2010.14337.pdf) [GitHub](https://github.com/breizhn/DTLN-aec)

[DTLN GitHub URL](https://github.com/breizhn/DTLN)

项目中的 DTLN AEC Model 是参考 Parper URL 中的论文与 DTLN 源码复现的

## Repitation Nots

1. ConCat 应该在 Channel 的维度上进行计算，而不是在 Time 维度上进行计算, 所以 `tf.concat` 的第二个参数应该是 `0`
2. 在第二个分离核心中应该输入的是加窗之后的信号数据

## Prepare Data

- 数据格式转换: 将所有数据转换为 16Khz Mono Wav 格式的音频文件.
- 完成回声音频文件生成, 同样为 16Khz Mono Wav 格式的音频文件.
- 完成干声音频文件与回声音频文件对应.
- 完成 Mixed 音频文件生成. (AudioSegment overlay 即可, 还是需要保持是 Mono).
    - 不同使用不同响度的回声音频文件.
    - 使用不同响度的干声音频文件.
- 完成 Tensorflow Dataset.
- 切分训练集, 切分验证集.
- 开始模型训练.

## Validation


## Results


## Some ideas

- 经过 DTLN-AEC 网络处理的干声是否能自动进行响度泛化? 将训练集中的干声全都泛化到同一个响度, 输入的包含回声及噪音的干声泛化到不同的响度.
- 用户干声先使用 fine-tuning 后的 DTLN Model 降噪之后再用, 保证目标输出音轨的 '纯净度' 效果会不会好一些?