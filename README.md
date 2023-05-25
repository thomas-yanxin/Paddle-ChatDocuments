# ChatDocuments with Paddle

本项目聚焦于PaddlePaddle生态, 利用飞桨生态内的技术实现 `LangChain+ChatGLM: 基于本地知识库实现自动问答` 的效果, 避免依赖的过度繁杂冗余.

## 🔊技术原理

多路召回是指采用不同的策略、特征或者简单的模型, 分别召回一部分候选集合, 然后把这些候选集混合在一起供后续的排序模型进行重排, 也可以定制自己的重排序的规则等等. 本项目使用关键字和语义检索两路召回的检索系统, 系统的架构如下, 用户输入的Query会分别通过关键字召回BMRetriever(Okapi BM 25算法, Elasticsearch默认使用的相关度评分算法, 是基于词频和文档频率和文档长度相关性来计算相关度), 语义向量检索召回DenseRetriever(使用RocketQA抽取向量, 然后比较向量之间相似度)后得到候选集, 然后通过JoinResults进行结果聚合, 最后通过通用的Ranker模块得到重排序的结果返回给用户.

## 🚀 使用方式

详情请见: [部署文档](./docs/deploy.md)

## 💪 更新日志

详情请见: [更新日志](./docs/update_history.md)

项目处于初期阶段, 有很多可以做的地方和优化的空间, 欢迎感兴趣的社区大佬们一起加入!

## ❤️ 引用

<details><summary><b>ChatGLM论文引用</b></summary>

```
@inproceedings{
  zeng2023glm-130b,
  title={{GLM}-130B: An Open Bilingual Pre-trained Model},
  author={Aohan Zeng and Xiao Liu and Zhengxiao Du and Zihan Wang and Hanyu Lai and Ming Ding and Zhuoyi Yang and Yifan Xu and Wendi Zheng and Xiao Xia and Weng Lam Tam and Zixuan Ma and Yufei Xue and Jidong Zhai and Wenguang Chen and Zhiyuan Liu and Peng Zhang and Yuxiao Dong and Jie Tang},
  booktitle={The Eleventh International Conference on Learning Representations (ICLR)},
  year={2023},
  url={https://openreview.net/forum?id=-Aw0rrrPUF}
}
```

```
@inproceedings{du2022glm,
  title={GLM: General Language Model Pretraining with Autoregressive Blank Infilling},
  author={Du, Zhengxiao and Qian, Yujie and Liu, Xiao and Ding, Ming and Qiu, Jiezhong and Yang, Zhilin and Tang, Jie},
  booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={320--335},
  year={2022}
}
```

</details>

```
@misc{=paddlenlp,
    title={PaddleNLP: An Easy-to-use and High Performance NLP Library},
    author={PaddleNLP Contributors},
    howpublished = {\url{https://github.com/PaddlePaddle/PaddleNLP}},
    year={2021}
}
```
