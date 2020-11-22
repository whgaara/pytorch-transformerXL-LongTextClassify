# 复现：TransformerXL
- 更清晰、直观的torch版TransformerXL
- csdn：https://blog.csdn.net/BmwGaara
- 知乎：https://zhuanlan.zhihu.com/c_1264223811643252736

## 说明
transformerxl模型复杂程度较bert还是有一定提升的，网上相关的源码功能较杂，调试也不是很便利，本人基于论文复现了该网络，能力有限，如右不足之处烦请各位看官记事批评指正。
如果对您有帮助，麻烦给我点一个星星，非常感谢。

## 分类的模型使用
本模型没有设置预训练模型的加载模块：
- 第一步，将训练文本添加到data/src_data中，文本内容就是一行行的句子即可。
- 第二步，进入train_module，运行stp1_gen_train_test.py生成对应的训练和测试集。
- 第三步，打开根目录的pretrain_config.py设置你需要的参数。
- 第四步，修改好参数后，即可运行python3 step2_pretrain_mlm.py来训练了，这里训练的只是掩码模型。训练生成的模型保存在checkpoint/finetune里。
- 第五步，如果你需要预测并测试你的模型，则需要运行根目录下的step3_inference.py。需要注意的事，你需要将训练生成的模型改名成：mlm_trained_xx.model，xx是设置的句子最大长度，或者自行统一模型名称。

我是隔壁小王，欢迎大家留言了。
