# 基于Bert的分词

说明本项目基于Allennlp实现,快速了解[Allennlp](https://github.com/allenai/allennlp/blob/v0.6.1/tutorials/getting_started/using_as_a_library_pt1.md)

## 实现思路

将问题转化为一个序列标注问题,利用预训练的bert,通过预测每个文字位置是否需要被切割得到最后的分词预测结果。
在自定义数据上,f1可达到97%

## 代码结构
```
├─preprocess \\用于数据前处理
└─textsegment \\核心代码
    ├─data  \\存放数据
    └─library \\Allennlp所需
        ├─config \\存放模型训练等参数的配置文件
        ├─dataset_readers \\自定义读数据的类
        ├─models \\自定义基于bert的序列标注模型
        ├─predictor \\ 自定义模型的预测类
        └─training  
            └─metrics \\自定义评价指标
```

# 数据说明
利用`preprocess/process_data.py`将分词的数据转化为下列格式
```
原始切分数据
心功不全 心功 三 级
转化后的数据
心功不全心功三级<sep>3	5	6
```

# 运行脚本
## 安装依赖
`pip install -r requirements.txt`

## 下载bert模型
点击[这里](https://pan.baidu.com/s/1KBNNygpDlLeO7dvKB79zTg)(提取码aup8)
下载预训练好的bert模型。

## 修改配置文件  
训练数据地址，bert模型地址，训练器，学习率等参数均写在`textsegment/library/config/bert_sl_tagger.jsonnet`中
,需要根据实际情况对其进行修改

## 训练模型
直接运行`train.py`可以进行训练，因为不能nohup，所以只用来调式

使用命令行进行训练  
```  
cd textsegement
export PYTHONPATH="$PWD"
allennlp train -f config_path -s model_path --include-package textsegment.library  
```

## 使用训练好的模型进行分词
详见`text_segment.py`