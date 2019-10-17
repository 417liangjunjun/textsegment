{
    "dataset_reader": {
        "type": "bert_sl_tagger",
        "token_indexers": {
            "bert": {
                "type": "bert-pretrained",
                "pretrained_model": "~/chinese_L-12_H-768_A-12/",
                // 将bert模型地址改为相应的地址
                "do_lowercase": true,
                "use_starting_offsets": false
            }
        },
    },
  "train_data_path": "~/textsegment/textsegment/data/train.txt",
  "validation_data_path": "~/textsegment/textsegment/data/dev.txt",
  // 将训练数据和验证数据地址改为相应的地址
    "model": {
        "type": "bert_sl_tagger",
        "text_field_embedder": {
            "allow_unmatched_keys": true,
            "embedder_to_indexer_map": {
                "bert": ["bert", "bert-offsets", "bert-type-ids"],
            },
            "token_embedders": {
                "bert": {
                    "type": "bert-pretrained",
                    "pretrained_model": "~/chinese_L-12_H-768_A-12/",
                    // 将bert模型地址改为相应的地址
                    "requires_grad": true,
                    "top_layer_only": true
                }
            }
        }
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [
            [
                "text",
                "num_tokens"
            ]
        ],
        "batch_size":6 ,
        "max_instances_in_memory": 600
    },
    "trainer": {
        "num_epochs":20 ,
        "grad_norm": 5,
        "patience": 5,
        "validation_metric": "+f1",
        "cuda_device":3 ,
        "optimizer": {
            "type": "bert_adam",
            "lr": 1e-5,
            "warmup": 0.1,
            "t_total": 60000
        }
    }
}
