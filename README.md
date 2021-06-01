# Syntax Infused GlossBERT with Ranking Objective
This code is modified from the code for the paper paper "[Adapting BERT for Word Sense Disambiguation with Gloss Selection Objective and Example Sentences](https://arxiv.org/abs/2009.11795)". The original code is available at https://github.com/BPYap/BERT-WSD.

The changes from original model include:
- Adding POS tag and dependency information to the model embeddings.
- Extending the model gloss with the ambiguous words and [TGT] tokens
- The inclusion of candidate senses from different word classes.
 
## Installation
```
python3 -m virtualenv env
source env/bin/activate

pip install -r requirements.txt
```



## Dataset preparation
### Training dataset
Usage:
```
python script/prepare_dataset.py --corpus_dir CORPUS_DIR --output_dir OUTPUT_DIR
                                 --max_num_gloss MAX_NUM_GLOSS
                                 [--use_augmentation]

arguments:
  --corpus_dir CORPUS_DIR
                        Path to directory consisting of a .xml file and a .txt
                        file corresponding to the sense-annotated data and its
                        gold keys respectively.
  --output_dir OUTPUT_DIR
                        The output directory where the .csv file will be
                        written.
  --max_num_gloss MAX_NUM_GLOSS
                        Maximum number of candidate glosses a record can have
                        (include glosses from ground truths)
  --use_augmentation    Whether to augment training dataset with example
                        sentences from WordNet
  --cross_pos_train     Whether to include candidate senses from different word classes
                        
```

Example:
```
python3 script/prepare_dataset_syntax.py \
    --corpus_dir "data/corpus/SemCor" \
    --output_dir "data/train" \
    --max_num_gloss 6 \
    --cross_pos_train
```

### Development/Test dataset
Usage:
```
python script/prepare_dataset.py --corpus_dir CORPUS_DIR --output_dir OUTPUT_DIR

arguments:
  --corpus_dir CORPUS_DIR
                        Path to directory consisting of a .xml file and a .txt
                        file corresponding to the sense-annotated data and its
                        gold keys respectively.
  --output_dir OUTPUT_DIR
                        The output directory where the .csv file will be
                        written.
```

Example:
```
python script/prepare_dataset.py \
    --corpus_dir "data/corpus/semeval2007" \
    --output_dir "data/dev"
```

###Caching data
Caches dataset in a format the model can read. Can create multiple cached datasets at once. The run_model syntax can also do this, but if including POS tag or dependency embeddings caching times increase. To save time, the same data can be cached for different model settings at the same time.
```
python3 script/cache_data_syntax.py --train_path --model_name_or_path "bert-base-uncased"
        --to_cache_with_gloss_extensions --to_cache_wo_gloss_extensions

arguments:
    --train_path:                     the path to the data to be cached
    --model_name_or_path:             name of the BERT model to cache data for. Model path not implemented yet.
    --to_cache_with_gloss_extensions: cached datasets using gloss extensions. Takes a string for input, where units are separated by "_".
                                      pos for a model with pos embeddings, dep for dependency embeddings and pd for a dataset using both.
                                      To run without any syntax embeddings use no.
                                      Empty string ("") means that no dataset with gloss extensions is made.
    --to_cache_wo_gloss_extensions:   cached datasets without gloss extensions. Input is the same format as for with gloss extensions.
```

Examples
```
python3 script/cache_data_syntax.py \
    --train_path "data/dev/wo_gloss_extensions/semeval2007.csv" \
    --model_name_or_path "bert-large-uncased-whole-word-masking" \
    --to_cache_with_gloss_extensions "" \
    --to_cache_wo_gloss_extensions "no_pos"
```


## Fine-tuning BERT
Usage:
```
python script/run_model_syntax.py --do_train --train_path TRAIN_PATH
                           --model_name_or_path MODEL_NAME_OR_PATH 
                           --output_dir OUTPUT_DIR
                           [--evaluate_during_training]
                           [--eval_path EVAL_PATH]
                           [--per_gpu_train_batch_size PER_GPU_TRAIN_BATCH_SIZE]
                           [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]
                           [--learning_rate LEARNING_RATE]
                           [--num_train_epochs NUM_TRAIN_EPOCHS]
                           [--logging_steps LOGGING_STEPS] 
                           [--save_steps SAVE_STEPS]

arguments:
  --do_train            Whether to run training on train set.
  --train_path TRAIN_PATH
                        Path to training dataset (.csv file).
  --model_name_or_path MODEL_NAME_OR_PATH
                        Path to pre-trained model or shortcut name selected in
                        the list: bert-base-uncased, bert-large-uncased, bert-
                        base-cased, bert-large-cased, bert-large-uncased-
                        whole-word-masking, bert-large-cased-whole-word-
                        masking
  --output_dir OUTPUT_DIR
                        The output directory where the model predictions and
                        checkpoints will be written. If "auto_create", a custome
                        name based on model parameters will be made.
  --evaluate_during_training
                        Run evaluation during training at each logging step.
  --eval_path EVAL_PATH
                        Path to evaluation dataset (.csv file).
  --per_gpu_train_batch_size PER_GPU_TRAIN_BATCH_SIZE
                        Batch size per GPU/CPU for training.
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                        Number of updates steps to accumulate before
                        performing a backward/update pass.
  --learning_rate LEARNING_RATE
                        The initial learning rate for Adam.
  --num_train_epochs NUM_TRAIN_EPOCHS
                        Total number of training epochs to perform.
  --logging_steps LOGGING_STEPS
                        Log every X updates steps.
  --save_steps SAVE_STEPS
                        Save checkpoint every X updates steps.
  --use_dependencies
                        Use dependency embeddings
  --use_pos_tags
                        Use POS tag embeddings
  --use_gloss_extensions
                        Extend glosses with the target word
  --gloss_extensions_w_tgt
                        Include [TGT] tokens in the gloss extension
```

Example:
```
python3 script/run_model_syntax.py \
    --do_train \
    --train_path "data/train/semcor-cross_pos.csv" \
    --model_name_or_path "bert-large-uncased" \
    --output_dir "auto_create" \
    --per_gpu_train_batch_size 128 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-5 \
    --num_train_epochs 4 \
    --logging_steps 1000 \
    --save_steps 1000 \
    --use_dependencies \
    --use_pos_tags \
    --use_gloss_extensions \
    --gloss_extensions_w_tgt
```


## Evaluation
### Generate predictions
Usage:
```
python script/run_model.py --do_eval --eval_path EVAL_PATH
                           --model_name_or_path MODEL_NAME_OR_PATH 
                           --output_dir OUTPUT_DIR

arguments:
  --do_eval             Whether to run evaluation on dev/test set.
  --eval_path EVAL_PATH
                        Path to evaluation dataset (.csv file).
  --model_name_or_path MODEL_NAME_OR_PATH
                        Path to pre-trained model.
  --output_dir OUTPUT_DIR
                        The output directory where the model predictions and
                        checkpoints will be written.
```
Example:

```
python3 script/run_model.py \
    --do_eval \
    --eval_path "data/test/all.csv" \
    --model_name_or_path "model/bert-large-uncased-whole-word-masking-pos-dep-no_syntax_for_special-glosses_extended_w_tgt-batch_size=1-lr=2e-05" \
    --output_dir "model/bert-large-uncased-whole-word-masking-pos-dep-no_syntax_for_special-glosses_extended_w_tgt-batch_size=1-lr=2e-05"
```

### Scoring
Usage:
```
java Scorer GOLD_KEYS PREDICTIONS

arguments:
  GOLD_KEYS    Path to gold key file
  PREDICTIONS  Path to predictions file
```
Example:
```
java Scorer data/corpus/semeval2007/semeval2007.gold.key.txt \
    model/Vanilla_model/semeval2007_predictions.txt
```


## References
- Raganato, Alessandro, Jose Camacho-Collados, and Roberto Navigli. "Word sense disambiguation: A unified evaluation framework and empirical comparison." Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 1, Long Papers. 2017.
- Huang, Luyao, et al. "GlossBERT: BERT for Word Sense Disambiguation with Gloss Knowledge." Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP). 2019.
- Wolf, Thomas, et al. "Huggingface’s transformers: State-of-the-art natural language processing." ArXiv, abs/1910.03771 (2019).





