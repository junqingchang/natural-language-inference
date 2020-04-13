# Natural Language Inference
Authors: Chang Jun Qing, Zhao Lin

## Directory Structure
```
multinli_1.0/
    multinli_1.0_train.jsonl
    multinli_1.0_train.txt
    ...
utils/
    dataprocessing.py
model.py
eval_bert.py
train_eval_bert.py
roberta_eval.py
lossedited_bert.py
.gitignore
README.md
```

## Data
The data we are using is the MultiNLI dataset from NYU which can be downloaded at the following address

https://www.nyu.edu/projects/bowman/multinli/

## Benchmarks
BERT(base) Match Acc: 0.7950076413652573, Mismatch Acc: 0.8078722538649309

BERT(sum loss) Match Acc: 0.801018848700968, Mismatch Acc: 0.8034987794955248

Word Masking Bert Match Accuracy: 0.8002037697401936, Mismatch Accuracy: 0.8029902359641985