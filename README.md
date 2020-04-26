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
BERT(base), Match Acc: 0.7950076413652573, Mismatch Acc: 0.8078722538649309

Finetune with Sum Loss, Match Acc: 0.801018848700968, Mismatch Acc: 0.8034987794955248

Finetune with Word Activation Masking, Match Accuracy: 0.8002037697401936, Mismatch Accuracy: 0.8029902359641985

Finetuning with DialogNLI corpus: Match Accuracy 0.7866530820173204, Mismatch Accuracy:0.7955655004068348

Finetuning with MSRP corpus: Match Accuracy: 0.8008150789607743, Mismatch Accuracy:0.8045158665581774

Finetuning with Sigmoid activation function in BertPooler: Match Acc: 0.8020376974019359, Mismatch Acc: 0.809092758340114

Finetuning with ReLU activation function in BertPooler: Match Acc: 0.8011207335710647, Mismatch Acc: 0.7971928397070789

Finetuning with Sum Loss + Sigmoid activation + Word Actication Masking, Match Acc: 0.801935812531839, Mismatch Acc: 0.8019731489015459

Finetuning with Sum Loss + Sigmoid activation, Match Acc: 0.800509424350484, Mismatch Acc:0.8046175752644427