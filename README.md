# Natural Language Inference
Authors: Chang Jun Qing, Zhao Lin

## Directory Structure
```
multinli_1.0/
    multinli_1.0_train.jsonl
    multinli_1.0_train.txt
    ...
configs/
    maskbert-vocab.txt
.gitignore
dataprocessing.py
load_dnli.py
load_msrp.py
model_attention.py
model_pooler.py
eval_roberta.py
model.py
pre_finetuning_DNLI.py
pre_finetuning_msrp.py
pretrain_maskbert.py
README.md
train_eval_attentionsigmoid.py
train_eval_bert.py
train_eval_fullmodel.py
train_eval_lossedited_bert.py
train_eval_maskbert.py
train_eval_poolerlosseditedbert.py
train_eval_poolersigmoid.py
train_eval_pretrain_DNLI.py
train_eval_pretrain_MSRP.py
train_eval_pretrainedmaskbert.py
```

## Approach

### Sum Loss of Sentence Pair
Typically, cross entropy loss is used for MNLI task where we calculate the loss from the predicted label against the targets. In MNLI, the data provided contains 2 sentences and the normal approach would be to use sentence_1 -> sentence_2 as the input to the model to predict the label for the pair of sentence.

However, MNLI is not like Next Sentence Prediction where the order of the sentences matter. Due to the nature of our task, we want to avoid the model learning like a Next Sentence Prediction Model and believe that we can augment the dataset by utilising the sentence pairs in both orders with the same target.

This enables us to ensure that our model learns to optimise in both orderings.

### Word Activation Masking
BERT models make use of Byte-Pair Encoding(BPE) which is an in-between of word encoding and character encoding, this can be thought of as sub-word encoding, which allows handling vocabularies common in natural language even when certain full words may not be part of the model's vocabulary. This also opens up to potential learning of relations between sub-words instead of full words by the model. Due to this, we decided to include another layer of masking into the embedding layer of the model which activates the first sub-word of each full word. This enables our model to know where every next word of the sentence is. Similar to how BERT's Whole Word Masking implemented in May 2019 that improved the performance of the model, we believe that this will help reduce the impact of relations between subwords.

### Finetuning using other NLI datasets
Given the idea that BERT using NSP, ALBERT using SOP and RoBERT doesnt use any task to compute pre-training loss, we came out the idea to experiment on different prediction task to pre-train BERT model. The task are not exactly language inference task but similar task that the model are required to learn and understand the context. Two corpus are used for experiment in this session. They are Dialog NLI (DNLI) corpus and Microsoft Research Paraphrase (MSRP) Corpus. Both corpus are well prepared for classification tasks. The details for corpus will be introduced later. Since we do not have enough resources to re-train the whole system. To implement this experiment, we treat it as an additional fine-tuning corpus before fine-tuning on MNLI corpus.We hope the system can learn the parameter from more relevant resources before fine-tuning on specific downstream task. 

### Changing activation of attention
BERT model is in general the encoder part of transformer. It contains the attention layer to select the context that better fit the current node instead of fitting all the contents into a vectors. In attention layer, softmax activation function plays an important role in the attention mechanism. In the attention layer, the decoder does not use the inputs from all encoders. Instead, it selects the state that is best matches with current node by calculating compatibility score. To convert these scores into valid attention weights, we need to normalize them and sum the values to 1. This is achieved by softmax function. Softmax function also helps to enlarge the higher component and minimize the lower component. The CrossEntropy cost function is applied to compute the output from softmax function for classification or regression. There are studies that reveal the bottleneck for softmax activation. In the paper from, they proved that there is representational capability issue in language model when using softmax.They demonstrated that the softmax-based model is restricted by the length of the hidden nodes in output layer. Therefore, a few methods are proposed to replace softmax. They are 1) sigsoftmax which combine softmax with sigmoid function. 2) weighted sum of softmax 3) ReLU and sigmoid combination. We plan to try out sigsoftmax first followed by weighted sum of softmax. 

## Data
The data we are using as benchmark is the MultiNLI dataset from NYU which can be downloaded at https://www.nyu.edu/projects/bowman/multinli/

The datasets used for additional pretraining are DNLI https://wellecks.github.io/dialogue_nli/ 

and 

MSRP Corpus https://www.microsoft.com/en-us/research/project/nlpwin/?from=http%3A%2F%2Fresearch.microsoft.com%2Fresearch%2Fnlp%2Fmsr_

For pretraining, we are using Cornell Newsroom dataset which can be requested at https://summari.es/

## Benchmarks
| Model                                                                        | Match Accuracy | Mismatch Accuracy |
|------------------------------------------------------------------------------|----------------|-------------------|
| BERT (base)                                                                  | 79.50          | 80.79             |
| Finetuning with  + Sum Loss                                                  | 80.10          | 80.35             |
| Finetuning with  + Word Activation Masking                                   | 80.02          | 80.30             |
| Pretraining  Word Activation Masking  (1000000 iterations)                   | 31.82          | 31.82             |
| Pretraining  Word Activation Masking  (6000000 iterations)                   | 35.45          | 35.22             |
| Finetuning with  + DialogNLI Corpus                                          | 78.67          | 79.56             |
| Finetuning with  + MSRP Corpus                                               | 80.08          | 80.45             |
| Finetuning with  + Sigmoid Activation                                        | 80.20          | 80.91             |
| Finetuning with  + ReLU Activation                                           | 80.11          | 79.72             |
| Finetuning with  + Sum Loss  + Sigmoid Activation  + Word Activation Masking | 80.19          | 80.19             |
| Finetuning with  + Sum Loss  + Sigmoid Activation                            | 80.05          | 80.46             |