import torch
from dataprocessing import MNLI
roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
roberta.eval()  # disable dropout (or leave in train mode to finetune)

TRAIN_DATA_DIR = 'multinli_1.0/multinli_1.0_train.jsonl'
Eval_data_dir1 = 'multinli_1.0/multinli_1.0_dev_matched.jsonl'
Eval_data_dir2 = 'multinli_1.0/multinli_1.0_dev_mismatched.jsonl'
mnli_train = MNLI(TRAIN_DATA_DIR)
mnli_eval = MNLI(Eval_data_dir1)
mnli_eval_mis = MNLI(Eval_data_dir2)
print(mnli_train[0])
print(mnli_eval[0])
print(mnli_eval_mis[0])

# Download RoBERTa already finetuned for MNLI
roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
label_map = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
num_correct, nsamples = 0, 0
roberta.cuda()
roberta.eval()    
with torch.no_grad():
  #with open('') as testset:
      #testset.readline()
      for index, line in enumerate(mnli_eval_mis):
        print(f'indxe {index}')
        print(f'line {line}')
        #tokens = line.strip().split('\t')
        sentense1, sentense2, target = line[0], line[1], line[-1]
        tokens = roberta.encode(sentense1, sentense2)
        prediction = roberta.predict('mnli', tokens).argmax().item()
        if (prediction==2):
          prediction=0
        elif (prediction==0):
          prediction=2
        prediction_label = label_map[prediction]
        print(f' prediction={prediction},prediction_label={prediction_label}, target = {target} ')
        num_correct = num_correct + int(prediction == target)
        nsamples = nsamples+1
          
print('accuracy ', float(num_correct)/float(nsamples))   

'''
#register a new classification head
roberta.register_classification_head('new_task', num_classes=3)
logprobs = roberta.predict('new_task', tokens)  # tensor([[-1.1050, -1.0672, -1.1245]], grad_fn=<LogSoftmaxBackward>)
'''
'''
# Extract the last layer's features
last_layer_features = roberta.extract_features(tokens)
assert last_layer_features.size() == torch.Size([1, 5, 1024])

# Extract all layer's features (layer 0 is the embedding layer)
all_layers = roberta.extract_features(tokens, return_all_hiddens=True)
assert len(all_layers) == 25
assert torch.all(all_layers[-1] == last_layer_features)
'''
