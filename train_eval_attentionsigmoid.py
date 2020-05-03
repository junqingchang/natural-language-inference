from utils.dataprocessing import BERTMNLI
import torch
#from fastai.layers import MSELossFlat
import torch.nn as nn
from model_attention import BERT_withAttentionActivation
from torch.optim import Adam
import torch.nn.functional as F


TRAIN_DATA_DIR = 'multinli_1.0/multinli_1.0_train - Copy.jsonl'
MATCH_DATA_DIR = 'multinli_1.0/multinli_1.0_dev_matched - Copy.jsonl'
MISMATCH_DATA_DIR = 'multinli_1.0/multinli_1.0_dev_mismatched - Copy.jsonl'
SAVED_MODEL_PATH = 'BERT_withAttentionActivation.pt'
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
LEARNING_RATE = 3e-5
NUM_EPOCHS = 3
BERT_TYPE = 'bert-base-cased'

def train(dataset, model, criterion, optimizer, device, print_every=10):
    model.train()

    total_loss = 0
    for i in range(len(dataset)):
        data, target = dataset[i]
        for key in data:
            data[key] = data[key].to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        #loss = criterion(output.view((-1, 3)), target.view(-1))
        #outvalue = [max(output[:,:])]
        #print(f'outvalue: {outvalue}')

        #outvalue =torch.cuda.FloatTensor(outvalue)
        #loss = criterion(output[:, 0], target.view(-1))
        loss = F.nll_loss(F.log_softmax(output).view((-1, 3)), target.view(-1))
        #loss = criterion(outvalue.view(-1), target.view(-1))
        #print(f'Loss: {loss.item()}')
       # print(f' output is {output}, , {output.view(-1,3)}')
        total_loss += loss.item()
        
        #loss = loss.item().type(dtype = torch.float)
        loss = loss.float()
        loss.backward()
        optimizer.step()

        if i%print_every == 0:
          print(f'{i}/{len(dataset)} Loss: {loss.item()}')
        if i%(print_every*10000) == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, 'storage/BERT_withAttentionActivation.pt')

    return total_loss/len(dataset)

def eval(dataset, model, device):
    model.eval()

    total = 0
    total_correct = 0
    for i in range(len(dataset)):
        data, target = dataset[i]
        for key in data:
            data[key] = data[key].to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        
        preds = output.argmax(dim=1)
        for j in range(len(preds)):
            total += 1
            if preds[j] == target[j]:
                total_correct += 1

    return total_correct/total


mnli = BERTMNLI(TRAIN_DATA_DIR, bert_type=BERT_TYPE)
match = BERTMNLI(MATCH_DATA_DIR, bert_type=BERT_TYPE)
mismatch = BERTMNLI(MISMATCH_DATA_DIR, bert_type=BERT_TYPE)

model = BERT_withAttentionActivation(bert_type=BERT_TYPE)
model.to(device)

optimizer = Adam(model.parameters(), lr = LEARNING_RATE)
criterion = nn.CrossEntropyLoss()
#criterion = MSELossFlat()
#criterion = F.nll_loss()
#criterion = nn.NLLLoss()
best_acc = 0

for epoch in range(1, NUM_EPOCHS+1):
    train_loss = train(mnli, model, criterion, optimizer, device)
    match_acc = eval(match, model, device)
    mismatch_acc= eval(mismatch, model, device)

    print(f'Epoch {epoch}, Train Loss: {train_loss}, Match Acc: {match_acc}, Mismatch Acc:{mismatch_acc}')
    if match_acc+mismatch_acc > best_acc:
        best_acc = match_acc+mismatch_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'match_acc': match_acc,
            'mismatch_acc': mismatch_acc
            }, SAVED_MODEL_PATH)