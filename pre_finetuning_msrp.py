from dataprocessing import MNLI, BERTMNLI,BERTMNLI_preFinetuning
import torch
import torch.nn as nn
from model import BERT,  BERTwithMSRP
from torch.optim import Adam
from load_msrp import train_data,test_data
from pandas import DataFrame


SAVED_MODEL_PATH = 'storage/bert-base-msrp.pt'
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
LEARNING_RATE = 3e-5
NUM_EPOCHS = 3
BERT_TYPE = 'bert-base-cased'

def train(dataset, model, criterion, optimizer, device, print_every=200):
    model.train()

    total_loss = 0
    for i in range(len(dataset)):
        data, target = dataset[i]

        for key in data:
            data[key] = data[key].to(device)
        target = target.to(device)
        optimizer.zero_grad()        

        next_output= model.forward(data)
        loss = criterion(next_output, target.view(-1))
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        print(f'{i}/{len(dataset)} Loss: {loss.item()}')
        if i%(print_every*100) == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, 'storage/bert-base-msrp-backup.pt')

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


if __name__ == '__main__':

    model = BERTwithMSRP(bert_type = BERT_TYPE)
    
    print('------msrp---------')

    train_data1 = train_data.values.tolist()
    test_data1 = test_data.values.tolist()
    
    data_nsp =BERTMNLI_preFinetuning(train_data1,  bert_type=BERT_TYPE)
    data_test_nsp =BERTMNLI_preFinetuning(test_data1,  bert_type=BERT_TYPE)

    model.to(device)

    optimizer = Adam(model.parameters(), lr = LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0

    for epoch in range(1, NUM_EPOCHS+1):
        print(f'enter epoch {epoch}')
       # train_loss = train(mnli, model, criterion, optimizer, device)
        train_loss = train(data_nsp, model, criterion, optimizer, device)
        match_acc = eval(data_test_nsp, model, device)
        mismatch_acc =0

        print(f'Epoch {epoch}, Train Loss: {train_loss}, Match Acc: {match_acc}, Mismatch Acc:{mismatch_acc}')
        if match_acc+mismatch_acc >= best_acc:
            best_acc = match_acc+mismatch_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'match_acc': match_acc,
                'mismatch_acc': mismatch_acc
                }, SAVED_MODEL_PATH)
