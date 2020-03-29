from utils.dataprocessing import LossEditedBERTMNLI, BERTMNLI
import torch
import torch.nn as nn
from model import BERT
from torch.optim import Adam


TRAIN_DATA_DIR = 'multinli_1.0/multinli_1.0_train.jsonl'
MATCH_DATA_DIR = 'multinli_1.0/multinli_1.0_dev_matched.jsonl'
MISMATCH_DATA_DIR = 'multinli_1.0/multinli_1.0_dev_mismatched.jsonl'
SAVED_MODEL_PATH = 'lossedited_bert.pt'
use_cuda = torch.cuda.is_available()
# device = torch.device('cuda' if use_cuda else 'cpu')
device = torch.device('cpu')
LEARNING_RATE = 3e-5
NUM_EPOCHS = 3
BERT_TYPE = 'bert-base-cased'

def train(dataset, model, criterion, optimizer, device, print_every=1000):
    model.train()

    total_loss = 0
    for i in range(len(dataset)):
        sentences, target = dataset[i]
        target = target.to(device)
        losses = []
        for j in range(len(sentences)):
            data = sentences[j]

            for key in data:
                data[key] = data[key].to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.view((-1, 3)), target.view(-1))
            losses.append(loss)

        step_loss = sum(losses)
        total_loss += step_loss.item()

        step_loss.backward()
        optimizer.step()

        if i%print_every == 0:
            print(f'{i}/{len(dataset)} Loss: {loss.item()}')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, 'backup.pt')

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
    mnli = LossEditedBERTMNLI(TRAIN_DATA_DIR, bert_type=BERT_TYPE)
    match = BERTMNLI(MATCH_DATA_DIR, bert_type=BERT_TYPE)
    mismatch = BERTMNLI(MISMATCH_DATA_DIR, bert_type=BERT_TYPE)

    model = BERT(bert_type=BERT_TYPE)
    model.to(device)

    optimizer = Adam(model.parameters(), lr = LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

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