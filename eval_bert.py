from utils.dataprocessing import MNLI, BERTMNLI
import torch
import torch.nn as nn
from model import BERT


MATCH_DATA_DIR = 'drive/My Drive/multinli_1.0/multinli_1.0_dev_matched.jsonl'
MISMATCH_DATA_DIR = 'drive/My Drive/multinli_1.0/multinli_1.0_dev_mismatched.jsonl'
SAVED_MODEL_PATH = 'bert.pt'
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

def eval(dataset, model, device):
    model.eval()

    total = 0
    total_correct = 0
    for i in range(len(dataset)):
        data, target = dataset[i]
        for key in data:
            data[key] = data[key].to(device)
        target = target.to(device)
        output = model(data)
        
        preds = output.argmax(dim=1)
        for j in range(len(preds)):
            total += 1
            if preds[j] == target[j]:
                total_correct += 1

    return total_correct/total

if __name__ == '__main__':
    match = BERTMNLI(MATCH_DATA_DIR)
    mismatch = BERTMNLI(MISMATCH_DATA_DIR)

    checkpoint = torch.load(SAVED_MODEL_PATH)

    model = BERT()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    match_acc = eval(match, model, device)
    mismatch_acc= eval(mismatch, model, device)

    print(f'Match Acc: {match_acc}, Mismatch Acc:{mismatch_acc}')
