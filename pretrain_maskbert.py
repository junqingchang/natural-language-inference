from dataprocessing import Newsroom
from model import BertForMaskedLMWithWordMasking, BertTokenizerWithWordMasking
from transformers import BertConfig
import torch
from torch.optim import Adam
import sys


NEWSROOM_PATH = 'release/newsroom.txt'
LEARNING_RATE = 3e-5
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
SAVED_MODEL_PATH = 'wordmask_bert.pt'
NUM_ITER = 1000000
SAVE_EVERY = 100


if __name__ == '__main__':
    tokenizer = BertTokenizerWithWordMasking('configs/maskbert-vocab.txt')

    dataset = Newsroom(NEWSROOM_PATH, tokenizer=tokenizer)

    config = BertConfig(vocab_size=tokenizer.vocab_size)
    model = BertForMaskedLMWithWordMasking(config)
    model.to(device)

    optimizer = Adam(model.parameters(), lr = LEARNING_RATE)
    best_loss = sys.maxsize

    for i in range(1, NUM_ITER+1):
        data = dataset.get_random()
        optimizer.zero_grad()
        for key in data:
            data[key] = data[key].to(device)
        outputs = model(data['input_ids'], attention_mask=data['attention_mask'], token_type_ids=data['token_type_ids'], word_mask=data['word_mask'], masked_lm_labels=data['input_ids'])
        loss, prediction_scores = outputs[:2]

        loss.backward()
        optimizer.step()

        if i % SAVE_EVERY == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, f'chkpt/{i}-{loss.item()}-maskbert.pt')
        print(f'Iter {i}, Loss: {loss.item()}', flush=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, f'chkpt/{i}-{loss.item()}-maskbert.pt')