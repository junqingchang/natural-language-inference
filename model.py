import torch
import torch.nn as nn
from transformers import BertModel


class BERT(nn.Module):
    def __init__(self, num_classes=3, bert_type='bert-large-cased'):
        super(BERT, self).__init__()
        assert bert_type == 'bert-large-cased' or bert_type == 'bert-base-cased'
        self.bert = BertModel.from_pretrained(bert_type)
        self.dropout = nn.Dropout(0.1)
        if bert_type == 'bert-base-cased':
            self.classifier = nn.Linear(768, num_classes)
        elif bert_type == 'bert-large-cased':
            self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        outputs = self.bert(
            x['input_ids'], attention_mask=x['attention_mask'], token_type_ids=x['token_type_ids'])
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits
