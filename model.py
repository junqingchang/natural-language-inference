import torch
import torch.nn as nn
from transformers import BertModel


class BERT(nn.Module):
    def __init__(self, num_classes=3):
        super(BERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        outputs = self.bert(x)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits