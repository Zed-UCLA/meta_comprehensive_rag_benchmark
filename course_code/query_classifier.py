from transformers import BertModel
import torch.nn as nn

class MultiTaskBERT(nn.Module):
    def __init__(self, num_domain_classes):
        super(MultiTaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.domain_classifier = nn.Linear(self.bert.config.hidden_size, num_domain_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # CLS token representation
        domain_logits = self.domain_classifier(pooled_output)
        return domain_logits
