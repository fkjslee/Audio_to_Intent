import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertConfig
from torchcrf import CRF
from .module import IntentClassifier, SlotClassifier


class JointBERT(BertPreTrainedModel):
    def __init__(self, config, args, intent_label_lst):
        super(JointBERT, self).__init__(config)
        self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.bert = BertModel(config=config)  # Load pretrained bert
        self.intent_classifier = IntentClassifier(config.hidden_size, self.num_intent_labels, args.dropout_rate)

    def forward(self, input_ids, attention_mask, intent_label_ids):
        outputs = self.bert(input_ids, attention_mask=attention_mask)  # sequence_output, pooled_output, (hidden_states), (attentions)
        pooled_output = outputs[1]  # [CLS]
        intent_logits = self.intent_classifier(pooled_output)

        result = {'net': outputs, 'intent_logits': intent_logits}

        if intent_label_ids[0] >= 0:
            # 1. Intent Softmax
            intent_loss = nn.CrossEntropyLoss()(intent_logits, intent_label_ids)
            result['intent_acc'] = (torch.argmax(intent_logits, dim=1) == intent_label_ids).float().mean()
            result['total_loss'] = intent_loss

        return result  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits
