import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertConfig
from torchcrf import CRF
from .module import IntentClassifier, SlotClassifier


class JointBERT(BertPreTrainedModel):
    def __init__(self, config, args, intent_label_lst, which_slot=False):
        super(JointBERT, self).__init__(config)
        self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.bert = BertModel(config=config)  # Load pretrained bert
        self.intent_classifier = IntentClassifier(config.hidden_size, self.num_intent_labels, args.dropout_rate)
        self.which_slot = which_slot

    def forward(self, input_ids, attention_mask, intent_label_ids):
        outputs = self.bert(input_ids, attention_mask=attention_mask)  # sequence_output, pooled_output, (hidden_states), (attentions)

        result = {}

        if self.which_slot == 'slot':
            slot_logits = self.intent_classifier(outputs[0])
            result['logits'] = slot_logits
            if torch.any(intent_label_ids >= 0):
                # Slot
                active_slot = intent_label_ids >= 0
                slot_loss = nn.CrossEntropyLoss()(slot_logits[active_slot], intent_label_ids[active_slot])
                result['acc'] = (torch.argmax(slot_logits[active_slot], dim=1) == intent_label_ids[active_slot]).float().mean()
                result['total_loss'] = slot_loss
        else:
            intent_logits = self.intent_classifier(outputs[1])
            result['logits'] = intent_logits
            if torch.any(intent_label_ids[0] >= 0):
                # Intent Softmax
                intent_loss = nn.CrossEntropyLoss()(intent_logits, intent_label_ids)
                result['acc'] = (torch.argmax(intent_logits, dim=1) == intent_label_ids).float().mean()
                result['total_loss'] = intent_loss

        return result  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits
