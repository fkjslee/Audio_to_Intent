import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertConfig
from torchcrf import CRF
from .module import IntentClassifier, SlotClassifier


class JointBERT(BertPreTrainedModel):
    def __init__(self, config, args, intent_label_lst, slot_label_lst):
        super(JointBERT, self).__init__(config)
        self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        self.bert = BertModel(config=config)  # Load pretrained bert

        self.intent_classifier = IntentClassifier(config.hidden_size, self.num_intent_labels, args.dropout_rate)
        self.slot_classifier = SlotClassifier(config.hidden_size, self.num_slot_labels, args.dropout_rate)

    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids, slot_labels_ids):
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        intent_logits = self.intent_classifier(pooled_output)
        all_slot_logits = self.slot_classifier(sequence_output)

        # remove ignore slot_logits
        slot_logits = [slot_logit[slot_label_id != -2] for slot_logit, slot_label_id in zip(all_slot_logits, slot_labels_ids)]

        result = {'net': outputs, 'intent_logits': intent_logits, 'slot_logits': slot_logits}

        if intent_label_ids[0] >= 0:
            total_loss = 0
            # 1. Intent Softmax
            intent_loss = nn.CrossEntropyLoss()(intent_logits, intent_label_ids)
            total_loss += intent_loss
            result['intent_acc'] = (torch.argmax(intent_logits, dim=1) == intent_label_ids).float().mean()

            # 2. Slot Softmax
            # Only keep active parts of the loss
            active_slot = slot_labels_ids >= 0
            slot_loss = nn.CrossEntropyLoss()(all_slot_logits[active_slot], slot_labels_ids[active_slot])
            total_loss += self.args.slot_loss_coef * slot_loss
            result['total_loss'] = total_loss
            result['slot_acc'] = (torch.argmax(all_slot_logits[active_slot]) == slot_labels_ids[active_slot]).float().mean()

        return result  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits
