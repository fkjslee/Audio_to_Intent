import os
import shutil


def vocab_process(data_dir):
    slot_label_vocab = 'slot_label.yml'
    intent_label_vocab = 'intent_label.yml'

    train_dir = os.path.join(data_dir, 'train')
    # slot
    with open(os.path.join(train_dir, 'seq.out'), 'r', encoding='utf-8') as f_r, open(os.path.join(data_dir, slot_label_vocab), 'w',
                                                                                      encoding='utf-8') as f_w:
        slot_vocab = set()
        for line in f_r:
            line = line.strip()
            slots = line.split()
            for slot in slots:
                slot_vocab.add(slot)

        slot_vocab.add("PAD")
        slot_vocab.add("UNK")
        slot_vocab = sorted(list(slot_vocab), key=lambda x: (x[2:], x[:2]))

        for slot in slot_vocab:
            f_w.write(slot + ':\n')

    # sentence
    if not os.path.exists(os.path.join(train_dir, "labeled_sentences.yml")):
        shutil.copy(os.path.join(train_dir, "seq.out"), os.path.join(train_dir, "labeled_sentences.yml"))
    with open(os.path.join(train_dir, "label"), encoding="utf-8") as f_label, open(os.path.join(train_dir, "seq.in"), encoding="utf-8") as f_in, \
            open(os.path.join(train_dir, "seq.out"), encoding="utf-8") as f_out, open(os.path.join(train_dir, "labeled_sentences.yml"), "w", encoding="utf-8") as f_write:
        intents_content = f_label.readlines()
        texts_content = f_in.readlines()
        slots_conent = f_out.readlines()
        for intent, text, slot in zip(intents_content, texts_content, slots_conent):
            text = text.replace(":", "[:]")
            f_write.write("-\n")
            f_write.write("  sentence: " + text)
            f_write.write("  intent: " + intent)
            f_write.write("  slot: " + slot)


if __name__ == "__main__":
    vocab_process('atis')
    vocab_process('snips')
