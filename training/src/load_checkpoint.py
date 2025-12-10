from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
import torch

ORIGINAL = "Born in Bristol and raised in Glastonbury to an English father and Belgian mother, Norris began competitive kart racing aged eight. After a successful karting career, which culminated in his victory at the direct-drive World Championship in 2014, Norris graduated to junior formulae. He won his first title at the 2015 MSA Formula Championship with Carlin. He then won the Toyota Racing Series, Formula Renault Eurocup, and Formula Renault NEC in 2016, receiving the Autosport BRDC Award that year. Norris won the FIA Formula 3 European Championship in 2017, and finished runner-up to George Russell in the FIA Formula 2 Championship in 2018, both with Carlin."
CHAT_GPT = "Born in Bristol and raised in Glastonbury to an English father and a Belgian mother, Norris began competing in karting at the age of eight. He enjoyed a successful karting career, culminating in his victory at the direct-drive World Championship in 2014, before progressing into the junior single-seater categories. Norris claimed his first car-racing title in the 2015 MSA Formula Championship with Carlin. The following year, he secured championships in the Toyota Racing Series, the Formula Renault Eurocup, and Formula Renault NEC, and was awarded the Autosport BRDC Award. In 2017, Norris won the FIA Formula 3 European Championship, and in 2018 he finished runner-up to George Russell in the FIA Formula 2 Championship, again racing with Carlin."
GEMINI = "Born in Bristol and raised in Glastonbury by an English father and Belgian mother, Norris began competitive karting at the age of eight. His karting career culminated in a victory at the 2014 World Championship, after which he graduated to junior formulae. Norris secured his first single-seater title in 2015 at the MSA Formula Championship driving for Carlin. The following year, he won the Toyota Racing Series, Formula Renault Eurocup, and Formula Renault NEC, a performance that earned him the Autosport BRDC Award. Continuing with Carlin, Norris claimed the 2017 FIA Formula 3 European Championship and finished as runner-up to George Russell in the 2018 FIA Formula 2 Championship."
CLAUDE = "Born in Bristol and raised in Glastonbury to an English father and Belgian mother, Norris began competitive kart racing at the age of eight. His successful karting career culminated in victory at the direct-drive World Championship in 2014, after which he graduated to junior formulae. Norris won his first title at the 2015 MSA Formula Championship with Carlin, then swept the 2016 season by winning the Toyota Racing Series, Formula Renault Eurocup, and Formula Renault NEC, earning him the Autosport BRDC Award that year. He continued his ascent by winning the FIA Formula 3 European Championship in 2017 and finishing runner-up to George Russell in the 2018 FIA Formula 2 Championship, both with Carlin."

if __name__ == '__main__':
    checkpoint = "../models/bert-base-classifier-peft/best-acc-checkpoint-2304"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        id2label = {0: "HUMAN", 1: "AI"}
    ).to('cuda')

    test_examples = {
        "Original": ORIGINAL,
        "ChatGPT": CHAT_GPT,
        "Gemini": GEMINI,
        "Claude": CLAUDE,
    }

    model.eval()

    device = model.device
    print(f"\nModel is on device: {device}\n")

    for text_source, test_text in test_examples.items():
        inputs = tokenizer(test_text, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        decoded_input = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

        with torch.no_grad():
            logits = model(**inputs).logits

        predicted_class_id = logits.argmax().item()
        predicted_label = model.config.id2label[predicted_class_id]
        certainty = 100.0 * torch.softmax(logits, dim=-1)[0, predicted_class_id]

        print(f"{text_source}: {predicted_label} ({certainty:.2f}%)")