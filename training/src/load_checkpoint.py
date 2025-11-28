from transformers import (
    AutoModelForSeq2SeqLM,
    BartTokenizer,
    BatchEncoding,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import torch

from utils import remove_emojis

POST = """
🔍 Generative AI is not neutral.
It is WEIRD — Western, Educated, Industrialized, Rich, Democratic.

This is the profile that dominates the data, the teams and the epistemic frameworks shaping much of today’s Artificial Intelligence — the same AI that increasingly governs public and private decision-making.

In other words:
AI is ethnocoded.
Models that “learn” about the world, but only from the world as experienced by Global North elites. Systems that reproduce — at speed and scale — the racial, class, gender and territorial biases already embedded in our societies.

For those of us working in human rights, on the rights of Afro-descendant and Indigenous peoples, on the rights of migrants and asylum seekers, and on intersectional discrimination and gender inequality, this is not a technical anecdote:
➡️ it is a matter of algorithmic justice.

Because when algorithms cannot “see” certain bodies, they erase them.
When they cannot recognize certain territories, they subordinate them.
When they fail to understand certain accents, languages or cultural patterns, they penalize the people who embody them.

So the question is not only “How do we reduce bias?” but:
✨ Who designs this technology? From which worldview? And with what consequences for those who were never part of the dominant “we”?

Algorithmic justice requires:
✔️ Diversifying teams and datasets beyond the WEIRD model.
✔️ Designing technology with real participation from racialized and marginalized communities.
✔️ Embedding decolonial and intersectional perspectives across the entire tech cycle.
✔️ Demanding transparency, audits and democratic accountability.

AI can expand rights — or deepen inequalities.
It depends on whether we continue accepting ethnocoded technology…
…or whether we choose to build systems that reflect the full plurality of the world we actually live in.

This conversation is not optional.
It is urgent.

Look at the image and ask yourself:
How close is your own worldview to the way ChatGPT “thinks”?


(text by Diego Battistessa , Social Change School advisor)

hashtag#AI hashtag#AlgorithmicJustice hashtag#Etnocoding hashtag#HumanRights hashtag#WEIRD hashtag#Bias
""".strip()

POST = remove_emojis(POST)

if __name__ == '__main__':
    checkpoint = "../models/bart-base-finetuned/with-aug-checkpoint-1920"
    tokenizer = BartTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to("cuda")

    test_examples = [
        POST
    ]

    model.eval()

    device = model.device
    print(f"\nModel is on device: {device}\n")

    for i, test_text in enumerate(test_examples, 1):
        inputs = tokenizer(test_text, return_tensors="pt", max_length=2048, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        decoded_input = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

        print(f"{'='*60}")
        print(f"Input:")
        print(decoded_input)
        
        outputs = model.generate(
            inputs["input_ids"],
            max_length=2048,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2,
        )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"{'='*60}")
        print(f"Generated:")
        print(generated)