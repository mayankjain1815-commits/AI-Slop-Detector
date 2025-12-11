from typing import Callable

from transformers import AutoTokenizer, AutoModelForSequenceClassification, PreTrainedTokenizer
from peft import PeftModel  #, PeftConfig
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model() -> tuple[PeftModel, PreTrainedTokenizer]:
    base_model_name = "google-bert/bert-base-cased"
    lora_model_name = "../models/bert-base-classifier-peft/best-acc-checkpoint-2304"

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    # peft_config = PeftConfig.from_pretrained(lora_model_name)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=2
    )

    model = PeftModel.from_pretrained(base_model, lora_model_name)

    model = model.to(DEVICE)
    model.eval()

    return model, tokenizer


def _batch_probabilities(paragraphs: list[str], model: PeftModel, tokenizer: PreTrainedTokenizer) -> torch.Tensor:
    inputs = tokenizer(
        paragraphs, 
        return_tensors="pt", 
        truncation=True, 
        max_length=512,
        padding=True
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    return probabilities[:, 1]


def run_inference(text: str, model: PeftModel, tokenizer: PreTrainedTokenizer) -> torch.Tensor:
    """Returns estimated probability that a text was produced by AI."""
    paragraphs = text.split('\n')
    paragraphs = [p.strip() for p in paragraphs if len(p) > 7]

    paragraphs = paragraphs or [""]

    return _batch_probabilities(paragraphs, model, tokenizer)


if __name__ == '__main__':
    # Example usage
    text1 = "After the outbreak of the Second World War, he joined the Royal Navy as a probationary acting sub-lieutenant, being confirmed in the rank on 15 October 1940. He was promoted to the rank of lieutenant on 15 October 1942. He served in the Royal Naval Volunteer Reserve following the end of the war, being promoted to the rank of acting lieutenant commander on 1 January 1947, to acting commander on 1 January 1951, and to the rank of commander on 30 June 1954. He was made an OBE in the 1954 Birthday Honours. He died at Berkhamsted, Hertfordshire on 6 September 2010."
    text2 = "Born in Bristol and raised in Glastonbury, Norris is the son of an English father and a Belgian mother. He began competing in karting at the age of eight and quickly rose through the ranks, ultimately winning the direct-drive Karting World Championship in 2014. From there, he moved into junior single-seater racing. Norris captured his first car-racing title in the 2015 MSA Formula Championship with Carlin. In 2016, he added victories in the Toyota Racing Series, Formula Renault Eurocup, and Formula Renault NEC, and earned the Autosport BRDC Award. He went on to win the FIA Formula 3 European Championship in 2017 and finished second to George Russell in the 2018 FIA Formula 2 Championship, again driving for Carlin."

    model, tokenizer = load_model()
    # result = _batch_probabilities([text1, text2], model, tokenizer)
    result = run_inference(text1 + "\n" + text2, model, tokenizer)

    print(result)
