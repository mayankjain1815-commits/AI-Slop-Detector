from transformers import PreTrainedTokenizer, AutoTokenizer
from datasets import Dataset, DatasetDict
import numpy as np

from train import get_datasets

def analyze_token_lengths(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizer
):
    def tokenize_and_get_length(examples: Dataset) -> dict[str, list[int]]:
        """Tokenize texts and return their lengths."""
        tokenized = tokenizer(examples["text"], truncation=False, padding=False)
        return {"token_length": [len(ids) for ids in tokenized["input_ids"]]}  # type: ignore
    
    # Add token lengths to the dataset
    dataset_with_lengths = dataset.map(
        tokenize_and_get_length,
        batched=True,
        desc="Tokenizing and computing lengths"
    )
    
    # Compute statistics for each split
    stats = {}
    for split_name in dataset_with_lengths.keys():
        lengths = dataset_with_lengths[split_name]["token_length"]
        lengths = np.array(lengths)
        
        stats[split_name] = {
            "count": len(lengths),
            "min": int(np.min(lengths)),
            "max": int(np.max(lengths)),
            "mean": float(np.mean(lengths)),
            "median": float(np.median(lengths)),
            "q95": float(np.percentile(lengths, 95)),
            "q99": float(np.percentile(lengths, 99)),
            "num_above_512": np.sum(lengths > 512),
            "num_above_1024": np.sum(lengths > 1024),
        }
    
    return stats, dataset_with_lengths

if __name__ == '__main__':
    # MODEL = 0

    # if MODEL == 0:
    #     print("Using Bart!\n")
    #     checkpoint = "facebook/bart-base"
    #     tokenizer = BartTokenizer.from_pretrained(checkpoint)
    #     # model = BartForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    # elif MODEL == 1:
    #     print("Using Bert!\n")
    #     checkpoint = "bert-base-cased"
    #     tokenizer = BertTokenizer.from_pretrained(checkpoint)
    #     # model = BertForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    # else:
    #     quit()

    raw_datasets = get_datasets()

    checkpoints = ["facebook/bart-base", "bert-base-cased"]

    for checkpoint in checkpoints:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        stats, _ = analyze_token_lengths(raw_datasets, tokenizer)
        for split_name, split_stats in stats.items():
            print(split_name)
            print("="*len(split_name))
            for stat, value in split_stats.items():
                print(f"{stat}: {value}")
            print("")



# train_data = {
#     "input": [
#         "First line. Second line. Third line.",
#         "Item 1: First thing Item 2: Second thing Item 3: Third thing",
#         "Title goes here Main content is this Summary at the end",
#         "Step one do this Step two do that Step three finish up",
#         "Introduction paragraph Body paragraph Conclusion paragraph",
#         "Header text Subheader text Content text Footer text",
#         "Question: What is AI? Answer: Artificial Intelligence Question: Why important? Answer: Transforms technology",
#         "Name: John Age: 30 City: NYC Job: Engineer",
#         "Monday meeting Tuesday deadline Wednesday presentation Thursday review Friday submit",
#         "Breakfast: eggs Lunch: salad Dinner: pasta Snack: fruit",
#     ],
#     "target": [
#         "First line.\nSecond line.\nThird line.",
#         "Item 1: First thing\nItem 2: Second thing\nItem 3: Third thing",
#         "Title goes here\nMain content is this\nSummary at the end",
#         "Step one do this\nStep two do that\nStep three finish up",
#         "Introduction paragraph\nBody paragraph\nConclusion paragraph",
#         "Header text\nSubheader text\nContent text\nFooter text",
#         "Question: What is AI?\nAnswer: Artificial Intelligence\nQuestion: Why important?\nAnswer: Transforms technology",
#         "Name: John\nAge: 30\nCity: NYC\nJob: Engineer",
#         "Monday meeting\nTuesday deadline\nWednesday presentation\nThursday review\nFriday submit",
#         "Breakfast: eggs\nLunch: salad\nDinner: pasta\nSnack: fruit",
#     ]
# }

# dataset = Dataset.from_dict(train_data)

# def preprocess(examples):
#     inputs = tokenizer(examples["input"], max_length=1024, truncation=True, padding="max_length")
#     targets = tokenizer(examples["target"], max_length=512, truncation=True, padding="max_length")
    
#     inputs["labels"] = targets["input_ids"]
#     return inputs

# tokenized_dataset = dataset.map(preprocess, batched=True)

# training_args = TrainingArguments(
#     output_dir="./bart-newline-finetuned",
#     num_train_epochs=32,
#     per_device_train_batch_size=2,
#     save_steps=500,
#     save_total_limit=2,
#     learning_rate=5e-5,
#     logging_steps=10,
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_dataset,
# )

# print("Starting training...")
# trainer.train()
# print("Training complete!\n")


# # TEST GENERATION WITH NEWLINES
# print("="*60)
# print("TESTING NEWLINE GENERATION")
# print("="*60)

# test_examples = [
#     "First line. Second line. Third line.",  # From training
#     "Item 1: First thing Item 2: Second thing Item 3: Third thing",  # From training
#     "Apple Banana Orange Grape",  # New example
#     "Chapter one begins Chapter two continues Chapter three ends",  # New example
#     "Morning routine Afternoon work Evening relax Night sleep",  # New example
# ]

# model.eval()

# device = model.device
# print(f"\nModel is on device: {device}\n")

# for i, test_text in enumerate(test_examples, 1):
#     print(f"\n{'='*60}")
#     print(f"Test {i}:")
#     print(f"Input: {test_text}")
#     print(f"{'='*60}")
    
#     inputs = tokenizer(test_text, return_tensors="pt", max_length=1024, truncation=True)
    
#     inputs = {k: v.to(device) for k, v in inputs.items()}
    
#     outputs = model.generate(
#         inputs["input_ids"],
#         max_length=150,
#         num_beams=4,
#         early_stopping=True,
#         no_repeat_ngram_size=2,
#     )
    
#     generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
#     print(f"Generated (repr): {repr(generated)}")
#     print(f"Generated (formatted):\n{generated}")
    
#     newline_count = generated.count('\n')
#     print(f"Newlines found: {newline_count}")

# print("\n" + "="*60)
# print("TESTING COMPLETE")
# print("="*60)
