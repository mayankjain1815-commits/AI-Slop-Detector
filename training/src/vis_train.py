import json
import matplotlib.pyplot as plt

checkpoint = "../models/bart-base-finetuned/with-aug-checkpoint-6450"

with open(f"{checkpoint}/trainer_state.json", "r") as file:
    trainer_state = json.load(file)

history = trainer_state["log_history"]

train_loss = []
train_loss_epochs = []

test_loss = []
test_loss_epochs = []

for item in history:
    if "eval_loss" in item.keys():
        test_loss.append(item["eval_loss"])
        test_loss_epochs.append(item["epoch"])
    elif "loss" in item.keys():
        train_loss.append(item["loss"])
        train_loss_epochs.append(item["epoch"])

plt.plot(train_loss_epochs, train_loss, label="Training")
plt.plot(train_loss_epochs, [2*x for x in train_loss], "C0--", label="Approx training")
plt.plot(test_loss_epochs, test_loss, label="Test")
plt.show()