import json
import matplotlib.pyplot as plt

checkpoint = "../models/bert-base-classifier-peft/best-acc-checkpoint-2304"

with open(f"{checkpoint}/trainer_state.json", "r") as file:
    trainer_state = json.load(file)

history = trainer_state["log_history"]

train_losses = []
train_epochs = []

test_losses = []
test_accuracies = []
test_epochs = []

for item in history:
    if "eval_loss" in item.keys():
        test_losses.append(item["eval_loss"])
        test_accuracies.append(item["eval_accuracy"])
        test_epochs.append(item["epoch"])
    elif "loss" in item.keys():
        train_losses.append(item["loss"])
        train_epochs.append(item["epoch"])

plt.plot(train_epochs, train_losses, label="Train")
plt.plot(test_epochs, test_losses, label="Test")
plt.show()


plt.plot(test_epochs, test_accuracies, label="Test")
plt.show()