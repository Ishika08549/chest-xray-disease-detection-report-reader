
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load training history
history = pd.read_csv("outputs/training_history.csv")

# Display training results
print("\nTRAINING HISTORY")
print(history)

# Find best epoch
best_epoch = history.loc[
    history["val_loss"].idxmin()
]

print("\nBEST MODEL DETAILS")
print("Best Epoch:", int(best_epoch["epoch"]))
print("Training Loss:", best_epoch["train_loss"])
print("Validation Loss:", best_epoch["val_loss"])

# Plot training and validation loss
plt.figure(figsize=(9, 5))

plt.plot(
    history["epoch"],
    history["train_loss"],
    marker="o",
    label="Training Loss"
)

plt.plot(
    history["epoch"],
    history["val_loss"],
    marker="o",
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ResNet18 Training vs Validation Loss")

plt.xticks(history["epoch"])
plt.legend()
plt.grid(True)

# Save graph
os.makedirs("outputs/plots", exist_ok=True)

plt.savefig(
    "outputs/plots/training_validation_loss.png",
    dpi=300,
    bbox_inches="tight"
)

print("\nGraph saved successfully!")
print("Location: outputs/plots/training_validation_loss.png")

plt.show()
