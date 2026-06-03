import itertools
import matplotlib.pyplot as plt
import numpy as np

# Improved recommendation logic
def recommend(G, R, Pop, H):
    return (G and R) or (G and Pop) or (G and H)

# Better ground truth
def actual_like(G, R, Pop, H):
    return (G and R) or (G and H)

# Generate combinations (4 variables now)
combinations = list(itertools.product([0, 1], repeat=4))

TP = TN = FP = FN = 0

print("G R Pop H | Pred  Act  Result")
print("--------------------------------")

for combo in combinations:
    G, R, Pop, H = combo
    
    pred = int(recommend(G, R, Pop, H))
    actual = int(actual_like(G, R, Pop, H))
    
    if pred == 1 and actual == 1:
        TP += 1
        result = "TP"
    elif pred == 0 and actual == 0:
        TN += 1
        result = "TN"
    elif pred == 1 and actual == 0:
        FP += 1
        result = "FP"
    else:
        FN += 1
        result = "FN"
    
    print(G, R, Pop, H, "| ", pred, "   ", actual, " ", result)

# Metrics
total = TP + TN + FP + FN

accuracy = (TP + TN) / total
precision = TP / (TP + FP) if (TP + FP) else 0
recall = TP / (TP + FN) if (TP + FN) else 0
f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0

print("\n📊 Improved Metrics")
print("----------------------")
print(f"TP={TP}, TN={TN}, FP={FP}, FN={FN}")
print(f"Accuracy  = {accuracy:.2f}")
print(f"Precision = {precision:.2f}")
print(f"Recall    = {recall:.2f}")
print(f"F1 Score  = {f1:.2f}")

# Confusion matrix
cm = np.array([[TP, FP],
               [FN, TN]])

plt.imshow(cm)

labels = [["TP", "FP"], ["FN", "TN"]]

for i in range(2):
    for j in range(2):
        plt.text(j, i, f"{labels[i][j]}\n{cm[i][j]}",
                 ha="center", va="center")

plt.xticks([0,1], ["Pred +", "Pred -"])
plt.yticks([0,1], ["Actual +", "Actual -"])

plt.title("Improved Confusion Matrix")
plt.show()