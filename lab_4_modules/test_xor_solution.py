import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

# Test different configurations
test_points = [
    (-1.5, -1.5, 0, "BL"),
    (1.5, 1.5, 0, "TR"),
    (-1.5, 1.5, 1, "TL"),
    (1.5, -1.5, 1, "BR")
]

print("="*70)
print("TESTING XOR SOLUTIONS")
print("="*70)

# Configuration 1: Vertical + Horizontal (MY WRONG SOLUTION)
print("\n1. H1=vertical(5,0,0), H2=horizontal(0,5,0), Out=(5,5,-7)")
print("-"*70)
correct = 0
for x1, x2, target, label in test_points:
    h1 = sigmoid(5*x1)
    h2 = sigmoid(5*x2)
    z_out = 5*h1 + 5*h2 - 7
    out = sigmoid(z_out)
    pred = 1 if out > 0.5 else 0
    match = "OK" if pred == target else "FAIL"
    print(f"{label}: h1={h1:.3f}, h2={h2:.3f} -> out={out:.3f} -> {pred} (want {target}) {match}")
    if pred == target:
        correct += 1
print(f"Accuracy: {correct}/4 = {correct*25}%")

# Configuration 2: Diagonal neurons
print("\n2. H1=diagonal1(5,5,-4), H2=diagonal2(5,-5,0), Out=(5,5,-7)")
print("-"*70)
correct = 0
for x1, x2, target, label in test_points:
    h1 = sigmoid(5*x1 + 5*x2 - 4)
    h2 = sigmoid(5*x1 - 5*x2)
    z_out = 5*h1 + 5*h2 - 7
    out = sigmoid(z_out)
    pred = 1 if out > 0.5 else 0
    match = "OK" if pred == target else "FAIL"
    print(f"{label}: h1={h1:.3f}, h2={h2:.3f} -> out={out:.3f} -> {pred} (want {target}) {match}")
    if pred == target:
        correct += 1
print(f"Accuracy: {correct}/4 = {correct*25}%")

# Configuration 3: Try different output weights
print("\n3. Searching for working output layer with H1=(5,0,0), H2=(0,5,0)...")
print("-"*70)

best_acc = 0
best_config = None

for w1 in range(-10, 11):
    for w2 in range(-10, 11):
        for b in range(-10, 11):
            correct = 0
            for x1, x2, target, label in test_points:
                h1 = sigmoid(5*x1)
                h2 = sigmoid(5*x2)
                z_out = w1*h1 + w2*h2 + b
                out = sigmoid(z_out)
                pred = 1 if out > 0.5 else 0
                if pred == target:
                    correct += 1

            if correct > best_acc:
                best_acc = correct
                best_config = (w1, w2, b)
                if correct == 4:
                    print(f"FOUND PERFECT SOLUTION: w_out1={w1}, w_out2={w2}, b_out={b}")
                    # Test it
                    print("Verification:")
                    for x1, x2, target, label in test_points:
                        h1 = sigmoid(5*x1)
                        h2 = sigmoid(5*x2)
                        z_out = w1*h1 + w2*h2 + b
                        out = sigmoid(z_out)
                        pred = 1 if out > 0.5 else 0
                        print(f"  {label}: h1={h1:.3f}, h2={h2:.3f} -> out={out:.3f} -> {pred} (want {target})")
                    break
    if best_acc == 4:
        break

if best_acc < 4:
    print(f"\nBest found: {best_acc}/4 with config {best_config}")
    print("NO PERFECT SOLUTION EXISTS with vertical+horizontal split!")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print("With H1 splitting left/right and H2 splitting top/bottom,")
print("the hidden space looks like:")
print("  (0,0) -> Class 0  |  (1,0) -> Class 1")
print("  (0,1) -> Class 1  |  (1,1) -> Class 0")
print("\nThis creates a CHECKERBOARD pattern in hidden space!")
print("A single straight line CANNOT separate a checkerboard.")
print("We need diagonal hidden neurons instead!")
