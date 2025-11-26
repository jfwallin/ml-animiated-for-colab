import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

test_points = [
    (-1.5, -1.5, 0, "BL"),
    (1.5, 1.5, 0, "TR"),
    (-1.5, 1.5, 1, "TL"),
    (1.5, -1.5, 1, "BR")
]

print("Testing diagonal XOR solutions")
print("="*70)

# Solution: Use diagonal splits
# H1: detects bottom-left to top-right diagonal (x1 + x2)
# H2: detects top-left to bottom-right diagonal (x1 - x2)

configs = [
    # (w11, w12, b1, w21, w22, b2, w_out1, w_out2, b_out, name)
    (5, 5, 0, 5, -5, 0, 5, -5, 0, "Diagonal split"),
    (5, 5, -3, 5, -5, 0, 5, 5, -7, "Diagonal with bias"),
    (10, 10, 0, 10, -10, 0, 10, -10, 0, "Stronger diagonals"),
]

for w11, w12, b1, w21, w22, b2, w_out1, w_out2, b_out, name in configs:
    print(f"\n{name}:")
    print(f"  H1: ({w11}, {w12}, {b1})  [detects x1+x2>{-b1/(w11+w12) if (w11+w12)!=0 else 0}]")
    print(f"  H2: ({w21}, {w22}, {b2})  [detects x1-x2>{-b2/(w21-w22) if (w21-w22)!=0 else 0}]")
    print(f"  Out: ({w_out1}, {w_out2}, {b_out})")
    print(f"  {'Point':<6} {'x1':>6} {'x2':>6} {'h1':>6} {'h2':>6} {'out':>6} {'pred':>4} {'want':>4} {'':>6}")
    print("  " + "-"*64)

    correct = 0
    for x1, x2, target, label in test_points:
        h1 = sigmoid(w11*x1 + w12*x2 + b1)
        h2 = sigmoid(w21*x1 + w22*x2 + b2)
        z_out = w_out1*h1 + w_out2*h2 + b_out
        out = sigmoid(z_out)
        pred = 1 if out > 0.5 else 0
        match = "OK" if pred == target else "FAIL"
        if pred == target:
            correct += 1
        print(f"  {label:<6} {x1:6.1f} {x2:6.1f} {h1:6.3f} {h2:6.3f} {out:6.3f} {pred:4d} {target:4d} {match:>6}")

    print(f"  Accuracy: {correct}/4 = {correct*25}%")

    if correct == 4:
        print("\n  *** PERFECT SOLUTION! ***")
        print(f"\n  Hidden space visualization:")
        print(f"  h2")
        for h2_val in [1.0, 0.5, 0.0]:
            line = f"  {h2_val:3.1f} │"
            for h1_val in [0.0, 0.5, 1.0]:
                # Find which points map near this hidden location
                found = []
                for x1, x2, target, label in test_points:
                    h1 = sigmoid(w11*x1 + w12*x2 + b1)
                    h2 = sigmoid(w21*x1 + w22*x2 + b2)
                    if abs(h1 - h1_val) < 0.3 and abs(h2 - h2_val) < 0.3:
                        found.append((label, target))
                if found:
                    label_str = ",".join([f"{l}({t})" for l, t in found])
                    line += f" {label_str:12}"
                else:
                    line += "     -      "
            print(line)
        print(f"      └{'─'*40}→ h1")
        print(f"       0.0       0.5       1.0")

print("\n" + "="*70)
