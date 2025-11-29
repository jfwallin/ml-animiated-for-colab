import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

test_points = [
    (-1.5, -1.5, 0, "BL"),
    (1.5, 1.5, 0, "TR"),
    (-1.5, 1.5, 1, "TL"),
    (1.5, -1.5, 1, "BR")
]

print("Searching for working XOR solution...")
print("="*70)

best_acc = 0
solutions = []

# Search over hidden layer configurations
for w11 in range(-10, 11, 1):
    for w12 in range(-10, 11, 1):
        for b1 in range(-10, 11, 1):
            for w21 in range(-10, 11, 1):
                for w22 in range(-10, 11, 1):
                    for b2 in range(-10, 11, 1):
                        # For each hidden config, search output layer
                        for w_out1 in range(-10, 11, 1):
                            for w_out2 in range(-10, 11, 1):
                                for b_out in range(-10, 11, 1):
                                    correct = 0
                                    for x1, x2, target, label in test_points:
                                        h1 = sigmoid(w11*x1 + w12*x2 + b1)
                                        h2 = sigmoid(w21*x1 + w22*x2 + b2)
                                        z_out = w_out1*h1 + w_out2*h2 + b_out
                                        out = sigmoid(z_out)
                                        pred = 1 if out > 0.5 else 0
                                        if pred == target:
                                            correct += 1

                                    if correct == 4 and (w11, w12, b1, w21, w22, b2, w_out1, w_out2, b_out) not in solutions:
                                        config = (w11, w12, b1, w21, w22, b2, w_out1, w_out2, b_out)
                                        solutions.append(config)
                                        print(f"\nFound solution #{len(solutions)}:")
                                        print(f"  H1: w11={w11}, w12={w12}, b1={b1}")
                                        print(f"  H2: w21={w21}, w22={w22}, b2={b2}")
                                        print(f"  Out: w_out1={w_out1}, w_out2={w_out2}, b_out={b_out}")

                                        # Show hidden space mapping
                                        print("  Hidden space:")
                                        for x1, x2, target, label in test_points:
                                            h1 = sigmoid(w11*x1 + w12*x2 + b1)
                                            h2 = sigmoid(w21*x1 + w22*x2 + b2)
                                            z_out = w_out1*h1 + w_out2*h2 + b_out
                                            out = sigmoid(z_out)
                                            print(f"    {label}: ({h1:.2f}, {h2:.2f}) -> {out:.3f} -> {1 if out>0.5 else 0}")

                                        if len(solutions) >= 5:
                                            break
                                if len(solutions) >= 5:
                                    break
                            if len(solutions) >= 5:
                                break
                        if len(solutions) >= 5:
                            break
                    if len(solutions) >= 5:
                        break
                if len(solutions) >= 5:
                    break
            if len(solutions) >= 5:
                break
        if len(solutions) >= 5:
            break
    if len(solutions) >= 5:
        break

print(f"\n{'='*70}")
print(f"Found {len(solutions)} working solution(s)")
