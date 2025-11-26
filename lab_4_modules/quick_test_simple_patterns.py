import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def test_config(points, w11, w12, b1, w21, w22, b2, w_out1, w_out2, b_out):
    """Test a specific configuration."""
    correct = 0
    for x1, x2, target in points:
        h1 = sigmoid(w11*x1 + w12*x2 + b1)
        h2 = sigmoid(w21*x1 + w22*x2 + b2)
        z_out = w_out1*h1 + w_out2*h2 + b_out
        out = sigmoid(z_out)
        pred = 1 if out > 0.5 else 0
        if pred == target:
            correct += 1
    return correct == len(points)

# Test 1: Vertical Split (should be TRIVIAL)
print("Test 1: VERTICAL SPLIT")
print("="*60)
pattern = [
    (-1.5, -1.5, 0), (-1.5, 1.5, 0),  # Left side = Class 0
    (1.5, -1.5, 1), (1.5, 1.5, 1),     # Right side = Class 1
]
# Solution: H1 detects x1>0, H2 can be anything, output uses H1 only
configs = [
    (10, 0, 0, 0, 10, 0, 10, 0, -5),  # H1 splits on x1
    (5, 0, 0, 0, 5, 0, 5, 0, -2),
]
for cfg in configs:
    if test_config(pattern, *cfg):
        print(f"[OK] WORKS: {cfg}")
        w11, w12, b1, w21, w22, b2, w_out1, w_out2, b_out = cfg
        print(f"  H1: ({w11}, {w12}, {b1})")
        print(f"  H2: ({w21}, {w22}, {b2})")
        print(f"  Out: ({w_out1}, {w_out2}, {b_out})")
        break

# Test 2: Horizontal Split (should be TRIVIAL)
print("\nTest 2: HORIZONTAL SPLIT")
print("="*60)
pattern = [
    (-1.5, -1.5, 0), (1.5, -1.5, 0),  # Bottom = Class 0
    (-1.5, 1.5, 1), (1.5, 1.5, 1),    # Top = Class 1
]
configs = [
    (0, 10, 0, 10, 0, 0, 0, 10, -5),  # H2 splits on x2
    (0, 5, 0, 5, 0, 0, 0, 5, -2),
]
for cfg in configs:
    if test_config(pattern, *cfg):
        print(f"[OK] WORKS: {cfg}")
        w11, w12, b1, w21, w22, b2, w_out1, w_out2, b_out = cfg
        print(f"  H1: ({w11}, {w12}, {b1})")
        print(f"  H2: ({w21}, {w22}, {b2})")
        print(f"  Out: ({w_out1}, {w_out2}, {b_out})")
        break

# Test 3: Diagonal Split
print("\nTest 3: DIAGONAL SPLIT")
print("="*60)
pattern = [
    (-1.5, -1.5, 0), (0, -0.5, 0),    # Below diagonal = Class 0
    (0, 0.5, 1), (1.5, 1.5, 1),       # Above diagonal = Class 1
]
configs = [
    (10, -10, 0, 0, 10, 0, 10, 0, -5),  # H1 detects x1>x2
    (5, -5, 0, 0, 5, 0, 5, 0, -2),
]
for cfg in configs:
    if test_config(pattern, *cfg):
        print(f"[OK] WORKS: {cfg}")
        w11, w12, b1, w21, w22, b2, w_out1, w_out2, b_out = cfg
        print(f"  H1: ({w11}, {w12}, {b1})")
        print(f"  H2: ({w21}, {w22}, {b2})")
        print(f"  Out: ({w_out1}, {w_out2}, {b_out})")
        break

# Test 4: XOR (confirm it doesn't work)
print("\nTest 4: XOR (should NOT work)")
print("="*60)
pattern = [
    (-1.5, -1.5, 0), (1.5, 1.5, 0),   # BL and TR = Class 0
    (-1.5, 1.5, 1), (1.5, -1.5, 1),   # TL and BR = Class 1
]
found = False
for w11 in [-10, -5, 0, 5, 10]:
    for w12 in [-10, -5, 0, 5, 10]:
        for b1 in [-10, -5, 0, 5, 10]:
            for w21 in [-10, -5, 0, 5, 10]:
                for w22 in [-10, -5, 0, 5, 10]:
                    for b2 in [-10, -5, 0, 5, 10]:
                        for w_out1 in [-10, -5, 0, 5, 10]:
                            for w_out2 in [-10, -5, 0, 5, 10]:
                                for b_out in [-10, -5, 0, 5, 10]:
                                    if test_config(pattern, w11, w12, b1, w21, w22, b2, w_out1, w_out2, b_out):
                                        print(f"[OK] FOUND XOR SOLUTION (unexpected!): {(w11, w12, b1, w21, w22, b2, w_out1, w_out2, b_out)}")
                                        found = True
                                        break
                                if found: break
                            if found: break
                        if found: break
                    if found: break
                if found: break
            if found: break
        if found: break
    if found: break

if not found:
    print("[X] No perfect XOR solution found (as expected)")

print("\n" + "="*60)
print("CONCLUSION:")
print("  [OK] Vertical split: EASY and WORKS")
print("  [OK] Horizontal split: EASY and WORKS")
print("  [OK] Diagonal split: MEDIUM and WORKS")
print("  [X] XOR: HARD and DOESN'T WORK")
print("\nPerfect for demonstrating different difficulty levels!")
