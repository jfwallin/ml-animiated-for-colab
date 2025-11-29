import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

# XOR pattern
points = {
    'BL': (-1.5, -1.5, 0),  # Class 0
    'TR': (1.5, 1.5, 0),    # Class 0
    'TL': (-1.5, 1.5, 1),   # Class 1
    'BR': (1.5, -1.5, 1),   # Class 1
}

print("XOR Pattern Analysis")
print("="*70)
print("\nOriginal XOR in 2D:")
for name, (x1, x2, label) in points.items():
    print(f"  {name}: ({x1:+.1f}, {x2:+.1f}) -> Class {label}")

print("\n" + "="*70)
print("Goal in Hidden Space:")
print("We need the 4 points to map to hidden (h1, h2) such that")
print("a SINGLE straight line can separate Class 0 from Class 1.")
print("="*70)

print("\nPossible hidden mappings:")
print("\nOption 1: Map to 4 corners of unit square")
print("  BL(0) -> (0,0)   TR(0) -> (1,1)")
print("  TL(1) -> (0,1)   BR(1) -> (1,0)")
print("  Decision line: h1 = h2 separates (0,1) and (1,0) from (0,0) and (1,1)")
print("  Problem: (0,1) and (1,0) are on OPPOSITE sides of h1=h2 line!")
print("  CANNOT work with single line.")

print("\nOption 2: Map to diagonal line")
print("  BL(0) -> (0,0)   TR(0) -> (1,1)  [both on h1=h2 diagonal]")
print("  TL(1) -> (0,1)   BR(1) -> (1,0)  [both OFF the diagonal]")
print("  Decision line: h1 = h2 +/- epsilon")
print("  Works if both TL and BR are on SAME side of diagonal")

print("\nLet me find config where TL and BR map to same side...")
print("\nTrying: H1 detects (x1-x2 > threshold), H2 detects (x1+x2 > threshold)")

# Try this configuration
w11, w12, b1 = 5, -5, 0   # H1: x1-x2 > 0
w21, w22, b2 = 5, 5, 0     # H2: x1+x2 > 0

print(f"\nH1: ({w11}, {w12}, {b1}) detects x1-x2>0")
print(f"H2: ({w21}, {w22}, {b2}) detects x1+x2>0")
print("\nHidden mapping:")
for name, (x1, x2, label) in points.items():
    h1 = sigmoid(w11*x1 + w12*x2 + b1)
    h2 = sigmoid(w21*x1 + w22*x2 + b2)
    print(f"  {name}({label}): ({x1:+.1f}, {x2:+.1f}) -> h=({h1:.3f}, {h2:.3f})")

# Visualize hidden space
fig, ax = plt.subplots(figsize=(8, 8))

for name, (x1, x2, label) in points.items():
    h1 = sigmoid(w11*x1 + w12*x2 + b1)
    h2 = sigmoid(w21*x1 + w22*x2 + b2)
    color = 'blue' if label == 0 else 'red'
    ax.scatter(h1, h2, c=color, s=300, alpha=0.7, edgecolors='k', linewidths=2)
    ax.text(h1+0.03, h2+0.03, name, fontsize=12, fontweight='bold')

ax.set_xlabel('h₁', fontsize=14, fontweight='bold')
ax.set_ylabel('h₂', fontsize=14, fontweight='bold')
ax.set_title('XOR in Hidden Space\nH1=(5,-5,0), H2=(5,5,0)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.1, 1.1)
ax.set_aspect('equal')

# Try to draw separating lines
ax.plot([0, 1], [0, 1], 'g--', linewidth=2, label='h₁=h₂', alpha=0.5)
ax.plot([0, 1], [1, 0], 'orange', linestyle='--', linewidth=2, label='h₁+h₂=1', alpha=0.5)

ax.legend()
plt.tight_layout()
plt.savefig('xor_hidden_space.png', dpi=100)
print(f"\nSaved visualization to xor_hidden_space.png")
print("\nNow testing different output layer configs...")

for w_out1 in range(-10, 11):
    for w_out2 in range(-10, 11):
        for b_out in range(-10, 11):
            correct = 0
            for name, (x1, x2, label) in points.items():
                h1 = sigmoid(w11*x1 + w12*x2 + b1)
                h2 = sigmoid(w21*x1 + w22*x2 + b2)
                z_out = w_out1*h1 + w_out2*h2 + b_out
                out = sigmoid(z_out)
                pred = 1 if out > 0.5 else 0
                if pred == label:
                    correct += 1

            if correct == 4:
                print(f"\n*** FOUND SOLUTION ***")
                print(f"H1: ({w11}, {w12}, {b1})")
                print(f"H2: ({w21}, {w22}, {b2})")
                print(f"Out: ({w_out1}, {w_out2}, {b_out})")
                print("\nVerification:")
                for name, (x1, x2, label) in points.items():
                    h1 = sigmoid(w11*x1 + w12*x2 + b1)
                    h2 = sigmoid(w21*x1 + w22*x2 + b2)
                    z_out = w_out1*h1 + w_out2*h2 + b_out
                    out = sigmoid(z_out)
                    pred = 1 if out > 0.5 else 0
                    print(f"  {name}: h=({h1:.3f},{h2:.3f}) -> out={out:.3f} -> {pred} (want {label})")
                break
    if correct == 4:
        break
if correct == 4:
    print("\nDone!")
