import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def test_pattern(name, points, description):
    """Test if a pattern can be solved with 2-2-1 network."""
    print("\n" + "="*70)
    print(f"Testing: {name}")
    print(f"Description: {description}")
    print("="*70)

    # Visualize the pattern
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot original pattern
    ax = axes[0]
    for label in [0, 1]:
        mask = [p[2] == label for p in points]
        pts = [p for p, m in zip(points, mask) if m]
        if pts:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            color = 'blue' if label == 0 else 'red'
            ax.scatter(xs, ys, c=color, s=100, alpha=0.7, edgecolors='k', linewidths=1.5)
    ax.set_xlabel('x₁', fontsize=12, fontweight='bold')
    ax.set_ylabel('x₂', fontsize=12, fontweight='bold')
    ax.set_title('Original Pattern', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Search for solution
    print("\nSearching for solution (this may take a moment)...")
    best_acc = 0
    best_config = None
    solutions = []

    # Search with reasonable parameter range
    search_range = range(-10, 11, 2)  # Coarser search first

    for w11 in search_range:
        for w12 in search_range:
            for b1 in search_range:
                for w21 in search_range:
                    for w22 in search_range:
                        for b2 in search_range:
                            for w_out1 in search_range:
                                for w_out2 in search_range:
                                    for b_out in search_range:
                                        correct = 0
                                        for x1, x2, target in points:
                                            h1 = sigmoid(w11*x1 + w12*x2 + b1)
                                            h2 = sigmoid(w21*x1 + w22*x2 + b2)
                                            z_out = w_out1*h1 + w_out2*h2 + b_out
                                            out = sigmoid(z_out)
                                            pred = 1 if out > 0.5 else 0
                                            if pred == target:
                                                correct += 1

                                        acc_pct = (correct / len(points)) * 100
                                        if acc_pct > best_acc:
                                            best_acc = acc_pct
                                            best_config = (w11, w12, b1, w21, w22, b2, w_out1, w_out2, b_out)

                                        if acc_pct == 100 and len(solutions) < 3:
                                            solutions.append(best_config)
                                            print(f"  Found solution #{len(solutions)}: {best_config}")

    print(f"\nBest accuracy found: {best_acc:.1f}%")

    if solutions:
        print(f"\n✓ PATTERN IS SOLVABLE! Found {len(solutions)} perfect solution(s)")
        config = solutions[0]
        w11, w12, b1, w21, w22, b2, w_out1, w_out2, b_out = config

        print(f"\nExample solution:")
        print(f"  H1: w₁₁={w11}, w₁₂={w12}, b₁={b1}")
        print(f"  H2: w₂₁={w21}, w₂₂={w22}, b₂={b2}")
        print(f"  Out: w_out1={w_out1}, w_out2={w_out2}, b_out={b_out}")

        # Show hidden space mapping
        print(f"\nHidden space mapping:")
        ax = axes[1]
        for x1, x2, target in points:
            h1 = sigmoid(w11*x1 + w12*x2 + b1)
            h2 = sigmoid(w21*x1 + w22*x2 + b2)
            color = 'blue' if target == 0 else 'red'
            ax.scatter(h1, h2, c=color, s=100, alpha=0.7, edgecolors='k', linewidths=1.5)

        # Draw decision boundary in hidden space
        if abs(w_out2) > 0.01:
            h1_line = np.linspace(-0.1, 1.1, 100)
            h2_line = -(w_out1 * h1_line + b_out) / w_out2
            valid = (h2_line >= -0.1) & (h2_line <= 1.1)
            ax.plot(h1_line[valid], h2_line[valid], 'g-', linewidth=3, label='Decision Line')

        ax.set_xlabel('h₁', fontsize=12, fontweight='bold')
        ax.set_ylabel('h₂', fontsize=12, fontweight='bold')
        ax.set_title('Hidden Space', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect('equal')
        ax.legend()

        plt.tight_layout()
        plt.savefig(f'pattern_{name.replace(" ", "_")}.png', dpi=100)
        print(f"\nSaved visualization to pattern_{name.replace(' ', '_')}.png")

        return True, config
    else:
        print(f"\n✗ PATTERN NOT PERFECTLY SOLVABLE with 2-2-1 network")
        print(f"   Best achievable: {best_acc:.1f}% accuracy")
        print(f"   Best config: {best_config}")

        plt.close()
        return False, None

# Test Pattern 1: Diagonal Separation (Opposite Corners - NOT XOR)
print("\n" + "#"*70)
print("# TESTING CANDIDATE PATTERNS FOR MODULE 1")
print("#"*70)

pattern1 = [
    # Class 0: Top-left and Bottom-left (same side)
    (-1.5, -1.5, 0),
    (-1.5, 1.5, 0),
    # Class 1: Top-right and Bottom-right (same side)
    (1.5, -1.5, 1),
    (1.5, 1.5, 1),
]
test_pattern("Vertical Split", pattern1, "Two classes separated by vertical line")

# Test Pattern 2: Top vs Bottom
pattern2 = [
    # Class 0: Bottom (both corners)
    (-1.5, -1.5, 0),
    (1.5, -1.5, 0),
    # Class 1: Top (both corners)
    (-1.5, 1.5, 1),
    (1.5, 1.5, 1),
]
test_pattern("Horizontal Split", pattern2, "Two classes separated by horizontal line")

# Test Pattern 3: Diagonal Line (BL+TR vs TL+BR but with clear diagonal)
pattern3 = [
    # Class 0: Below diagonal
    (-1.5, -1.5, 0),
    (0, -0.5, 0),
    # Class 1: Above diagonal
    (0, 0.5, 1),
    (1.5, 1.5, 1),
]
test_pattern("Diagonal Split", pattern3, "Two classes separated by diagonal line")

# Test Pattern 4: Three corners vs one corner
pattern4 = [
    # Class 0: Three corners
    (-1.5, -1.5, 0),
    (1.5, -1.5, 0),
    (1.5, 1.5, 0),
    # Class 1: One corner
    (-1.5, 1.5, 1),
]
test_pattern("Three vs One", pattern4, "Three corners (Class 0) vs one corner (Class 1)")

# Test Pattern 5: Confirm XOR doesn't work
pattern5 = [
    (-1.5, -1.5, 0),  # BL - Class 0
    (1.5, 1.5, 0),    # TR - Class 0
    (-1.5, 1.5, 1),   # TL - Class 1
    (1.5, -1.5, 1),   # BR - Class 1
]
test_pattern("XOR (control)", pattern5, "XOR pattern - should NOT be solvable")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("\nPatterns we can use for Module 1:")
print("  1. Vertical Split - SIMPLE, easy to understand")
print("  2. Horizontal Split - SIMPLE, easy to understand")
print("  3. Diagonal Split - MEDIUM difficulty")
print("  4. Three vs One - INTERESTING asymmetric pattern")
print("  5. XOR - Use as 'hard case' that doesn't perfectly solve")
print("\nRecommendation: Use Vertical or Horizontal split as main demo,")
print("then show XOR as 'some patterns need more complex networks'")
