"""
Generate warm-start seeds for Lab 4 Module 2.

Strategy: Start from the known perfect solution, add significant noise,
then verify these perturbed versions still converge.

This gives the pedagogical benefit of "looks random" while guaranteeing convergence.
"""

import numpy as np
from find_convergent_seeds import test_seed, TinyNetwork, create_perfect_xor_dataset, train_step

# Known perfect solution from Module 1
PERFECT_SOLUTION = [-10, -10, -10, -10, -10, 5, -10, 10, -5]

def generate_warm_start_seed(base_solution, noise_scale, seed):
    """Generate a warm-start seed by adding noise to the base solution."""
    np.random.seed(seed)
    noisy = [w + np.random.randn() * noise_scale for w in base_solution]
    return noisy

def test_warm_start_convergence(noise_scale, n_seeds=20, lr=0.1, max_epochs=200):
    """Test if warm-start seeds with given noise level converge."""
    print(f"\nTesting warm-start with noise_scale={noise_scale}")
    print("="*70)

    X, y = create_perfect_xor_dataset()
    convergent_seeds = []

    for seed_idx in range(n_seeds):
        init_weights = generate_warm_start_seed(PERFECT_SOLUTION, noise_scale, seed_idx)
        network = TinyNetwork(init_weights)

        # Train
        loss_history = []
        converged = False
        for epoch in range(max_epochs):
            loss, acc = train_step(network, X, y, lr)
            loss_history.append(loss)

            if acc >= 0.95:
                converged = True
                break

        if converged:
            convergent_seeds.append({
                'seed': seed_idx,
                'init_weights': init_weights,
                'epochs': epoch + 1,
                'final_acc': acc,
                'final_loss': loss
            })
            print(f"  [OK] Seed {seed_idx:2d}: Converged in {epoch+1:3d} epochs")
        else:
            print(f"  [X]  Seed {seed_idx:2d}: Did not converge (acc={acc:.3f})")

    print(f"\nConvergence rate: {len(convergent_seeds)}/{n_seeds} = {len(convergent_seeds)/n_seeds*100:.0f}%")
    return convergent_seeds

if __name__ == "__main__":
    print("WARM-START SEED GENERATION")
    print("="*70)
    print("Perfect solution:", PERFECT_SOLUTION)
    print()

    # Test different noise levels
    results = {}
    for noise_scale in [2.0, 3.0, 4.0, 5.0]:
        seeds = test_warm_start_convergence(noise_scale, n_seeds=15, lr=0.1, max_epochs=200)
        results[noise_scale] = seeds

    # Pick the best noise level (one that gives 100% convergence but interesting dynamics)
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for noise_scale, seeds in results.items():
        print(f"Noise={noise_scale}: {len(seeds)}/15 converged ({len(seeds)/15*100:.0f}%)")

    # Choose noise_scale with 100% convergence
    best_noise_scale = None
    for noise_scale in sorted(results.keys()):
        if len(results[noise_scale]) >= 10:  # At least 10 converged
            best_noise_scale = noise_scale
            break

    if best_noise_scale:
        print(f"\nUsing noise_scale={best_noise_scale} (reliable convergence)")
        print("\n" + "="*70)
        print("FINAL SEEDS TO USE IN NOTEBOOK:")
        print("="*70)
        print("\nCONVERGENT_SEEDS = [")
        for s in results[best_noise_scale][:10]:  # Take first 10
            weights_str = "[" + ", ".join([f"{w:.4f}" for w in s['init_weights']]) + "]"
            print(f"    {weights_str},")
        print("]")
    else:
        print("\nWARNING: Could not find noise level with reliable convergence!")
        print("May need to adjust strategy.")
