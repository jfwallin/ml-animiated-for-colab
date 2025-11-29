"""
Add Section 8 multiple runs cells to Module 4 notebook (Breast Cancer)
"""
import json
import sys

# Load the notebook
with open('lab_4_module_4_breast_cancer_classification.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Loaded notebook with {len(nb['cells'])} cells")

# Find where to insert Section 8 - should be after confusion matrix visualizations
# and before "Section 7: Record Your Experiments"
insert_idx = None
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        source = ''.join(cell['source'])
        if 'Section 7: Record Your Experiments' in source:
            insert_idx = i
            print(f"Found 'Section 7: Record Your Experiments' at index {i}")
            break

if insert_idx is None:
    print("Could not find insertion point!")
    sys.exit(1)

# Create the new cells for Section 8
new_cells = []

# Section 8 header
section8_header = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "\n",
        "## Section 7: Understanding Variability - Running Multiple Experiments\n",
        "\n",
        "### Why Run Multiple Experiments?\n",
        "\n",
        "**Remember from Module 3:** Neural networks are stochastic!\n",
        "- Random weight initialization leads to different results each run\n",
        "- Professional ML practice: run multiple times, report mean ± std\n",
        "- This is especially important when comparing models\n",
        "\n",
        "**Let's apply this to medical diagnosis:**\n",
        "- Is the baseline model consistently good?\n",
        "- Do hidden layers reliably improve performance?\n",
        "- How much do false negatives (missed cancers) vary?\n",
        "\n",
        "---"
    ]
}
new_cells.append(section8_header)

# Markdown: Baseline experiment header
markdown_baseline = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### Experiment: Baseline Model with Multiple Runs\n",
        "\n",
        "Let's run the baseline linear model **5 times** and analyze the variability."
    ]
}
new_cells.append(markdown_baseline)

# Code: Run baseline multiple times
code_baseline = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Run baseline model multiple times\n",
        "num_runs = 5\n",
        "num_epochs = 100\n",
        "\n",
        "print(\"=\"*70)\n",
        "print(f\"RUNNING BASELINE LINEAR MODEL {num_runs} TIMES\")\n",
        "print(\"=\"*70)\n",
        "print(\"\\nWhy? To see how stable the model is across different initializations!\\n\")\n",
        "\n",
        "# Store results\n",
        "baseline_accuracies = []\n",
        "baseline_losses = []\n",
        "baseline_false_negatives = []  # Missed cancers - most critical!\n",
        "baseline_histories = []\n",
        "\n",
        "for run in range(num_runs):\n",
        "    print(f\"Run {run+1}/{num_runs}...\", end=\" \")\n",
        "    \n",
        "    # Create NEW model for fresh initialization\n",
        "    model = Sequential([\n",
        "        Dense(1, activation='sigmoid', input_dim=30, name='output')\n",
        "    ], name=f'Baseline_Run_{run+1}')\n",
        "    \n",
        "    model.compile(\n",
        "        optimizer='adam',\n",
        "        loss='binary_crossentropy',\n",
        "        metrics=['accuracy']\n",
        "    )\n",
        "    \n",
        "    # Train\n",
        "    history = model.fit(\n",
        "        X_train_scaled, y_train,\n",
        "        epochs=num_epochs,\n",
        "        batch_size=32,\n",
        "        validation_split=0.2,\n",
        "        verbose=0\n",
        "    )\n",
        "    \n",
        "    # Evaluate\n",
        "    test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)\n",
        "    \n",
        "    # Get predictions and confusion matrix\n",
        "    y_pred = (model.predict(X_test_scaled, verbose=0) > 0.5).astype(int).flatten()\n",
        "    cm = confusion_matrix(y_test, y_pred)\n",
        "    false_negatives = cm[0, 1]  # Malignant predicted as benign\n",
        "    \n",
        "    # Store\n",
        "    baseline_accuracies.append(test_acc)\n",
        "    baseline_losses.append(test_loss)\n",
        "    baseline_false_negatives.append(false_negatives)\n",
        "    baseline_histories.append(history)\n",
        "    \n",
        "    print(f\"Acc: {test_acc:.1%}, Loss: {test_loss:.4f}, FN: {false_negatives}\")\n",
        "\n",
        "print(\"\\n\" + \"=\"*70)\n",
        "print(\"BASELINE MODEL - STATISTICAL SUMMARY\")\n",
        "print(\"=\"*70)\n",
        "print(f\"Test Accuracy:        {np.mean(baseline_accuracies):.1%} ± {np.std(baseline_accuracies):.1%}\")\n",
        "print(f\"Test Loss:            {np.mean(baseline_losses):.4f} ± {np.std(baseline_losses):.4f}\")\n",
        "print(f\"False Negatives:      {np.mean(baseline_false_negatives):.1f} ± {np.std(baseline_false_negatives):.2f}\")\n",
        "print(f\"  (Missed cancers - MOST CRITICAL METRIC!)\")\n",
        "print(f\"\\nAccuracy Range:       {np.min(baseline_accuracies):.1%} to {np.max(baseline_accuracies):.1%}\")\n",
        "print(f\"FN Range:             {int(np.min(baseline_false_negatives))} to {int(np.max(baseline_false_negatives))}\")\n",
        "print(\"=\"*70)\n",
        "\n",
        "if np.std(baseline_accuracies) < 0.02:\n",
        "    print(\"\\n✅ Very consistent results - stable baseline!\")\n",
        "elif np.std(baseline_accuracies) < 0.05:\n",
        "    print(\"\\n✓ Reasonably consistent results\")\n",
        "else:\n",
        "    print(\"\\n⚠️ High variability - results depend on initialization\")"
    ]
}
new_cells.append(code_baseline)

# Markdown: Custom model experiment
markdown_custom = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### Experiment: Custom Model with Multiple Runs\n",
        "\n",
        "Now let's run your custom model (with hidden layers) multiple times."
    ]
}
new_cells.append(markdown_custom)

# Code: Run custom model multiple times
code_custom = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Adjustable parameters - CHANGE THESE!\n",
        "num_hidden_layers_exp = 1   # Try: 0, 1, 2\n",
        "units_per_layer_exp = 16    # Try: 8, 16, 32\n",
        "\n",
        "print(\"=\"*70)\n",
        "print(f\"RUNNING CUSTOM MODEL ({num_hidden_layers_exp} layers, {units_per_layer_exp} units) {num_runs} TIMES\")\n",
        "print(\"=\"*70)\n",
        "print()\n",
        "\n",
        "# Store results\n",
        "custom_accuracies = []\n",
        "custom_losses = []\n",
        "custom_false_negatives = []\n",
        "custom_histories = []\n",
        "\n",
        "for run in range(num_runs):\n",
        "    print(f\"Run {run+1}/{num_runs}...\", end=\" \")\n",
        "    \n",
        "    # Build model\n",
        "    model = Sequential(name=f'Custom_Run_{run+1}')\n",
        "    \n",
        "    # Add hidden layers\n",
        "    for i in range(num_hidden_layers_exp):\n",
        "        if i == 0:\n",
        "            model.add(Dense(units_per_layer_exp, activation='relu', input_dim=30, name=f'hidden_{i+1}'))\n",
        "        else:\n",
        "            model.add(Dense(units_per_layer_exp, activation='relu', name=f'hidden_{i+1}'))\n",
        "    \n",
        "    # Output layer\n",
        "    if num_hidden_layers_exp == 0:\n",
        "        model.add(Dense(1, activation='sigmoid', input_dim=30, name='output'))\n",
        "    else:\n",
        "        model.add(Dense(1, activation='sigmoid', name='output'))\n",
        "    \n",
        "    model.compile(\n",
        "        optimizer='adam',\n",
        "        loss='binary_crossentropy',\n",
        "        metrics=['accuracy']\n",
        "    )\n",
        "    \n",
        "    # Train\n",
        "    history = model.fit(\n",
        "        X_train_scaled, y_train,\n",
        "        epochs=num_epochs,\n",
        "        batch_size=32,\n",
        "        validation_split=0.2,\n",
        "        verbose=0\n",
        "    )\n",
        "    \n",
        "    # Evaluate\n",
        "    test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)\n",
        "    \n",
        "    # Confusion matrix\n",
        "    y_pred = (model.predict(X_test_scaled, verbose=0) > 0.5).astype(int).flatten()\n",
        "    cm = confusion_matrix(y_test, y_pred)\n",
        "    false_negatives = cm[0, 1]\n",
        "    \n",
        "    # Store\n",
        "    custom_accuracies.append(test_acc)\n",
        "    custom_losses.append(test_loss)\n",
        "    custom_false_negatives.append(false_negatives)\n",
        "    custom_histories.append(history)\n",
        "    \n",
        "    print(f\"Acc: {test_acc:.1%}, Loss: {test_loss:.4f}, FN: {false_negatives}\")\n",
        "\n",
        "print(\"\\n\" + \"=\"*70)\n",
        "print(f\"CUSTOM MODEL ({num_hidden_layers_exp} layers, {units_per_layer_exp} units) - STATISTICAL SUMMARY\")\n",
        "print(\"=\"*70)\n",
        "print(f\"Test Accuracy:        {np.mean(custom_accuracies):.1%} ± {np.std(custom_accuracies):.1%}\")\n",
        "print(f\"Test Loss:            {np.mean(custom_losses):.4f} ± {np.std(custom_losses):.4f}\")\n",
        "print(f\"False Negatives:      {np.mean(custom_false_negatives):.1f} ± {np.std(custom_false_negatives):.2f}\")\n",
        "print(f\"\\nAccuracy Range:       {np.min(custom_accuracies):.1%} to {np.max(custom_accuracies):.1%}\")\n",
        "print(f\"FN Range:             {int(np.min(custom_false_negatives))} to {int(np.max(custom_false_negatives))}\")\n",
        "print(\"=\"*70)"
    ]
}
new_cells.append(code_custom)

# Markdown: Statistical comparison
markdown_comparison = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### Statistical Comparison: Baseline vs Custom Model"
    ]
}
new_cells.append(markdown_comparison)

# Code: Comparison
code_comparison = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Statistical comparison\n",
        "print(\"=\"*70)\n",
        "print(\"STATISTICAL COMPARISON: BASELINE vs CUSTOM MODEL\")\n",
        "print(\"=\"*70)\n",
        "\n",
        "print(f\"\\nBaseline (no hidden layers):\")\n",
        "print(f\"  Accuracy:         {np.mean(baseline_accuracies):.1%} ± {np.std(baseline_accuracies):.1%}\")\n",
        "print(f\"  False Negatives:  {np.mean(baseline_false_negatives):.1f} ± {np.std(baseline_false_negatives):.2f}\")\n",
        "\n",
        "print(f\"\\nCustom Model ({num_hidden_layers_exp} layers, {units_per_layer_exp} units):\")\n",
        "print(f\"  Accuracy:         {np.mean(custom_accuracies):.1%} ± {np.std(custom_accuracies):.1%}\")\n",
        "print(f\"  False Negatives:  {np.mean(custom_false_negatives):.1f} ± {np.std(custom_false_negatives):.2f}\")\n",
        "\n",
        "acc_improvement = np.mean(custom_accuracies) - np.mean(baseline_accuracies)\n",
        "fn_improvement = np.mean(baseline_false_negatives) - np.mean(custom_false_negatives)\n",
        "\n",
        "print(f\"\\nMean Accuracy Improvement:  {acc_improvement:+.1%}\")\n",
        "print(f\"Mean FN Reduction:          {fn_improvement:+.1f} (positive = fewer missed cancers)\")\n",
        "print(\"=\"*70)\n",
        "\n",
        "# Box plots\n",
        "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=100)\n",
        "\n",
        "# Accuracy box plot\n",
        "box_data_acc = [baseline_accuracies, custom_accuracies]\n",
        "labels = ['Baseline\\n(0 layers)', f'Custom\\n({num_hidden_layers_exp}L, {units_per_layer_exp}U)']\n",
        "\n",
        "bp1 = ax1.boxplot(box_data_acc, labels=labels, patch_artist=True, showmeans=True, meanline=True)\n",
        "for patch, color in zip(bp1['boxes'], ['lightblue', 'lightgreen']):\n",
        "    patch.set_facecolor(color)\n",
        "\n",
        "ax1.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')\n",
        "ax1.set_title(f'Accuracy Distribution ({num_runs} runs)', fontsize=13, fontweight='bold')\n",
        "ax1.grid(True, alpha=0.3, axis='y')\n",
        "\n",
        "# Add points\n",
        "for i, data in enumerate(box_data_acc, 1):\n",
        "    x = np.random.normal(i, 0.04, size=len(data))\n",
        "    ax1.scatter(x, data, alpha=0.6, s=60, c='red', edgecolors='black', linewidths=1)\n",
        "\n",
        "# False Negatives box plot\n",
        "box_data_fn = [baseline_false_negatives, custom_false_negatives]\n",
        "\n",
        "bp2 = ax2.boxplot(box_data_fn, labels=labels, patch_artist=True, showmeans=True, meanline=True)\n",
        "for patch, color in zip(bp2['boxes'], ['lightcoral', 'lightgreen']):\n",
        "    patch.set_facecolor(color)\n",
        "\n",
        "ax2.set_ylabel('False Negatives (Missed Cancers)', fontsize=12, fontweight='bold')\n",
        "ax2.set_title(f'False Negatives Distribution ({num_runs} runs)', fontsize=13, fontweight='bold')\n",
        "ax2.grid(True, alpha=0.3, axis='y')\n",
        "\n",
        "# Add points\n",
        "for i, data in enumerate(box_data_fn, 1):\n",
        "    x = np.random.normal(i, 0.04, size=len(data))\n",
        "    ax2.scatter(x, data, alpha=0.6, s=60, c='red', edgecolors='black', linewidths=1)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print(\"\\n💡 Key Questions:\")\n",
        "print(\"   - Is the custom model CONSISTENTLY better?\")\n",
        "print(\"   - Do the boxes overlap? (If yes, improvement may not be reliable)\")\n",
        "print(\"   - Which metric matters more: accuracy or false negatives?\")\n",
        "print(\"   - Would you trust this model for medical diagnosis?\")"
    ]
}
new_cells.append(code_comparison)

# Markdown: Visualize training curves
markdown_curves = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### Visualize All Training Runs"
    ]
}
new_cells.append(markdown_curves)

# Code: Visualize curves
code_curves = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Plot all training curves together\n",
        "fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10), dpi=100)\n",
        "\n",
        "# Baseline accuracy\n",
        "for i, history in enumerate(baseline_histories):\n",
        "    ax1.plot(history.history['val_accuracy'], alpha=0.6, linewidth=2, label=f'Run {i+1}')\n",
        "ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')\n",
        "ax1.set_ylabel('Validation Accuracy', fontsize=11, fontweight='bold')\n",
        "ax1.set_title('Baseline: Validation Accuracy', fontsize=12, fontweight='bold')\n",
        "ax1.legend(fontsize=9)\n",
        "ax1.grid(True, alpha=0.3)\n",
        "\n",
        "# Baseline loss\n",
        "for i, history in enumerate(baseline_histories):\n",
        "    ax2.plot(history.history['val_loss'], alpha=0.6, linewidth=2, label=f'Run {i+1}')\n",
        "ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')\n",
        "ax2.set_ylabel('Validation Loss', fontsize=11, fontweight='bold')\n",
        "ax2.set_title('Baseline: Validation Loss', fontsize=12, fontweight='bold')\n",
        "ax2.legend(fontsize=9)\n",
        "ax2.grid(True, alpha=0.3)\n",
        "\n",
        "# Custom accuracy\n",
        "for i, history in enumerate(custom_histories):\n",
        "    ax3.plot(history.history['val_accuracy'], alpha=0.6, linewidth=2, label=f'Run {i+1}')\n",
        "ax3.set_xlabel('Epoch', fontsize=11, fontweight='bold')\n",
        "ax3.set_ylabel('Validation Accuracy', fontsize=11, fontweight='bold')\n",
        "ax3.set_title(f'Custom ({num_hidden_layers_exp}L, {units_per_layer_exp}U): Validation Accuracy', \n",
        "             fontsize=12, fontweight='bold')\n",
        "ax3.legend(fontsize=9)\n",
        "ax3.grid(True, alpha=0.3)\n",
        "\n",
        "# Custom loss\n",
        "for i, history in enumerate(custom_histories):\n",
        "    ax4.plot(history.history['val_loss'], alpha=0.6, linewidth=2, label=f'Run {i+1}')\n",
        "ax4.set_xlabel('Epoch', fontsize=11, fontweight='bold')\n",
        "ax4.set_ylabel('Validation Loss', fontsize=11, fontweight='bold')\n",
        "ax4.set_title(f'Custom ({num_hidden_layers_exp}L, {units_per_layer_exp}U): Validation Loss', \n",
        "             fontsize=12, fontweight='bold')\n",
        "ax4.legend(fontsize=9)\n",
        "ax4.grid(True, alpha=0.3)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print(\"\\n💡 What to observe:\")\n",
        "print(\"   - Do all runs converge to similar final values?\")\n",
        "print(\"   - Are there any outlier runs?\")\n",
        "print(\"   - Does one model show more variability than the other?\")"
    ]
}
new_cells.append(code_curves)

# Markdown: Key insights
markdown_insights = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### Key Insights: Stochasticity in Medical ML\n",
        "\n",
        "**What you should have learned:**\n",
        "\n",
        "1. **Results vary even on the same data!**\n",
        "   - Random initialization affects final performance\n",
        "   - Some runs may miss more cancers than others\n",
        "   - This is why clinical ML systems need extensive validation\n",
        "\n",
        "2. **Statistical reporting is essential in medicine**\n",
        "   - Reporting \"97% accuracy\" from one run is misleading\n",
        "   - \"97.2% ± 0.8%\" gives a true picture of reliability\n",
        "   - Variability in false negatives is critical - lives are at stake!\n",
        "\n",
        "3. **Model comparison requires statistics**\n",
        "   - If the boxes overlap significantly, the \"improvement\" may not be real\n",
        "   - Need statistical tests (t-test, etc.) to confirm differences\n",
        "   - In medicine, reproducibility is paramount\n",
        "\n",
        "4. **Simpler models can be more stable**\n",
        "   - Baseline model may have lower variance than complex models\n",
        "   - Trade-off: slightly lower mean accuracy but more consistent\n",
        "   - Consistency matters in clinical deployment\n",
        "\n",
        "**Medical Ethics Connection:**\n",
        "- If your model misses 2-4 cancers depending on initialization, that's a problem!\n",
        "- Real medical ML systems:\n",
        "  - Train on much larger datasets (thousands to millions of samples)\n",
        "  - Use ensemble methods (combine multiple models)\n",
        "  - Undergo extensive clinical trials\n",
        "  - Are validated on diverse patient populations\n",
        "\n",
        "**Professional Practice:**\n",
        "- Always run at least 5-10 times (ideally more)\n",
        "- Report mean ± std for all metrics\n",
        "- Show distributions (box plots, violin plots)\n",
        "- Consider worst-case runs, not just average\n",
        "\n",
        "---"
    ]
}
new_cells.append(markdown_insights)

print(f"\nPrepared {len(new_cells)} new cells to insert at index {insert_idx}")

# Insert all cells
for i, cell in enumerate(new_cells):
    nb['cells'].insert(insert_idx + i, cell)

print(f"Inserted cells at index {insert_idx}")

# Now renumber the subsequent sections
# Section 7: Record Your Experiments -> Section 8
# Section 8: Connection to Earlier Labs -> Section 9
# Section 9: Key Takeaways -> Section 10 (and add stochasticity item)

for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        source_text = ''.join(cell['source'])

        if '## Section 7: Record Your Experiments' in source_text:
            cell['source'] = [s.replace('## Section 7:', '## Section 8:') for s in cell['source']]
            print("Renumbered Section 7 -> Section 8 (Record Your Experiments)")

        elif '## Section 8: Connection to Earlier Labs' in source_text:
            cell['source'] = [s.replace('## Section 8:', '## Section 9:') for s in cell['source']]
            print("Renumbered Section 8 -> Section 9 (Connection to Earlier Labs)")

        elif '## Section 9: Key Takeaways' in source_text:
            # Renumber and add stochasticity takeaway
            new_source = []
            for s in cell['source']:
                new_source.append(s.replace('## Section 9:', '## Section 10:'))

            # Find where to insert new takeaway (after item 6)
            for i, line in enumerate(new_source):
                if '### 6. ML is Stochastic' in line or '### 6. Diminishing Returns' in line:
                    # Find next section or end
                    j = i + 1
                    while j < len(new_source) and not new_source[j].startswith('###') and not new_source[j].startswith('---'):
                        j += 1

                    # Check if we need to add the stochasticity section
                    if 'stochastic' not in ''.join(new_source).lower():
                        new_takeaway = [
                            "\n",
                            "### 7. Medical ML Requires Statistical Rigor\n",
                            "- Variability in predictions affects patient outcomes\n",
                            "- Always report mean ± std for clinical metrics\n",
                            "- Worst-case performance matters as much as average\n",
                            "- Ensemble methods can reduce variability\n",
                            "\n"
                        ]
                        new_source = new_source[:j] + new_takeaway + new_source[j:]
                    break

            cell['source'] = new_source
            print("Renumbered Section 9 -> Section 10 (Key Takeaways) and added stochasticity item")

# Save
with open('lab_4_module_4_breast_cancer_classification.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\nSuccessfully updated Module 4!")
print(f"Total cells now: {len(nb['cells'])}")
print("Saved to lab_4_module_4_breast_cancer_classification.ipynb")
