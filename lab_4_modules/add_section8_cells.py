"""
Add Section 8 multiple runs cells to Module 3 notebook
"""
import json
import sys

# Load the notebook
with open('lab_4_module_3_iris_classification.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find where Section 8 starts
section8_idx = None
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown' and 'Section 8: Understanding Variability' in ''.join(cell['source']):
        section8_idx = i
        break

if section8_idx is None:
    print("Could not find Section 8!")
    sys.exit(1)

print(f"Found Section 8 at index {section8_idx}")

# Find the markdown cell after Section 8 intro
insert_idx = section8_idx + 1
while insert_idx < len(nb['cells']) and nb['cells'][insert_idx]['cell_type'] != 'markdown':
    insert_idx += 1

# If we found a markdown cell, it might be the "Experiment: Linear Model" header
if insert_idx < len(nb['cells']):
    cell_text = ''.join(nb['cells'][insert_idx]['source'])
    if 'Experiment: Linear Model with Multiple Runs' in cell_text:
        insert_idx += 1  # Insert after this header
        print(f"Found experiment header at {insert_idx-1}, will insert code at {insert_idx}")

# Create the new cells to insert
new_cells = []

# Code cell 1: Run linear model multiple times
code_cell_1 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Run linear model multiple times to understand variability\n",
        "num_runs = 5\n",
        "num_epochs = 70\n",
        "\n",
        "print(\"=\"*70)\n",
        "print(f\"RUNNING LINEAR MODEL {num_runs} TIMES\")\n",
        "print(\"=\"*70)\n",
        "print(\"\\nWhy? To see how much results vary due to random initialization!\\n\")\n",
        "\n",
        "# Store results from each run\n",
        "linear_test_accuracies = []\n",
        "linear_test_losses = []\n",
        "linear_histories = []\n",
        "\n",
        "for run in range(num_runs):\n",
        "    print(f\"Run {run+1}/{num_runs}...\", end=\" \")\n",
        "    \n",
        "    # IMPORTANT: Create a NEW model each time!\n",
        "    # This ensures fresh random initialization\n",
        "    linear_model_run = Sequential([\n",
        "        Dense(3, activation='softmax', input_dim=4, name='output')\n",
        "    ], name=f'Linear_Model_Run_{run+1}')\n",
        "    \n",
        "    linear_model_run.compile(\n",
        "        optimizer='adam',\n",
        "        loss='sparse_categorical_crossentropy',\n",
        "        metrics=['accuracy']\n",
        "    )\n",
        "    \n",
        "    # Train the model\n",
        "    history = linear_model_run.fit(\n",
        "        X_train_scaled, y_train,\n",
        "        epochs=num_epochs,\n",
        "        batch_size=16,\n",
        "        validation_split=0.2,\n",
        "        verbose=0\n",
        "    )\n",
        "    \n",
        "    # Evaluate on test set\n",
        "    test_loss, test_accuracy = linear_model_run.evaluate(X_test_scaled, y_test, verbose=0)\n",
        "    \n",
        "    # Store results\n",
        "    linear_test_accuracies.append(test_accuracy)\n",
        "    linear_test_losses.append(test_loss)\n",
        "    linear_histories.append(history)\n",
        "    \n",
        "    print(f\"Accuracy: {test_accuracy:.1%}, Loss: {test_loss:.4f}\")\n",
        "\n",
        "print(\"\\n\" + \"=\"*70)\n",
        "print(\"STATISTICAL SUMMARY - LINEAR MODEL\")\n",
        "print(\"=\"*70)\n",
        "print(f\"Test Accuracy:  {np.mean(linear_test_accuracies):.1%} ± {np.std(linear_test_accuracies):.1%}\")\n",
        "print(f\"Test Loss:      {np.mean(linear_test_losses):.4f} ± {np.std(linear_test_losses):.4f}\")\n",
        "print(f\"\\nMin Accuracy:   {np.min(linear_test_accuracies):.1%}\")\n",
        "print(f\"Max Accuracy:   {np.max(linear_test_accuracies):.1%}\")\n",
        "print(f\"Range:          {np.max(linear_test_accuracies) - np.min(linear_test_accuracies):.1%}\")\n",
        "print(\"=\"*70)\n",
        "\n",
        "print(\"\\n💡 What does this tell us?\")\n",
        "if np.std(linear_test_accuracies) < 0.05:\n",
        "    print(\"   - Low standard deviation → Results are CONSISTENT\")\n",
        "    print(\"   - The model is stable across different initializations\")\n",
        "else:\n",
        "    print(\"   - Higher standard deviation → Results VARY\")\n",
        "    print(\"   - Random initialization matters more for this model\")\n",
        "print(f\"   - We can report: {np.mean(linear_test_accuracies):.1%} ± {np.std(linear_test_accuracies):.1%}\")"
    ]
}

new_cells.append(code_cell_1)

# Markdown cell: Visualize header
markdown_viz = {
    "cell_type": "markdown",
    "metadata": {},
    "source": ["### Visualize All Runs Together"]
}
new_cells.append(markdown_viz)

# Code cell 2: Visualization
code_cell_2 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Plot training curves from all runs on the same plot\n",
        "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=100)\n",
        "\n",
        "# Accuracy plot - all runs\n",
        "for i, history in enumerate(linear_histories):\n",
        "    ax1.plot(history.history['val_accuracy'], alpha=0.6, linewidth=2, label=f'Run {i+1}')\n",
        "\n",
        "ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')\n",
        "ax1.set_ylabel('Validation Accuracy', fontsize=12, fontweight='bold')\n",
        "ax1.set_title(f'Linear Model: {num_runs} Runs - Validation Accuracy', fontsize=13, fontweight='bold')\n",
        "ax1.legend(fontsize=10)\n",
        "ax1.grid(True, alpha=0.3)\n",
        "\n",
        "# Loss plot - all runs\n",
        "for i, history in enumerate(linear_histories):\n",
        "    ax2.plot(history.history['val_loss'], alpha=0.6, linewidth=2, label=f'Run {i+1}')\n",
        "\n",
        "ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')\n",
        "ax2.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')\n",
        "ax2.set_title(f'Linear Model: {num_runs} Runs - Validation Loss', fontsize=13, fontweight='bold')\n",
        "ax2.legend(fontsize=10)\n",
        "ax2.grid(True, alpha=0.3)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print(\"\\n💡 What to observe:\")\n",
        "print(\"   - Do all runs converge to similar accuracy?\")\n",
        "print(\"   - Do some runs converge faster than others?\")\n",
        "print(\"   - Is there a spread in final performance?\")\n",
        "print(\"   - This variability is NORMAL and expected in ML!\")"
    ]
}
new_cells.append(code_cell_2)

# Markdown: Hidden layer experiment header
markdown_hidden = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### Experiment: Hidden Layer Model with Multiple Runs\n",
        "\n",
        "Now let's do the same for the hidden layer model. Do models with more parameters vary more or less?"
    ]
}
new_cells.append(markdown_hidden)

# Code cell 3: Hidden layer multiple runs
code_cell_3 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Adjustable parameter - change this to experiment!\n",
        "hidden_units_experiment = 16  # Try: 8, 16, 32\n",
        "\n",
        "print(\"=\"*70)\n",
        "print(f\"RUNNING HIDDEN LAYER MODEL ({hidden_units_experiment} units) {num_runs} TIMES\")\n",
        "print(\"=\"*70)\n",
        "print()\n",
        "\n",
        "# Store results from each run\n",
        "hidden_test_accuracies = []\n",
        "hidden_test_losses = []\n",
        "hidden_histories = []\n",
        "\n",
        "for run in range(num_runs):\n",
        "    print(f\"Run {run+1}/{num_runs}...\", end=\" \")\n",
        "    \n",
        "    # Create a NEW model each time for fresh initialization\n",
        "    hidden_model_run = Sequential([\n",
        "        Dense(hidden_units_experiment, activation='relu', input_dim=4, name='hidden'),\n",
        "        Dense(3, activation='softmax', name='output')\n",
        "    ], name=f'Hidden_Model_Run_{run+1}')\n",
        "    \n",
        "    hidden_model_run.compile(\n",
        "        optimizer='adam',\n",
        "        loss='sparse_categorical_crossentropy',\n",
        "        metrics=['accuracy']\n",
        "    )\n",
        "    \n",
        "    # Train the model\n",
        "    history = hidden_model_run.fit(\n",
        "        X_train_scaled, y_train,\n",
        "        epochs=num_epochs,\n",
        "        batch_size=16,\n",
        "        validation_split=0.2,\n",
        "        verbose=0\n",
        "    )\n",
        "    \n",
        "    # Evaluate on test set\n",
        "    test_loss, test_accuracy = hidden_model_run.evaluate(X_test_scaled, y_test, verbose=0)\n",
        "    \n",
        "    # Store results\n",
        "    hidden_test_accuracies.append(test_accuracy)\n",
        "    hidden_test_losses.append(test_loss)\n",
        "    hidden_histories.append(history)\n",
        "    \n",
        "    print(f\"Accuracy: {test_accuracy:.1%}, Loss: {test_loss:.4f}\")\n",
        "\n",
        "print(\"\\n\" + \"=\"*70)\n",
        "print(f\"STATISTICAL SUMMARY - HIDDEN LAYER MODEL ({hidden_units_experiment} units)\")\n",
        "print(\"=\"*70)\n",
        "print(f\"Test Accuracy:  {np.mean(hidden_test_accuracies):.1%} ± {np.std(hidden_test_accuracies):.1%}\")\n",
        "print(f\"Test Loss:      {np.mean(hidden_test_losses):.4f} ± {np.std(hidden_test_losses):.4f}\")\n",
        "print(f\"\\nMin Accuracy:   {np.min(hidden_test_accuracies):.1%}\")\n",
        "print(f\"Max Accuracy:   {np.max(hidden_test_accuracies):.1%}\")\n",
        "print(f\"Range:          {np.max(hidden_test_accuracies) - np.min(hidden_test_accuracies):.1%}\")\n",
        "print(\"=\"*70)"
    ]
}
new_cells.append(code_cell_3)

# Markdown: Visualize hidden layer header
markdown_viz_hidden = {
    "cell_type": "markdown",
    "metadata": {},
    "source": ["### Visualize Hidden Layer Model Runs"]
}
new_cells.append(markdown_viz_hidden)

# Code cell 4: Visualize hidden layer runs
code_cell_4 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Plot training curves from all runs\n",
        "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=100)\n",
        "\n",
        "# Accuracy plot - all runs\n",
        "for i, history in enumerate(hidden_histories):\n",
        "    ax1.plot(history.history['val_accuracy'], alpha=0.6, linewidth=2, label=f'Run {i+1}')\n",
        "\n",
        "ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')\n",
        "ax1.set_ylabel('Validation Accuracy', fontsize=12, fontweight='bold')\n",
        "ax1.set_title(f'Hidden Layer Model ({hidden_units_experiment}U): {num_runs} Runs - Validation Accuracy', \n",
        "             fontsize=13, fontweight='bold')\n",
        "ax1.legend(fontsize=10)\n",
        "ax1.grid(True, alpha=0.3)\n",
        "\n",
        "# Loss plot - all runs\n",
        "for i, history in enumerate(hidden_histories):\n",
        "    ax2.plot(history.history['val_loss'], alpha=0.6, linewidth=2, label=f'Run {i+1}')\n",
        "\n",
        "ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')\n",
        "ax2.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')\n",
        "ax2.set_title(f'Hidden Layer Model ({hidden_units_experiment}U): {num_runs} Runs - Validation Loss', \n",
        "             fontsize=13, fontweight='bold')\n",
        "ax2.legend(fontsize=10)\n",
        "ax2.grid(True, alpha=0.3)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ]
}
new_cells.append(code_cell_4)

# Markdown: Final comparison header
markdown_comparison = {
    "cell_type": "markdown",
    "metadata": {},
    "source": ["### Final Comparison: Linear vs Hidden Layer"]
}
new_cells.append(markdown_comparison)

# Code cell 5: Final comparison
code_cell_5 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Statistical comparison\n",
        "print(\"=\"*70)\n",
        "print(\"FINAL COMPARISON: LINEAR vs HIDDEN LAYER (STATISTICAL)\")\n",
        "print(\"=\"*70)\n",
        "print(f\"\\nLinear Model (no hidden layer):\")\n",
        "print(f\"  Mean Accuracy:  {np.mean(linear_test_accuracies):.1%} ± {np.std(linear_test_accuracies):.1%}\")\n",
        "print(f\"  Mean Loss:      {np.mean(linear_test_losses):.4f} ± {np.std(linear_test_losses):.4f}\")\n",
        "\n",
        "print(f\"\\nHidden Layer Model ({hidden_units_experiment} units):\")\n",
        "print(f\"  Mean Accuracy:  {np.mean(hidden_test_accuracies):.1%} ± {np.std(hidden_test_accuracies):.1%}\")\n",
        "print(f\"  Mean Loss:      {np.mean(hidden_test_losses):.4f} ± {np.std(hidden_test_losses):.4f}\")\n",
        "\n",
        "mean_improvement = np.mean(hidden_test_accuracies) - np.mean(linear_test_accuracies)\n",
        "print(f\"\\nMean Improvement: {mean_improvement:+.1%}\")\n",
        "print(\"=\"*70)\n",
        "\n",
        "# Create box plot to visualize distributions\n",
        "fig, ax = plt.subplots(figsize=(10, 6), dpi=100)\n",
        "\n",
        "box_data = [linear_test_accuracies, hidden_test_accuracies]\n",
        "box_labels = ['Linear Model\\n(0 hidden layers)', f'Hidden Layer Model\\n({hidden_units_experiment} units)']\n",
        "\n",
        "bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True,\n",
        "                showmeans=True, meanline=True)\n",
        "\n",
        "# Color the boxes\n",
        "colors = ['lightblue', 'lightgreen']\n",
        "for patch, color in zip(bp['boxes'], colors):\n",
        "    patch.set_facecolor(color)\n",
        "\n",
        "ax.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')\n",
        "ax.set_title(f'Test Accuracy Distribution Across {num_runs} Runs', fontsize=13, fontweight='bold')\n",
        "ax.grid(True, alpha=0.3, axis='y')\n",
        "\n",
        "# Add individual points\n",
        "for i, data in enumerate(box_data, 1):\n",
        "    x = np.random.normal(i, 0.04, size=len(data))\n",
        "    ax.scatter(x, data, alpha=0.6, s=60, c='red', edgecolors='black', linewidths=1)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print(\"\\n💡 Understanding the box plot:\")\n",
        "print(\"   - Box shows the middle 50% of results (interquartile range)\")\n",
        "print(\"   - Line in box is the median\")\n",
        "print(\"   - Green dashed line is the mean\")\n",
        "print(\"   - Red dots are individual run results\")\n",
        "print(\"   - Whiskers show the full range (min to max)\")\n",
        "print(\"\\n   → This is how you should report ML results in research!\")"
    ]
}
new_cells.append(code_cell_5)

# Markdown: Key insights
markdown_insights = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### Key Insights from Multiple Runs\n",
        "\n",
        "**What you should have learned:**\n",
        "\n",
        "1. **ML is stochastic** - Same code, different results each time!\n",
        "   - Random weight initialization matters\n",
        "   - Results vary from run to run\n",
        "\n",
        "2. **Reporting means ± std is standard practice**\n",
        "   - Never trust a single run\n",
        "   - Always run experiments multiple times\n",
        "   - Report: \"Accuracy: 93.2% ± 2.1%\" not \"Accuracy: 95.0%\"\n",
        "\n",
        "3. **Variability can tell you about model stability**\n",
        "   - Low std → Consistent, reliable model\n",
        "   - High std → Sensitive to initialization, may need more tuning\n",
        "\n",
        "4. **Statistical comparison is more robust**\n",
        "   - Comparing single runs: \"95% vs 93%\" (unreliable!)\n",
        "   - Comparing distributions: \"94.5±1.2% vs 92.8±2.3%\" (reliable!)\n",
        "\n",
        "**Connection to Module 2:**\n",
        "- Remember testing multiple random starting points for XOR?\n",
        "- You saw some converged faster, some to different local minima\n",
        "- **Same concept here**, but now you're quantifying it statistically!\n",
        "\n",
        "---"
    ]
}
new_cells.append(markdown_insights)

# Insert all new cells
print(f"Inserting {len(new_cells)} new cells at index {insert_idx}")
for i, cell in enumerate(new_cells):
    nb['cells'].insert(insert_idx + i, cell)

# Update remaining sections to renumber them
# Section 7 -> Section 9 (Connection to Earlier Labs)
# Section 8 -> Section 10 (Key Takeaways - needs to add item about stochasticity)

# Find and update "Section 7: Connection to Earlier Labs"
for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        source = ''.join(cell['source'])
        if '## Section 7: Connection to Earlier Labs' in source:
            cell['source'] = [s.replace('## Section 7:', '## Section 9:') for s in cell['source']]
            print("Renumbered Section 7 to Section 9")
        elif '## Section 8: Key Takeaways' in source:
            # Update to Section 10 and add stochasticity takeaway
            new_source = []
            for s in cell['source']:
                new_source.append(s.replace('## Section 8:', '## Section 10:'))

            # Find where to insert the new takeaway (after item 5)
            for i, line in enumerate(new_source):
                if '### 5. Real Data Has Realistic Accuracy' in line:
                    # Find the end of this section
                    j = i + 1
                    while j < len(new_source) and not new_source[j].startswith('###') and not new_source[j].startswith('---'):
                        j += 1
                    # Insert new takeaway
                    new_takeaway = [
                        "\n",
                        "### 6. ML is Stochastic - Always Run Multiple Experiments!\n",
                        "- Random initialization causes result variability\n",
                        "- Professional ML practice: run multiple times, report mean ± std\n",
                        "- Low variability = stable, reliable model\n",
                        "\n"
                    ]
                    new_source = new_source[:j] + new_takeaway + new_source[j:]
                    break

            cell['source'] = new_source
            print("Renumbered Section 8 to Section 10 and added stochasticity takeaway")

# Save the updated notebook
with open('lab_4_module_3_iris_classification.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\n✓ Successfully added {len(new_cells)} cells to the notebook!")
print(f"✓ Total cells now: {len(nb['cells'])}")
print("✓ Saved to lab_4_module_3_iris_classification.ipynb")
