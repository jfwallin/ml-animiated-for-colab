import json

# Read notebook
with open('lab_4_module_2_training_neural_network.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 12: Markdown - Section 3 header
nb['cells'].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n\n## Section 3: Watch Learning Happen! (8-10 min)\n\nNow let's see gradient descent in action. You'll watch the network **learn** to solve XOR automatically.\n\n### What You'll See:\n- **Left panel**: Decision boundary evolving in real-time\n- **Right panel**: Loss decreasing as the network learns\n- **Training controls**: Step through learning at your own pace"
    ]
})

# Cell 13: Code - Visualization function
nb['cells'].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "def plot_training_state(network, X, y, loss_history, epoch, show_log_scale=True):\n    \"\"\"2-panel visualization: Decision boundary + Loss curve.\"\"\"\n    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))\n    \n    # Left panel: Decision boundary in input space\n    x_min, x_max = -2.5, 2.5\n    y_min, y_max = -2.5, 2.5\n    h = 0.05\n    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),\n                         np.arange(y_min, y_max, h))\n    mesh_points = np.c_[xx.ravel(), yy.ravel()]\n    \n    Z_out, _, _ = network.predict_batch(mesh_points)\n    Z_out = Z_out.reshape(xx.shape)\n    \n    ax1.contourf(xx, yy, Z_out, levels=20, alpha=0.4, cmap='RdBu_r')\n    ax1.contour(xx, yy, Z_out, levels=[0.5], colors='green', linewidths=3)\n    ax1.scatter(X[y==0, 0], X[y==0, 1], c='blue', s=50, alpha=0.7,\n               edgecolors='k', linewidths=1, label='Class 0')\n    ax1.scatter(X[y==1, 0], X[y==1, 1], c='red', s=50, alpha=0.7,\n               edgecolors='k', linewidths=1, label='Class 1')\n    ax1.set_xlim(x_min, x_max)\n    ax1.set_ylim(y_min, y_max)\n    ax1.set_xlabel('x₁', fontsize=12, fontweight='bold')\n    ax1.set_ylabel('x₂', fontsize=12, fontweight='bold')\n    ax1.set_title(f'Decision Boundary (Epoch {epoch})', fontsize=12, fontweight='bold')\n    ax1.legend(loc='upper right')\n    ax1.grid(True, alpha=0.3)\n    ax1.set_aspect('equal')\n    \n    # Right panel: Loss curve\n    if len(loss_history) > 0:\n        ax2.plot(loss_history, 'b-', linewidth=2)\n        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')\n        ax2.set_ylabel('Loss (Binary Cross-Entropy)', fontsize=12, fontweight='bold')\n        ax2.set_title('Training Loss Curve', fontsize=12, fontweight='bold')\n        ax2.grid(True, alpha=0.3)\n        \n        if show_log_scale and len(loss_history) > 5:\n            ax2.set_yscale('log')\n            ax2.set_ylabel('Loss (log scale)', fontsize=12, fontweight='bold')\n    else:\n        ax2.text(0.5, 0.5, 'No training yet...', ha='center', va='center',\n                fontsize=14, transform=ax2.transAxes)\n        ax2.set_xlabel('Epoch', fontsize=12)\n        ax2.set_ylabel('Loss', fontsize=12)\n    \n    plt.tight_layout()\n    plt.show()\n\nprint(\"Visualization function ready!\")"
    ]
})

# Cell 14: Code - Training state and buttons
nb['cells'].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Training state management\ntraining_state = {\n    'network': TinyNetwork(CONVERGENT_SEEDS[0]),  # Start with first seed\n    'epoch': 0,\n    'loss_history': [],\n    'learning_rate': 0.1,\n    'current_seed_idx': 0\n}\n\n# Status display\nstatus_html = HTML(value=\"<h3>Ready to train!</h3>\")\nplot_output = Output()\n\n# Buttons\ntrain_1_btn = Button(\n    description='Train 1 Step',\n    button_style='info',\n    layout=Layout(width='150px', height='40px')\n)\n\ntrain_10_btn = Button(\n    description='Train 10 Steps',\n    button_style='primary',\n    layout=Layout(width='150px', height='40px')\n)\n\ntrain_converge_btn = Button(\n    description='Train to Convergence',\n    button_style='success',\n    layout=Layout(width='180px', height='40px')\n)\n\nreset_btn = Button(\n    description='Reset Network',\n    button_style='warning',\n    layout=Layout(width='150px', height='40px')\n)\n\ndef update_display():\n    \"\"\"Update status and visualization.\"\"\"\n    epoch = training_state['epoch']\n    loss = training_state['loss_history'][-1] if training_state['loss_history'] else compute_loss(training_state['network'], X_train, y_train)\n    acc = compute_accuracy(training_state['network'], X_train, y_train)\n    \n    status_html.value = f\"<h3>Epoch {epoch} | Loss: {loss:.6f} | Accuracy: {acc:.2%}</h3>\"\n    \n    with plot_output:\n        clear_output(wait=True)\n        plot_training_state(training_state['network'], X_train, y_train, \n                          training_state['loss_history'], epoch)\n\ndef train_n_steps(n):\n    \"\"\"Train for n steps.\"\"\"\n    for _ in range(n):\n        loss, acc = train_step(training_state['network'], X_train, y_train, \n                              training_state['learning_rate'])\n        training_state['loss_history'].append(loss)\n        training_state['epoch'] += 1\n        \n        # Early stopping if converged\n        if acc >= 0.99:\n            break\n    \n    update_display()\n\ndef on_train_1(btn):\n    train_n_steps(1)\n\ndef on_train_10(btn):\n    train_n_steps(10)\n\ndef on_train_converge(btn):\n    \"\"\"Train until convergence or max 200 epochs.\"\"\"\n    max_epochs = 200\n    while training_state['epoch'] < max_epochs:\n        loss, acc = train_step(training_state['network'], X_train, y_train,\n                              training_state['learning_rate'])\n        training_state['loss_history'].append(loss)\n        training_state['epoch'] += 1\n        \n        if acc >= 0.99 or loss < 0.01:\n            break\n    \n    update_display()\n\ndef on_reset(btn):\n    \"\"\"Reset to new random seed.\"\"\"\n    training_state['current_seed_idx'] = (training_state['current_seed_idx'] + 1) % len(CONVERGENT_SEEDS)\n    training_state['network'] = TinyNetwork(CONVERGENT_SEEDS[training_state['current_seed_idx']])\n    training_state['epoch'] = 0\n    training_state['loss_history'] = []\n    update_display()\n\n# Connect buttons\ntrain_1_btn.on_click(on_train_1)\ntrain_10_btn.on_click(on_train_10)\ntrain_converge_btn.on_click(on_train_converge)\nreset_btn.on_click(on_reset)\n\nprint(\"Training interface ready!\")"
    ]
})

# Cell 15: Code - Display interface
nb['cells'].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Display the interactive training interface\nprint(\"=\"*70)\nprint(\"INTERACTIVE TRAINING: WATCH GRADIENT DESCENT LEARN!\")\nprint(\"=\"*70)\nprint(\"\\nInstructions:\")\nprint(\"  1. Click 'Train 1 Step' to see one gradient descent update\")\nprint(\"  2. Click 'Train 10 Steps' to speed things up\")\nprint(\"  3. Click 'Train to Convergence' to watch it finish automatically\")\nprint(\"  4. Click 'Reset Network' to try a different random starting point\")\nprint(\"\\nWatch the LEFT panel: Decision boundary evolves!\")\nprint(\"Watch the RIGHT panel: Loss decreases!\")\nprint(\"=\"*70)\n\ndisplay(status_html)\ndisplay(HBox([train_1_btn, train_10_btn, train_converge_btn, reset_btn]))\ndisplay(plot_output)\n\n# Show initial state\nupdate_display()"
    ]
})

# Save notebook
with open('lab_4_module_2_training_neural_network.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f'[OK] Added Section 3! Now {len(nb["cells"])} cells total.')
print('\nCells added:')
print('  12: [markdown] Section 3 header')
print('  13: [code]     plot_training_state() visualization function')
print('  14: [code]     Training state + button callbacks')
print('  15: [code]     Display interactive interface')
print('\n[OK] Notebook updated successfully!')
