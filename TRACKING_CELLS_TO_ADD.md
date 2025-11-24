# Additional Cells for Attempt Tracking

These cells need to be integrated into the interactive sections to track student attempts.

---

## Section 3: Line Fitting (Interactive Slider)

**MODIFY the existing `plot_guess` function** to include attempt tracking:

```python
import pandas as pd
from ipywidgets import interact, FloatSlider, Checkbox

# Reset attempt history each time this cell is run
attempt_history = []

def plot_guess(m, b, show_residuals=True):
    """Plot the user's guessed line, compute SSE, and record history."""
    global attempt_history

    # === ADD THIS: Track attempts ===
    section_attempts["section_3_line_fitting"] += 1
    if section_timestamps["section_3_line_fitting"]["started"] is None:
        section_timestamps["section_3_line_fitting"]["started"] = datetime.now().isoformat()
    # === END ADD ===

    y_pred = m * x + b
    loss = sse(y, y_pred)

    prev_loss = attempt_history[-1]["loss"] if attempt_history else None
    attempt_history.append({"m": m, "b": b, "loss": loss})

    # Plot data and guessed line
    plt.figure(figsize=(7, 4))
    plt.scatter(x, y, label="Data")
    plt.plot(x, y_pred, label=f"Your line: y = {m:.2f}x + {b:.2f}", color='green')

    # Plot residuals if requested
    if show_residuals:
        for xi, yi, ypi in zip(x, y, y_pred):
            plt.plot([xi, xi], [yi, ypi], linestyle='--', alpha=0.6)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Your Global Error (SSE) = {loss:.2f}")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Warm/colder feedback
    if prev_loss is not None:
        if loss < prev_loss:
            print("Feedback: 🔥 Warmer (your global error decreased).")
        elif loss > prev_loss:
            print("Feedback: 🧊 Colder (your global error increased).")
        else:
            print("Feedback: No change in global error.")

    # Show last few attempts
    df = pd.DataFrame(attempt_history)
    print("Recent attempts:")
    display(df.tail(5))

interact(
    plot_guess,
    m = FloatSlider(value=0.0, min=-5, max=5, step=0.1, description="Slope m"),
    b = FloatSlider(value=0.0, min=-5, max=5, step=0.1, description="Intercept b"),
    show_residuals = Checkbox(value=True, description="Show residuals"),
);
```

---

## Section 3.2: Parameter Space Game

**MODIFY the `on_submit_clicked` function:**

```python
def on_submit_clicked(b_widget):
    """Record a guess and update the table + (m,b) plot with colors = MSE."""
    m_guess = m_slider.value
    b_guess = b_slider.value
    mse_val = compute_mse(m_guess, b_guess)

    # === ADD THIS: Track attempts ===
    section_attempts["section_3_2_parameter_space"] += 1
    if section_timestamps["section_3_2_parameter_space"]["started"] is None:
        section_timestamps["section_3_2_parameter_space"]["started"] = datetime.now().isoformat()
    # === END ADD ===

    mse_history.append({
        "attempt": len(mse_history) + 1,
        "m": m_guess,
        "b": b_guess,
        "MSE": mse_val
    })

    # ... rest of the function ...
```

---

## Section 4: Hidden Function Optimization

**MODIFY the `record_and_update_display` function:**

```python
def record_and_update_display(x_guess):
    """Evaluate hidden_func at x_guess, record it, and update plot + table."""
    global opt_history

    # === ADD THIS: Track attempts ===
    section_attempts["section_4_hidden_function"] += 1
    if section_timestamps["section_4_hidden_function"]["started"] is None:
        section_timestamps["section_4_hidden_function"]["started"] = datetime.now().isoformat()
    # === END ADD ===

    y_guess = hidden_func(x_guess)
    prev = opt_history[-1] if opt_history else None

    opt_history.append({
        "attempt": len(opt_history) + 1,
        "x": x_guess,
        "f(x)": y_guess
    })

    # ... rest of the function ...
```

---

## Section 5: Mountain Landscape

**MODIFY the `on_sample_clicked` function:**

```python
def on_sample_clicked(b_widget):
    """Record a sample (x, y, height) and update the plot + table."""
    x_guess = x_slider_2d.value
    y_guess = y_slider_2d.value
    h_val = float(mountain_height(x_guess, y_guess))

    # === ADD THIS: Track attempts ===
    section_attempts["section_5_mountain_landscape"] += 1
    if section_timestamps["section_5_mountain_landscape"]["started"] is None:
        section_timestamps["section_5_mountain_landscape"]["started"] = datetime.now().isoformat()
    # === END ADD ===

    samples_2d.append({
        "attempt": len(samples_2d) + 1,
        "x": x_guess,
        "y": y_guess,
        "height": h_val
    })

    # ... rest of the function ...
```

---

## Alternative: Simpler Auto-Tracking Approach

Instead of modifying existing cells, we can use the existing `attempt_history`, `mse_history`, `opt_history`, and `samples_2d` lists that are already in the notebook!

**ADD this cell just before the export section:**

```python
# ============================================================================
# AUTO-CALCULATE ATTEMPT COUNTS FROM EXISTING DATA
# ============================================================================

# The notebook already tracks attempts in various lists
# Let's extract the counts from those

print("📊 Calculating engagement metrics from your interactions...")
print()

# Section 3: Line fitting (uses attempt_history)
if 'attempt_history' in globals() and attempt_history:
    section_attempts["section_3_line_fitting"] = len(attempt_history)
    print(f"Section 3 (Line Fitting): {len(attempt_history)} attempts")

# Section 3.2: Parameter space (uses mse_history)
if 'mse_history' in globals() and mse_history:
    section_attempts["section_3_2_parameter_space"] = len(mse_history)
    print(f"Section 3.2 (Parameter Space): {len(mse_history)} guesses")

# Section 4: Hidden function (uses opt_history)
if 'opt_history' in globals() and opt_history:
    section_attempts["section_4_hidden_function"] = len(opt_history)
    print(f"Section 4 (Hidden Function): {len(opt_history)} attempts")

# Section 5: Mountain landscape (uses samples_2d)
if 'samples_2d' in globals() and samples_2d:
    section_attempts["section_5_mountain_landscape"] = len(samples_2d)
    print(f"Section 5 (Mountain Landscape): {len(samples_2d)} samples")

print()
print(f"Total interactions: {sum(section_attempts.values())}")
print("✓ Engagement metrics ready for export")
```

This is much simpler because it uses the data structures already in the notebook!

---

## Recommendation

**Use the simpler auto-tracking approach** because:

1. **No modification of existing interactive cells needed**
2. **Uses data already being collected**
3. **Students don't need to do anything different**
4. **Less code to maintain**
5. **Works retroactively** - counts all interactions automatically

Just add the auto-calculation cell before the export, and the export will include:
- Number of attempts per section
- Total engagement across all sections
- Individual answer timestamps

This gives you rich data about student engagement without any additional complexity!
