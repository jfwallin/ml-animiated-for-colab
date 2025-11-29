"""
Test script to validate Module 3 notebook can run successfully
Executes the key code cells from the notebook to verify functionality
"""

import sys
print("="*70)
print("TESTING MODULE 3: IRIS CLASSIFICATION")
print("="*70)

# Test 1: Package check and imports
print("\n[Test 1] Checking packages and imports...")
try:
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for testing
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense

    # Set random seeds
    np.random.seed(42)
    tf.random.set_seed(42)

    print(f"  ✓ All imports successful")
    print(f"  ✓ TensorFlow version: {tf.__version__}")
except Exception as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Load dataset
print("\n[Test 2] Loading Iris dataset...")
try:
    iris = load_iris()
    X = iris.data
    y = iris.target

    assert X.shape == (150, 4), f"Expected shape (150, 4), got {X.shape}"
    assert y.shape == (150,), f"Expected shape (150,), got {y.shape}"
    assert len(iris.target_names) == 3, "Expected 3 classes"

    print(f"  ✓ Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  ✓ Classes: {', '.join(iris.target_names)}")
except Exception as e:
    print(f"  ✗ Dataset loading failed: {e}")
    sys.exit(1)

# Test 3: Train/test split and scaling
print("\n[Test 3] Splitting and scaling data...")
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    assert X_train_scaled.shape == (120, 4), f"Expected train shape (120, 4), got {X_train_scaled.shape}"
    assert X_test_scaled.shape == (30, 4), f"Expected test shape (30, 4), got {X_test_scaled.shape}"
    assert abs(X_train_scaled[:, 0].mean()) < 0.01, "Mean should be ~0 after scaling"
    assert abs(X_train_scaled[:, 0].std() - 1.0) < 0.01, "Std should be ~1 after scaling"

    print(f"  ✓ Train set: {X_train_scaled.shape[0]} samples")
    print(f"  ✓ Test set: {X_test_scaled.shape[0]} samples")
    print(f"  ✓ Features scaled: mean={X_train_scaled[:, 0].mean():.3f}, std={X_train_scaled[:, 0].std():.3f}")
except Exception as e:
    print(f"  ✗ Split/scaling failed: {e}")
    sys.exit(1)

# Test 4: Build and train linear model
print("\n[Test 4] Building and training linear model...")
try:
    linear_model = Sequential([
        Dense(3, activation='softmax', input_dim=4, name='output')
    ], name='Linear_Model')

    linear_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Count parameters: (4 inputs × 3 outputs) + 3 biases = 15
    total_params = sum([np.prod(w.shape) for w in linear_model.trainable_weights])
    assert total_params == 15, f"Expected 15 parameters, got {total_params}"

    print(f"  ✓ Linear model built: {total_params} parameters")

    # Train
    history_linear = linear_model.fit(
        X_train_scaled, y_train,
        epochs=50,
        batch_size=16,
        validation_split=0.2,
        verbose=0
    )

    test_loss, test_accuracy = linear_model.evaluate(X_test_scaled, y_test, verbose=0)

    print(f"  ✓ Training complete")
    print(f"  ✓ Test accuracy: {test_accuracy:.1%}")

    # Sanity check: should achieve >85% on Iris
    assert test_accuracy > 0.85, f"Accuracy too low: {test_accuracy:.1%}"

except Exception as e:
    print(f"  ✗ Linear model failed: {e}")
    sys.exit(1)

# Test 5: Build and train hidden layer model
print("\n[Test 5] Building and training hidden layer model...")
try:
    hidden_units = 8

    hidden_model = Sequential([
        Dense(hidden_units, activation='relu', input_dim=4, name='hidden'),
        Dense(3, activation='softmax', name='output')
    ], name='Hidden_Layer_Model')

    hidden_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Count parameters: (4×8 + 8) + (8×3 + 3) = 32 + 24 + 11 = 67
    total_params_hidden = sum([np.prod(w.shape) for w in hidden_model.trainable_weights])
    expected_params = (4 * hidden_units + hidden_units) + (hidden_units * 3 + 3)
    assert total_params_hidden == expected_params, f"Expected {expected_params} parameters, got {total_params_hidden}"

    print(f"  ✓ Hidden layer model built: {total_params_hidden} parameters")

    # Train
    history_hidden = hidden_model.fit(
        X_train_scaled, y_train,
        epochs=50,
        batch_size=16,
        validation_split=0.2,
        verbose=0
    )

    test_loss_hidden, test_accuracy_hidden = hidden_model.evaluate(X_test_scaled, y_test, verbose=0)

    print(f"  ✓ Training complete")
    print(f"  ✓ Test accuracy: {test_accuracy_hidden:.1%}")

    # Sanity check
    assert test_accuracy_hidden > 0.85, f"Accuracy too low: {test_accuracy_hidden:.1%}"

except Exception as e:
    print(f"  ✗ Hidden layer model failed: {e}")
    sys.exit(1)

# Test 6: Comparison
print("\n[Test 6] Model comparison...")
try:
    print(f"  Linear model:      {test_accuracy:.1%}")
    print(f"  Hidden layer ({hidden_units} units): {test_accuracy_hidden:.1%}")

    improvement = test_accuracy_hidden - test_accuracy
    if improvement > 0:
        print(f"  ✓ Hidden layer improvement: +{improvement:.1%}")
    else:
        print(f"  ✓ Both models perform well (linear already sufficient)")

except Exception as e:
    print(f"  ✗ Comparison failed: {e}")
    sys.exit(1)

# Summary
print("\n" + "="*70)
print("ALL TESTS PASSED! ✅")
print("="*70)
print("\nModule 3 notebook is ready for student use!")
print("\nKey validation points:")
print("  ✓ All packages import successfully")
print("  ✓ Iris dataset loads with correct shape (150 samples, 4 features, 3 classes)")
print("  ✓ Train/test split creates 120/30 split")
print("  ✓ Feature scaling normalizes to mean=0, std=1")
print("  ✓ Linear model trains and achieves >85% accuracy")
print("  ✓ Hidden layer model trains and achieves >85% accuracy")
print("  ✓ Models are comparable on this well-separated dataset")
print("\n" + "="*70)
