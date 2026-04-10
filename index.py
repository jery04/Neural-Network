import numpy as np  # Core library for array operations and numerical computations

# K-Fold cross-validation for model evaluation
from sklearn.model_selection import KFold

# Performance metrics for multi-label/binary classification
from sklearn.metrics import (
    accuracy_score,        # Overall accuracy
    precision_score,       # Precision score (macro average)
    recall_score,          # Recall score (macro average)
    f1_score,              # F1-Score (macro average)
    confusion_matrix,      # Confusion matrix (used when n_output == 1)
    classification_report  # Detailed report with precision, recall, f1 per class
)
class CrossValidator:
    """Wrapper around a NeuralNetwork that performs K-Fold cross-validation and trains a final model on all data."""
    
    def __init__(self, 
                 n_features: int,
                 hidden_sizes: tuple = (64, 32),
                 n_output: int = 3,
                 learning_rate: float = 0.05,
                 k_folds: int = 5,
                 epochs: int = 8000,
                 random_state: int = 42):
        """Initialize CrossValidator with network and training hyperparameters."""
        self.n_features = n_features
        self.hidden_sizes = hidden_sizes
        self.n_output = n_output
        self.learning_rate = learning_rate
        self.k_folds = k_folds
        self.epochs = epochs
        self.random_state = random_state

    def _create_model(self):
        """Create a new NeuralNetwork instance for the current configuration."""
        return NeuralNetwork(
            n_features=self.n_features,
            hidden_sizes=self.hidden_sizes,
            n_output=self.n_output,
            learning_rate=self.learning_rate,
            seed=None  # We don't set a seed here so each fold is different
        )

    def cross_validate(self, X: np.ndarray, y: np.ndarray, verbose: bool = True):
        """Perform K-Fold cross validation and print detailed metrics. Returns mean F1-score."""
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_samples = X.shape[0]
        if n_samples < 2:
            raise ValueError("Se requieren al menos 2 muestras para cross validation.")

        n_splits = self.k_folds
        if n_splits > n_samples:
            if verbose:
                print(
                    f"Aviso: k_folds={self.k_folds} es mayor que n_samples={n_samples}. "
                    f"Se ajusta automaticamente a {n_samples}."
                )
            n_splits = n_samples
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        
        fold_accuracies = []
        fold_precisions = []
        fold_recalls = []
        fold_f1s = []
        
        print(f"=== K-Fold Cross Validation (K={n_splits}) ===\n")
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            print(f"Fold {fold}/{n_splits} ".ljust(50, "-"))
            
            # Create and train temporary model
            model = self._create_model()
            model.fit(X_train, y_train, epochs=self.epochs, verbose=False)
            
            # Predictions on validation set
            y_pred_prob = model.predict(X_val)
            y_pred_class = model.predict_class(X_val, threshold=0.5)
            
            # Metrics (multi-label)
            acc = accuracy_score(y_val.ravel() if y_val.shape[1]==1 else y_val, 
                                y_pred_class.ravel() if y_pred_class.shape[1]==1 else y_pred_class)
            
            # For multi-output we use average='macro' or 'samples'
            prec = precision_score(y_val, y_pred_class, average='macro', zero_division=0)
            rec  = recall_score(y_val, y_pred_class, average='macro', zero_division=0)
            f1   = f1_score(y_val, y_pred_class, average='macro', zero_division=0)
            
            fold_accuracies.append(acc)
            fold_precisions.append(prec)
            fold_recalls.append(rec)
            fold_f1s.append(f1)
            
            # Confusion matrix (only if single output)
            if self.n_output == 1:
                cm = confusion_matrix(y_val.ravel(), y_pred_class.ravel())
                print("Matriz de Confusión:")
                print(cm)
                print(f"TP: {cm[1,1] if cm.shape[0]>1 else 0} | FP: {cm[0,1] if cm.shape[1]>1 else 0}")
                print(f"FN: {cm[1,0] if cm.shape[0]>1 else 0} | TN: {cm[0,0] if cm.shape[0]>1 else 0}")
            
            print(f"Accuracy : {acc:.4f}")
            print(f"Precision: {prec:.4f}")
            print(f"Recall   : {rec:.4f}")
            print(f"F1-Score : {f1:.4f}\n")
        
        # Resultados promedio
        print("="*60)
        print("RESULTADOS PROMEDIO DEL CROSS VALIDATION")
        print("="*60)
        print(f"Accuracy promedio : {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
        print(f"Precision promedio: {np.mean(fold_precisions):.4f} ± {np.std(fold_precisions):.4f}")
        print(f"Recall promedio   : {np.mean(fold_recalls):.4f} ± {np.std(fold_recalls):.4f}")
        print(f"F1-Score promedio : {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")
        
        return np.mean(fold_f1s)  # devolvemos F1 como referencia

    def fit_final_model(self, X: np.ndarray, y: np.ndarray, verbose: bool = True):
        """Train the final model on ALL the data and return the trained model."""
        print("\n" + "="*70)
        print("ENTRENANDO MODELO FINAL CON TODOS LOS DATOS")
        print("="*70)
        
        self.final_model = self._create_model()
        self.final_model.fit(X, y, epochs=self.epochs, verbose=verbose)
        
        print("\nModelo final entrenado correctamente.")
        return self.final_model

class NeuralNetwork:
    """Simple feedforward neural network with tanh hidden activations and sigmoid output trained with binary cross-entropy."""
    def __init__(
        self,
        n_features: int,
        hidden_sizes: tuple[int, ...] = (8, 8),
        n_output: int = 1,
        learning_rate: float = 0.1,
        seed: int | None = None,
    ):
        """Initialize network parameters and weights using Xavier/Glorot initialization."""
        if seed is not None:
            np.random.seed(seed)

        self.learning_rate = learning_rate
        self.hidden_sizes = list(hidden_sizes)

        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []

        layer_sizes = [n_features, *self.hidden_sizes, n_output]

        # Xavier/Glorot to avoid initial saturation.
        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            w = np.random.uniform(-limit, limit, size=(fan_in, fan_out))
            b = np.zeros((1, fan_out))
            self.weights.append(w)
            self.biases.append(b)

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def _tanh_derivative(a: np.ndarray) -> np.ndarray:
        """Derivative of tanh activation given activation 'a'."""
        return 1.0 - (a ** 2)

    @staticmethod
    def _binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Compute binary cross-entropy loss between true labels and predicted probabilities."""
        y_prob = np.clip(y_prob, 1e-7, 1.0 - 1e-7)
        return float(-np.mean(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob)))

    def forward(self, x: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Perform a forward pass and return (activations, z_values)."""
        activations = [x]
        z_values = []

        current = x
        for layer_idx, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = current @ w + b
            z_values.append(z)

            if layer_idx == len(self.weights) - 1:
                current = self._sigmoid(z)
            else:
                current = np.tanh(z)

            activations.append(current)

        return activations, z_values

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return the network's output probabilities for input `x`."""
        activations, _ = self.forward(x)
        return activations[-1]

    def predict_class(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary class predictions using the given threshold."""
        return (self.predict(x) >= threshold).astype(int)

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 10000, verbose: bool = True) -> None:
        """Train the network using gradient descent with binary cross-entropy loss."""
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_samples = x.shape[0]

        for epoch in range(1, epochs + 1):
            activations, _ = self.forward(x)
            y_pred = activations[-1]

            deltas: list[np.ndarray | None] = [None] * len(self.weights)

            # For BCE + sigmoid at output: dL/dz = y_pred - y
            deltas[-1] = y_pred - y
            # Backprop on hidden layers
            for layer_idx in range(len(self.weights) - 2, -1, -1):
                back_signal = deltas[layer_idx + 1] @ self.weights[layer_idx + 1].T
                deltas[layer_idx] = back_signal * self._tanh_derivative(activations[layer_idx + 1])

            # Gradient descent
            for layer_idx in range(len(self.weights)):
                grad_w = (activations[layer_idx].T @ deltas[layer_idx]) / n_samples
                grad_b = np.sum(deltas[layer_idx], axis=0, keepdims=True) / n_samples

                self.weights[layer_idx] -= self.learning_rate * grad_w
                self.biases[layer_idx] -= self.learning_rate * grad_b

            if verbose and (epoch == 1 or epoch % 500 == 0 or epoch == epochs):
                loss = self._binary_cross_entropy(y, y_pred)
                print(f"Epoca {epoch:5d}/{epochs} - Loss: {loss:.6f}")

