import numpy as np
import torch
import torch.nn as nn
import pandas as pd

# === Rate Encode ===
def rate_encode_batch(X_batch, timesteps=100):
    B, N = X_batch.shape
    encoded = np.random.rand(timesteps, B, N) < X_batch[None, :, :]
    return torch.tensor(encoded, dtype=torch.float32)

# === LIF Inference Layer ===
class LIFNeuronLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.threshold = nn.Parameter(torch.rand(output_size) * 1.0)
        self.decay = nn.Parameter(torch.rand(output_size) * 0.5 + 0.4)
        self.v = None

    def reset_state(self, batch_size):
        self.v = torch.zeros(batch_size, self.fc.out_features)

    def forward(self, x):
        if self.v is None or self.v.shape[0] != x.shape[0]:
            self.reset_state(x.shape[0])
        self.v = self.v * self.decay + self.fc(x)
        spikes = torch.sigmoid(5 * (self.v - self.threshold))  # surrogate gradient
        self.v = self.v * (1.0 - spikes)
        return spikes

# === Full Inference Model ===
class SimpleSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = LIFNeuronLayer(784, 32)
        self.l2 = LIFNeuronLayer(32, 32)
        self.l3 = LIFNeuronLayer(32, 32)  # 🆕 New hidden layer
        self.l4 = LIFNeuronLayer(32, 10)  # Output layer

    def reset_state(self, batch_size):
        self.l1.reset_state(batch_size)
        self.l2.reset_state(batch_size)
        self.l3.reset_state(batch_size)
        self.l4.reset_state(batch_size)

    def forward(self, x_seq):
        spike_counts = torch.zeros(x_seq.shape[1], 10)
        for t in range(x_seq.shape[0]):
            h1 = self.l1(x_seq[t])
            h2 = self.l2(h1)
            h3 = self.l3(h2)
            o = self.l4(h3)
            spike_counts += o
        return spike_counts


# === Load Checkpoint ===
def load_model_from_checkpoint(model, checkpoint_path="parameters.pt"):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Model loaded from {checkpoint_path}")

# === MAIN TEST ===
if __name__ == "__main__":
    df = pd.read_csv("DATASET/mnist_test.csv")
    data = df.values
    X = data[:, 1:] / 255.0
    y = data[:, 0].astype(int)

    # Load a sample of data
    indices = np.random.choice(len(X), 2, replace=False)
    X_batch = X[indices]
    y_batch = y[indices]

    # Prepare encoded input
    encoded = rate_encode_batch(X_batch, timesteps=100)

    # Load and run model
    model = SimpleSNN()
    load_model_from_checkpoint(model, "parameters.pt")
    model.eval()
    model.reset_state(batch_size=len(X_batch))

    with torch.no_grad():
        output = model(encoded)
        predicted = torch.argmax(output, dim=1)
        correct = (predicted.numpy() == y_batch).sum().item()
        print("True labels     :", y_batch)
        print("Predicted labels:", predicted.numpy())
        print(f"🎯 Accuracy: {correct / len(y_batch):.4f}")
        print("Spike counts    :\n", output)
