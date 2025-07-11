import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# === Checkpoint Save & Load ===
def save_checkpoint(model, optimizer, epoch, filename="parameters.pt"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, filename)

def load_checkpoint(model, optimizer, filename="parameters.pt"):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']


# === Rate Encoder ===
def rate_encode_batch(X_batch, timesteps=100):
    B, N = X_batch.shape
    encoded = np.random.rand(timesteps, B, N) < X_batch[None, :, :]
    return torch.tensor(encoded, dtype=torch.float32)

# === Trainable LIF Layer ===
class TrainableLIFLayer(nn.Module):
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

# === SNN Model ===
class TrainableSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = TrainableLIFLayer(784, 32)
        self.l2 = TrainableLIFLayer(32, 32)
        self.l3 = TrainableLIFLayer(32, 32)  # 🆕 New hidden layer
        self.l4 = TrainableLIFLayer(32, 10)  # Output layer

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

# === Training Loop ===
def train_snn(model, X, y, epochs=10, batch_size=64, timesteps=100, resume=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    start_epoch = 0

    if resume:
        try:
            start_epoch = load_checkpoint(model, optimizer)
            print(f"✅ Resuming from epoch {start_epoch}")
        except FileNotFoundError:
            print("🚀 Starting training from scratch...")

    for epoch in range(0, epochs):
        indices = np.random.permutation(len(X))
        X, y = X[indices], y[indices]
        total_loss, correct = 0, 0
        model.train()

        for i in range(0, len(X), batch_size):
            xb = X[i:i+batch_size]
            yb = y[i:i+batch_size]
            xb_encoded = rate_encode_batch(xb, timesteps)
            yb_tensor = torch.tensor(yb, dtype=torch.long)

            model.reset_state(len(xb))
            out = model(xb_encoded)
            loss = loss_fn(out, yb_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == yb_tensor).sum().item()
            cumulative_acc = correct / (i + len(xb))
            print(f"correct: {correct}, total: {i + len(xb)}, acc: {cumulative_acc*100}")
            # print(f"pred: {pred}, true: {yb_tensor}")
        acc = correct / len(X)
        print(f"📘 Epoch {epoch+1}: Loss={total_loss:.4f}, Accuracy={acc:.4f}")

        # Save checkpoint
        save_checkpoint(model, optimizer, epoch + 1)

# === Main ===
if __name__ == "__main__":
    df = pd.read_csv("DATASET/mnist_train.csv")
    data = df.values
    X = data[:, 1:] / 255.0
    y = data[:, 0].astype(int)

    model = TrainableSNN()
    train_snn(model, X, y, epochs=2, batch_size=64, resume=True)
