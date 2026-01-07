import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class BlinkSequenceDataset(Dataset):
	"""Simple dataset for blink time-series.

	Each sample:
	  - x: tensor of shape (T, F) where F is number of features
		(e.g., [open_closed_binary, blink_duration_ms, gap_ms, ...])
	  - y: target sequence index (e.g., Morse symbol class or character index)
	This is a placeholder; you should implement __init__ to load your
	precomputed sequences and labels.
	"""

	def __init__(self, sequences, labels):
		self.sequences = sequences
		self.labels = labels

	def __len__(self):
		return len(self.sequences)

	def __getitem__(self, idx):
		return self.sequences[idx], self.labels[idx]


class BlinkLSTM(nn.Module):
	def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
		super().__init__()
		self.lstm = nn.LSTM(
			input_dim,
			hidden_dim,
			num_layers=num_layers,
			batch_first=True,
			bidirectional=True,
		)
		self.fc = nn.Linear(hidden_dim * 2, num_classes)

	def forward(self, x, lengths):
		# x: (B, T, F), lengths: (B,)
		packed = nn.utils.rnn.pack_padded_sequence(
			x, lengths.cpu(), batch_first=True, enforce_sorted=False
		)
		packed_out, (h_n, _) = self.lstm(packed)
		# Use final hidden state (both directions)
		h_final = torch.cat((h_n[-2], h_n[-1]), dim=1)
		logits = self.fc(h_final)
		return logits


def collate_pad(batch):
	sequences, labels = zip(*batch)
	lengths = torch.tensor([seq.size(0) for seq in sequences], dtype=torch.long)
	padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
	labels = torch.tensor(labels, dtype=torch.long)
	return padded, lengths, labels


def train_lstm_model(train_dataset, val_dataset, input_dim, num_classes, epochs=30, lr=1e-3):
	train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_pad)
	val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_pad)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = BlinkLSTM(input_dim, hidden_dim=64, num_layers=2, num_classes=num_classes).to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=lr)

	best_val_acc = 0.0

	for epoch in range(epochs):
		model.train()
		for x, lengths, y in train_loader:
			x, lengths, y = x.to(device), lengths.to(device), y.to(device)
			optimizer.zero_grad()
			logits = model(x, lengths)
			loss = criterion(logits, y)
			loss.backward()
			optimizer.step()

		# Validation
		model.eval()
		correct = 0
		total = 0
		with torch.no_grad():
			for x, lengths, y in val_loader:
				x, lengths, y = x.to(device), lengths.to(device), y.to(device)
				logits = model(x, lengths)
				preds = torch.argmax(logits, dim=1)
				total += y.size(0)
				correct += (preds == y).sum().item()

		acc = 100.0 * correct / max(1, total)
		print(f"Epoch {epoch+1}/{epochs} - LSTM Val Acc: {acc:.2f}%")
		if acc > best_val_acc:
			best_val_acc = acc
			torch.save(model.state_dict(), "blink_lstm.pth")
			print("Saved best LSTM model.")

	return model


if __name__ == "__main__":
	print("This file defines the BlinkLSTM and training utilities.\n"
		  "Prepare blink feature sequences (e.g., from mc_decoder.py) and "
		  "construct BlinkSequenceDataset instances to train the LSTM.")

