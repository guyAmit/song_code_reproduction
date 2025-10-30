# train_with_cve.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from cve_loss.py import CorrelatedValueEncodingLoss, reconstruct_numeric_from_params

# Example dataset (CIFAR-style images turned to grayscale float for the secret)
tx = transforms.Compose([transforms.ToTensor()])  # keeps [0,1] floats
train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=tx)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(num_classes=10).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# Build the CVE loss wrapper; by default it flattens dataset items numerically
cve = CorrelatedValueEncodingLoss(
    model=model,
    dataset=train_ds,      # <-- the secret is derived from *this* dataset
    lambda_c=1.0,          # tune; larger pushes stronger correlation (can hurt accuracy)
    device=device,
)

for epoch in range(3):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        task_loss = criterion(logits, y)
        lambd_loss, c_term = cve()
        total_loss = task_loss + lambd_loss

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

    print(f"epoch {epoch} done")

# --- Reconstruct numeric data correlated into params (rough demo) ---
# Suppose we want to reconstruct back to (C,H,W)=(1,32,32) grayscale-ish items
# and we assume we encoded up to K items by simple concatenation.
K = 200  # number of images you expect to approximately recover
item_shape = (1, 32, 32)
total_values = K * (1 * 32 * 32)

recon, raw_slice = reconstruct_numeric_from_params(
    model,
    total_values=total_values,
    item_shape=item_shape,
    value_range=(0.0, 255.0),
    try_invert=True,
    device=device,
)

print("Recovered tensor shape:", recon.shape)  # (K_recovered, 1, 32, 32)
# You can now save a grid of 'recon' images for inspection.
