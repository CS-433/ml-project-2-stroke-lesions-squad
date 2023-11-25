import torch
import torchvision
from dataset import MRIImages
from torch.utils.data import DataLoader


def get_loaders(
        train_img_dir,
        train_mask_dir,
        val_img_dir,
        val_mask_dir,
        batch_size,
        num_workers,
        pin_memory,
        train_transform):

    train_ds = MRIImages(train_img_dir, train_mask_dir, transform=train_transform)
    train_loader = DataLoader(train_ds,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                pin_memory=pin_memory,
                                shuffle=True,
                              )
    val_ds = MRIImages(val_img_dir, val_mask_dir, transform=train_transform)
    val_loader = DataLoader(val_ds,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                pin_memory=pin_memory,
                                shuffle=False,
                                )

    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixel = 0

    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.float().unsqueeze(1).to(device=device)

            scores = model(x)
            predictions = torch.round(torch.sigmoid(scores))
            num_correct += (predictions == y).sum()
            num_pixel += torch.numel(predictions)

    print(f"Got {num_correct}/{num_pixel} with accuracy {float(num_correct)/float(num_pixel)*100:.2f}")