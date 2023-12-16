import torch
from torch.utils.data import DataLoader, Dataset

from utils import save_predictions_as_imgs, check_accuracy


class testDataset(Dataset):
    def __len__(self):
        return 4
    def __getitem__(self, index):
        x = torch.randn(3, 128, 128, 128)
        y = torch.randint(0, 1, (1, 128, 128, 128))
        return x,y

def test_save_image():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.nn.Sequential(
        torch.nn.Conv3d(3, 1, kernel_size=1, stride=1, padding=0, bias=False),
    ).to(DEVICE)
    loader = DataLoader(testDataset(), batch_size = 2)
    #check_accuracy(loader, model, (64,64,64), device=DEVICE)
    save_predictions_as_imgs(loader, model, (64,64,64), folder="saved_images/", device=DEVICE)