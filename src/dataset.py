from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from collections import defaultdict
import random

#1. CHARGER LE DATASET`

dataset = load_dataset("huggan/wikiart", split="train")

#2. LIMITER À X IMAGES PAR STYLE`

def limit_number_images(dataset, max_images_per_style=400):

    style_indices = defaultdict(list)
    for idx, item in enumerate(dataset):
        style_indices[item['style']].append(idx)

    selected = []
    for style, indices in style_indices.items():
        selected.extend(random.sample(indices, min(len(indices), max_images_per_style)))

    balanced_dataset = dataset.select(selected)
    return selected,balanced_dataset

selected, balanced_dataset = limit_number_images(dataset)

#3. SPLIT 70/15/15

def split(selected):
    random.shuffle(selected)
    n = len(selected)
    train_idx = selected[:int(0.7*n)]
    val_idx = selected[int(0.7*n):int(0.85*n)]
    test_idx = selected[int(0.85*n):]
    return train_idx,val_idx,test_idx

train_idx, val_idx, test_idx = split(selected)

#4. PYTORCH DATASET

class WikiArtDataset(Dataset):
    def init(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        styles = sorted(set(dataset[i]['style'] for i in indices))
        self.style_to_label = {s: i for i, s in enumerate(styles)}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        item = self.dataset[self.indices[idx]]
        image = item['image'].convert('RGB')
        image = self.transform(image)
        label = self.style_to_label[item['style']]
        return image, label

#5. DATALOADERS

def dataloader(balanced_dataset, train_idx, val_idx, test_idx, WikiArtDataset):
    transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])

    train_loader = DataLoader(WikiArtDataset(balanced_dataset, train_idx, transform), batch_size=32, shuffle=True)
    val_loader = DataLoader(WikiArtDataset(balanced_dataset, val_idx, transform), batch_size=32)
    test_loader = DataLoader(WikiArtDataset(balanced_dataset, test_idx, transform), batch_size=32)
    print(f"✅ Prêt ! {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")
    return train_loader, val_loader, test_loader

train_loader, val_loader, test_loader = dataloader(balanced_dataset, train_idx, val_idx, test_idx, WikiArtDataset)

def load_and_select_dataloaders():
    dataset = load_dataset("huggan/wikiart", split="train")
    selected, balanced_dataset = limit_number_images(dataset)
    train_idx, val_idx, test_idx = split(selected)
    return dataloader(balanced_dataset, train_idx, val_idx, test_idx, WikiArtDataset)