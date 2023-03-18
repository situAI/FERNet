from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transform
import os
from PIL import Image



class ImageData(Dataset):
    def __init__(self, data_root, transform):
        self.data_root = data_root
        self.transform = transform
        self.img_list = list()
        for root, dirs, files in os.walk(data_root):
            for name in files:
                if name[-1] == 'g':
                    self.img_list.append(os.path.join(root, name))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        video_name = img_path.split('/')[-2]
        img_name = img_path.split('/')[-1].split('.')[0]
        name = video_name + '/' + img_name
        img = Image.open(img_path)
        data = self.transform(img)

        return data, name

def get_loader():
    root = '/data1/ABAW/ABAW5/Aff-Wild2/cropped_aligned_images/cropped_aligned/'
    tfm = transform.Compose([
        transform.ToTensor(),
        transform.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    ds = ImageData(data_root=root, transform=tfm)
    dl = DataLoader(ds, batch_size=1, shuffle=False, pin_memory=True, num_workers=16, drop_last=False)

    return dl

loader = get_loader()
for img, name in loader:
    print(name)
    break
