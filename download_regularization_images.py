import requests
from io import BytesIO
import torch
from datasets import load_dataset
import json
from PIL import Image
from tqdm import tqdm
import os

print("Loading LAION aesthetics 6.5+...")
dataset = load_dataset(
    'ChristophSchuhmann/improved_aesthetics_6.5plus',
    streaming=True,
)['train']

datalist = []
for i, data in tqdm(enumerate(dataset)):
    datalist.append(data)

class dataset_wrapper(torch.utils.data.IterableDataset):
    def __init__(self, dataset):
        super(dataset_wrapper).__init__()
        self.dataset = dataset

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            num_workers = 1
            worker_id = 0
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
        it = iter(self.dataset)
        counter = 0
        while True:
            if counter % num_workers == worker_id:
                counter += 1
                data = next(it)
                try:
                    response = requests.get(data['URL'], timeout=2)
                    img = Image.open(BytesIO(response.content)).convert("RGB")
                except:
                    img = None
                yield data, img
            else:
                counter += 1
                next(it)

dataset = dataset_wrapper(datalist)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=16, collate_fn=lambda x: x[0])

counter = 0
if not os.path.exists("regularization_images"):
    os.mkdir("regularization_images")
with open("regularization_images.jsonl", "a") as f:
    for i, (data, img) in tqdm(enumerate(dataloader)):
        if img is None:
            continue
        else:
            file_name = f"regularization_images/{counter}.jpg"
            img.save(file_name, quality=95)
            f.write(json.dumps(dict(
                file_name=file_name,
                caption=data['TEXT'],
            )) + "\n")
            counter += 1