from pathlib import Path
import os
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

old_path = Path('../../imagenet/ILSVRC/Data/CLS-LOC/train')
new_path = Path('../../imagenet resized/ILSVRC/Data/CLS-LOC/train')
new_path.mkdir(parents=True, exist_ok=True)

for category in tqdm(os.listdir(old_path)):
    (new_path/category).mkdir(exist_ok=True)
    for file in os.listdir(old_path/category):
        im = Image.open(old_path/category/file).resize((256,256)).convert('RGB').save(new_path/category/file)
    

