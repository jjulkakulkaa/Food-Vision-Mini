import os
import torch
import requests
import zipfile
from pathlib import Path

# DOWNLOADING DATA FROM MDRBROUKE PYTORCH DEEP LEARNING COURSE

device = "cuda" if torch.cuda.is_available() else "cpu"

data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

if image_path.is_dir():
    print("it exist")
else:
    image_path.mkdir(parents=True, exist_ok=True)
    print("making dir")
    with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
        request=requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
        print("downloading")
        f.write(request.content)

# uznip
with zipfile.ZipFile(data_path /"pizza_steak_sushi.zip", "r") as zip_ref:
    zip_ref.extractall(image_path)

os.remove(data_path / "pizza_steak_sushi.zip")



