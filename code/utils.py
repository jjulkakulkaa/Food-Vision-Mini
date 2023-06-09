import torch
from pathlib import Path
from torchvision import transforms

# FN MADE TO MODEL SAVE

def save_model(model: torch.nn.Module,target_dir: str,model_name: str):
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                     exist_ok=True)
    
    # create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name
    
    # save model to model_save_path
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
                f=model_save_path)