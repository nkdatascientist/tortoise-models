# Tortoise 🚀 by Nishanth M, Apache 2.0 License                       
"""
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣴⣿⣿⣷⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣿⣿⣷⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣿⣿⡟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣻⣿⣿⣿⣁⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⣀⣤⣤⣤⣤⣠⡶⣿⡏⠉⠉⠉⠉⢙⣿⢷⣦⣤⣤⣤⣤⣀⡀⠀⠀⠀
⠀⢀⣴⣿⣿⣿⣿⣿⣿⡟⠀⠈⣻⣶⣶⣶⣶⣿⠃⠀⠙⣿⣿⣿⣿⣿⣿⣷⣄⠀
⠀⠈⠛⠛⠛⠛⠛⠋⣿⠀⠀⣼⠏⠀⠀⠀⠀⠘⢧⡀⠀⢻⡏⠛⠛⠛⠛⠛⠉⠀
⠀⠀⠀⠀⠀⠀⠀⠸⣿⠀⠈⠻⣦⡀⠀⠀⢀⣴⠟⠁⠀⢸⡇⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⣿⡀⠀⠀⣸⡿⠒⠒⢿⣏⠀⠀⠀⣾⠃⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠘⣷⡀⠰⣿⡀⠀⠀⢀⣽⠗⠀⣼⠏⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⢀⣤⣾⣿⣄⠈⠁⠀⠀⠈⠁⣠⣾⣿⣤⡀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⢠⣿⣿⣿⣿⡿⠿⢶⣤⣤⡶⠾⠿⣿⣿⣿⣿⣆⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⣿⣿⡿⠟⠉⠀⠀⠀⠻⡿⠁⠀⠀⠈⠛⠿⣿⣿⡆⠀⠀⠀⠀

Usage - Command
    $ python main.py val        # Validation 
                     train      # Model Finetuning 
                     quantize   # Quantization
                     export     # onnx export
"""

from tortoise.models import RetinanetModel
import torch, sys 

device = "cuda" if torch.cuda.is_available() else "cpu"
def main(args):
    img = torch.randn(1, 3, 640, 640).to(device)
    task = RetinanetModel.from_pretrained(backbone="resnet18", exp_name="experiment_7")
    if "val" in args: task.validation()
    if "train" in args: task.train()
    if "quant" in args: task.quantize()
    if "export" in args: task.export(img)

if __name__ == '__main__':
    main(sys.argv[1:])
