
<div align="center">
  <p>
    <a align="center" href="https://ultralytics.com/yolov5" target="_blank">
      <img width="20%" src="assert/icon.png"></a>
  </p>
</div>


## Documentation

<details open>
<summary> Installation </summary>

Install the necessercy packages by running the below command respectively and basic requirements, [**Python>=3.8.0**](https://www.python.org/) environment.

```sh
conda create --prefix env python=3.8

# Installing Requirements
curl -o tortoise_requirements.sh https://raw.githubusercontent.com/nkdatascientist/tortoise-models/main/data/scripts/aimet.sh && chmod +x tortoise_requirements.sh && sudo ./tortoise_requirements.sh env $(pwd)/ torch
```
</details>

<!-- ## Models
  - [Classification](https://github.com/nkdatascientist/tortoise-models/tree/main/docs/models/classification)
  - [Detection](https://github.com/nkdatascientist/tortoise-models/tree/main/docs/models/detection) -->



### Pretrained Checkpoints

| Model                                                                                           | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | mAP<sup>val<br>50 |  params<br><sup>(M) | GMac<br><sup>@640 (B) |
| ----------------------------------------------------------------------------------------------- | --------------------- | -------------------- | ----------------- | ------------------- | ---------------------- |
| [Retinanet18]()          | 640                   | 21.6                 | 37.0              | **21.4**             |          **75.45 GMac**       |
<!-- | [Retinanet34](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt)          | dynamic               | 23.7               | 45.7              | **1.9**             |          **4.5**       | -->
<!-- | [Retinanet50](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt)          | 640                   | 28.0                 | 45.7              | **1.9**             |          **4.5**       | -->
<!-- | [Retinanet101](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt)         | 640                   | 28.0                 | 45.7              | **1.9**             |          **4.5**       | -->


## Retinanet
The model is trained on dynamic input resolutions and we achieve 23.7 baseline accuracy

```py
from tortoise.models import RetinanetModel
from PIL import Image
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

img = Image.open("./data/imgs/1.jpg").convert("RGB")
task = RetinanetModel.from_pretrained(backbone="resnet18")
out = task.predict(img)
```

### Paper
RetinaNet is a popular object detection architecture that was introduced in the paper titled ["Focal Loss for Dense Object Detection" by Tsung-Yi Lin et al. in 2017](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf)

## License
  Apache 2.0