
<div align="center">
  <p>
    <a align="center" href="https://ultralytics.com/yolov5" target="_blank">
      <img width="30%" src="assert/icon.png"></a>
  </p>
</div>



# Documentation
<details open>
<summary> Installation </summary>

Install the necessercy packages by running the below command respectively,
check if you are using [**Python>=3.8.0**](https://www.python.org/) environment, including

```sh
# Torch
curl https://example.com/script.sh | sudo bash # && chmod +x torch.sh && sudo ./torch.sh

# Tensorflow
curl https://example.com/tf.sh && chmod +x tf.sh && sudo ./tf.sh

# Aimet
curl https://example.com/aimet.sh && chmod +x aimet.sh && sudo ./aimet.sh

# pypi ackage
pip install tortoise
```
</details>


python setup.py bdist_wheel sdist
# python -m pip install .

conda create --prefix env python=3.8
conda activate ./env
sudo bash setup.sh env $(pwd) torch
