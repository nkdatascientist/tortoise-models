
import torch.nn as nn

import math
import torch.utils.model_zoo as model_zoo
from torchvision.ops import nms
from tortoise.models.retinanet.utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
from tortoise.models.retinanet.anchors import Anchors
from tortoise.models.retinanet import losses
from tortoise.models.retinanet.dataloader import CocoDataset, Resizer, AspectRatioBasedSampler, Normalizer, Augmenter, collater
from tortoise.utils import timetaken, download_file

from torch.utils.data import DataLoader
from torchvision import transforms
import torch

from pycocotools.cocoeval import COCOeval
import os, json, time, copy, yaml
from tqdm import tqdm

# https://github.com/Delgan/loguru
from loguru import logger
# logger.info("If you're using Python {}, prefer {feature} of course!", 3.6, feature="f-strings")

class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]
class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)
class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)
class ResNet(nn.Module):

    def __init__(self, num_classes, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if block == BasicBlock:
            fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels,
                         self.layer4[layers[3] - 1].conv3.out_channels]
        else:
            raise ValueError(f"Block type {block} not understood")

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])
        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01
        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))
        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)
        self.freeze_bn()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs):
        img_batch = inputs
        x = self.conv1(img_batch)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        features = self.fpn([x2, x3, x4])
        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)
        classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)
        return regression, classification

class RetinanetModel:
    def __init__(self):
        self.model_dir = "weights"
        self.exp_name = "experiment"
        self.model_config = {
            "resnet18": {
                "block": BasicBlock,
                "layer": [2, 2, 2, 2],
                "resnet_url": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
                "retinanet_url": "/media/nk/PortableSSD1/RnD/pytorch-multimodels/weights/resnet18/experiment_7/best.pth",
                "config": "/media/nk/PortableSSD1/RnD/pytorch-multimodels/data/hyps/retinanet/hyp.scratch-low.yaml",
                "sha256": ""
                },
            "resnet34": {
                    "block": BasicBlock,
                    "layer": [3, 4, 6, 3],
                    "resnet_url": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
                    "retinanet_url": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
                    "sha256": ""
                },
            "resnet50": {
                    "block": Bottleneck,
                    "layer": [3, 4, 6, 3],
                    "resnet_url": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
                    "retinanet_url": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
                    "sha256": ""
                },
            "resnet101": {
                    "block": Bottleneck,
                    "layer": [3, 4, 23, 3],
                    "resnet_url": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
                    "retinanet_url": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
                    "sha256": ""
                },
            "resnet152": {
                    "block": Bottleneck,
                    "layer": [3, 8, 36, 3],
                    "resnet_url": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
                    "retinanet_url": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
                    "sha256": ""
                }
            }
    
    @classmethod
    def load_config(self, backbone, filename, model_dir):
        from munch import DefaultMunch
        if not os.path.exists(f"{model_dir}/{filename}"):
            download_file(self.model_config[backbone]["config"], filename, model_dir)
        with open(f"{model_dir}/{filename}", "r") as stream:
            try: config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc); exit()
        # https://stackoverflow.com/a/24852544
        return DefaultMunch.fromDict(config) 
    
    def get_info(self, backbone): return self.model_config[backbone]["retinanet_url"]
    @classmethod
    def from_scratch(self, backbone, model_dir=None, device="cuda"):
        self.__init__(self)
        if model_dir: self.model_dir = model_dir   
        if backbone not in self.model_config.keys(): 
            logger.debug(f"Backbone not Found: {backbone}")
            assert False, "backbone Not found" 
        
        self.args = self.load_config(backbone, "config.yaml", os.path.join(self.model_dir, backbone))
        self.args.device = device
        os.makedirs(os.path.join(self.model_dir, backbone), exist_ok=True)
        self.args.backbone = backbone
        self.model = ResNet(self.args.num_classes, self.model_config[backbone]["block"], self.model_config[backbone]["layer"])
        self.model.load_state_dict(model_zoo.load_url(self.model_config[backbone]['resnet_url'], model_dir=os.path.join(self.model_dir, backbone)), strict=False)
        self.args.exp_name = f"{os.path.join(self.model_dir, backbone)}/{self.exp_name}_{len(os.listdir(os.path.join(self.model_dir, backbone)))}"
        os.makedirs(f"{self.args.exp_name }/", exist_ok=True)
        logger.info(f"Export directory: {self.args.exp_name}")
        return self()
    
    @classmethod
    def from_pretrained(self, backbone, model_dir=None, exp_name=None, filename="best.pth",  device="cuda"):
        self.__init__(self)
        if model_dir: self.model_dir = model_dir   
        if backbone not in self.model_config.keys(): 
            logger.debug(f"Backbone not Found: {backbone}")
            assert False, "backbone Not found"

        self.args = self.load_config(backbone, "config.yaml", os.path.join(self.model_dir, backbone))
        self.args.device = device
        os.makedirs(os.path.join(self.model_dir, backbone), exist_ok=True)
        self.args.backbone = backbone
        self.fun_exp_name = exp_name
        self.model = ResNet(self.args.num_classes, self.model_config[backbone]["block"], self.model_config[backbone]["layer"])
        if exp_name:
            self.model.load_state_dict(torch.load(os.path.join(self.model_dir, backbone, exp_name, filename))["state_dict"])
        else:
            # self.model.load_state_dict(model_zoo.load_url(self.model_config[backbone]['retinanet_url'])["state_dict"], model_dir=os.path.join(self.model_dir, backbone))
            self.model.load_state_dict(torch.load(self.model_config[backbone]['retinanet_url'])["state_dict"])
        self.args.exp_name = f"{os.path.join(self.model_dir, backbone)}/{exp_name}" if exp_name else \
                            f"{os.path.join(self.model_dir, backbone)}/{self.exp_name}_{len(os.listdir(os.path.join(self.model_dir, backbone)))}"
        os.makedirs(f"{self.args.exp_name }/", exist_ok=True)
        logger.info(f"Export directory: {self.args.exp_name}")
        return self()
    
    @timetaken
    def run_oneepoch(self, dataloader_train, scaler, epoch):
        epoch_loss = torch.zeros(len(dataloader_train))
        pbar = tqdm(enumerate(dataloader_train), total=len(dataloader_train), desc="Epoch: {0}".format(epoch))
        self.focalLoss = losses.FocalLoss()
        self.anchors = Anchors()
        for iteration, data in pbar:
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                regressions, classifications = self.model(data['img'].to(self.args.device).float())
                classification_loss, regression_loss = self.focalLoss(classifications, regressions, self.anchors(data['img'].to(self.args.device).float()), data['annot']) 
                classification_loss, regression_loss = classification_loss.mean(), regression_loss.mean()
                loss = classification_loss + regression_loss
                loss = loss * 2
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            epoch_loss[iteration] = loss.cpu().item()
            tb_lr = [x['lr'] for x in self.optimizer.param_groups][0]
            pbar.set_postfix_str("Loss: {:.6f} lr: {:.10f}".format(epoch_loss.sum()/iteration, tb_lr))
        return epoch_loss.mean().item()
    
    @timetaken
    def train(self, best_acc=0.0):
        dataset_train = CocoDataset(self.args.coco_path, set_name='train2017',transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        sampler = AspectRatioBasedSampler(dataset_train, batch_size=self.args.train_batchsize, drop_last=False)
        dataloader_train = DataLoader(dataset_train, num_workers=8, collate_fn=collater, batch_sampler=sampler)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay, betas=(0.9, 0.95))
        if self.fun_exp_name: self.optimizer.load_state_dict(torch.load(os.path.join(self.args.exp_name, "best.pth"))["optimizer"])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3, verbose=True)
        scaler = torch.cuda.amp.GradScaler()
        self.model = torch.nn.DataParallel(self.model); self.training = True
        for epoch in range(self.args.start_epoch, self.args.total_epoch):
            self.model.train()
            self.model.module.freeze_bn()
            epoch_loss = self.run_oneepoch(dataloader_train, scaler, epoch)
            ap50_95, ap50 = self.validation()
            self.scheduler.step()
            if ap50_95 > best_acc: 
                torch.save({
                    "iterations": epoch * len(dataloader_train),
                    "state_dict": self.model.module.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "acc": [ap50_95, ap50]
                }, f"{self.args.exp_name}/best.pth")
                best_acc = copy.deepcopy(ap50_95)
            torch.save({
                "iterations": epoch * len(dataloader_train),
                "state_dict": self.model.module.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "acc": [ap50_95, ap50]
            }, f"{self.args.exp_name}/last.pth")
            time.sleep(300)

    @torch.no_grad()
    def validation(self, model=None, threshold=0.05):
        self.model = model if isinstance(model, torch.nn.Module) else self.model
        self.model.eval().to(self.args.device)
        dataset = CocoDataset(self.args.coco_path, set_name='val2017', transform=transforms.Compose([Normalizer(), Resizer()]))
        results, image_ids = [], []
        self.anchors = Anchors()
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        for index in tqdm(range(len(dataset)), desc="Validation "):
            if index == 500 and not threshold: break
            data = dataset[index]
            scale  = data['scale']
            img_batch = data['img'].permute(2, 0, 1).to(self.args.device).float().unsqueeze(dim=0)
            anchors = self.anchors(img_batch)
            regression, classification = self.model(img_batch)
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

            finalResult = [[], [], []]
            finalScores = torch.Tensor([])
            finalAnchorBoxesIndexes = torch.Tensor([]).long()
            finalAnchorBoxesCoordinates = torch.Tensor([])
            if torch.cuda.is_available():
                finalScores = finalScores.cuda()
                finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.cuda()
                finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates.cuda()

            for i in range(classification.shape[2]):
                scores = torch.squeeze(classification[:, :, i])
                scores_over_thresh = (scores > 0.05)
                if scores_over_thresh.sum() == 0:
                    # no boxes to NMS, just continue
                    continue

                scores = scores[scores_over_thresh]
                anchorBoxes = torch.squeeze(transformed_anchors)
                anchorBoxes = anchorBoxes[scores_over_thresh]
                anchors_nms_idx = nms(anchorBoxes, scores, 0.5)

                finalResult[0].extend(scores[anchors_nms_idx])
                finalResult[1].extend(torch.tensor([i] * anchors_nms_idx.shape[0]))
                finalResult[2].extend(anchorBoxes[anchors_nms_idx])

                finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
                finalAnchorBoxesIndexesValue = torch.tensor([i] * anchors_nms_idx.shape[0])
                if torch.cuda.is_available():
                    finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.cuda()
                finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
                finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))
            scores, labels, boxes = [finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates]
            scores, labels, boxes = scores.cpu(), labels.cpu(), boxes.cpu()
            boxes /= scale
            if boxes.shape[0] > 0:
                boxes[:, 2] -= boxes[:, 0]
                boxes[:, 3] -= boxes[:, 1]
                for box_id in range(boxes.shape[0]):
                    score = float(scores[box_id])
                    label = int(labels[box_id])
                    box = boxes[box_id, :]
                    if score < 0.05: break
                    image_result = {
                        'image_id'    : dataset.image_ids[index],
                        'category_id' : dataset.label_to_coco_label(label),
                        'score'       : float(score),
                        'bbox'        : box.tolist(),
                    }
                    results.append(image_result)
            image_ids.append(dataset.image_ids[index])
        if not len(results):
            logger.debug("No Object is detected...!!!"); return 0.0, 0.0
        json.dump(results, open(f'{self.args.exp_name}/{dataset.set_name}_bbox_results.json', 'w'), indent=4)
        coco_true = dataset.coco
        coco_pred = coco_true.loadRes(f'{self.args.exp_name}/{dataset.set_name}_bbox_results.json')
        coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # ap50_95, ap50
        return coco_eval.stats[0], coco_eval.stats[1]

    @timetaken
    def export(self, dummy_input, file_path):
        self.model.eval()
        torch.onnx.export(self.model, dummy_input, file_path, opset_version=12)

    # @timetaken
    def quantize(self, dummy_input: torch.nn.Module):
        from aimet_torch.model_preparer import prepare_model
        from aimet_common.defs import QuantScheme
        from aimet_torch.quantsim import QuantizationSimModel
        device = dummy_input.device
        # dummy_input = torch.randn(dummy_input)
        model = prepare_model(self.model).to(device)
        download_file(
            "https://raw.githubusercontent.com/quic/aimet/develop/TrainingExtensions/common/src/python/aimet_common/quantsim_config/default_config_per_channel.json",
            "pcq_config.json", os.path.join(self.model_dir, self.args.backbone)
        )
        quant_sim = QuantizationSimModel(model, dummy_input=dummy_input,
                                        quant_scheme=QuantScheme.post_training_tf_enhanced,
                                        default_param_bw=8, default_output_bw=8,
                                        config_file=os.path.join(self.model_dir, self.args.backbone, "pcq_config.json"))
        quant_sim.compute_encodings(self.validation, forward_pass_callback_args=(None))
        self.validation(quant_sim.model)
        os.makedirs(os.path.join(self.args.exp_name, "quant"), exist_ok=True)
        quant_sim.model.cpu()
        quant_sim.export(path=os.path.join(self.args.exp_name, "quant"), filename_prefix=f'quantized_{self.args.backbone}', dummy_input=dummy_input.cpu())
        
    
    @timetaken
    def predict(img, save_image=True):
        return []
    
    