
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import (
    FastRCNNOutputLayers, FastRCNNOutputs, fast_rcnn_inference,
    fast_rcnn_inference_single_image)


class FRCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # print(self.output_layers)
        self.cfg = get_cfg()
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_3x.yaml"))

        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_C4_3x.yaml")

        self.score_tresh = 0.0001
        self.nms_tresh = 0.001
        self.box2box_transform = Box2BoxTransform(weights=self.cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)
        self.predictor = DefaultPredictor(self.cfg)

        self.NUM_OBJECTS = 6

    def forward(self, x):
     #   shapes = [(z.size(1), z.size(2)) for z in x]
        with torch.no_grad():
            images = self.predictor.model.preprocess_image([{"image": z} for z in x])

            # Run Backbone Res1-Res4
            features = self.predictor.model.backbone(images.tensor)

            # Generate proposals with RPN
            proposals, _ = self.predictor.model.proposal_generator(images, features, None)
            #proposal = proposals[0]
            # print('Proposal Boxes size:', proposal.proposal_boxes.tensor.shape)

            # Run RoI head for each proposal (RoI Pooling + Res5)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            features = [features[f] for f in self.predictor.model.roi_heads.in_features]
            box_features = self.predictor.model.roi_heads._shared_roi_transform(
                features, proposal_boxes
            )
            feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
            # print('Pooled features size:', feature_pooled.shape)

            # Predict classes and boxes for each proposal.
            pred_class_logits, pred_proposal_deltas = self.predictor.model.roi_heads.box_predictor(feature_pooled)
            outputs = FastRCNNOutputs(
                self.box2box_transform,
                pred_class_logits,
                pred_proposal_deltas,
                proposals,
                self.cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA,
            )
            probs = outputs.predict_probs()
            boxes = outputs.predict_boxes()

            # # Note: BUTD uses raw RoI predictions,
            # #       we use the predicted boxes instead.
            # boxes = proposal_boxes[0].tensor

            # NMS
            out_features = []
            for i, (prob, box) in enumerate(zip(probs, boxes)):
                for nms_thresh in np.arange(self.nms_tresh, 1.0, 0.1):
                    instances, ids = fast_rcnn_inference_single_image(
                        box, prob, images.image_sizes[i],
                        score_thresh=self.score_tresh, nms_thresh=nms_thresh, topk_per_image=self.NUM_OBJECTS
                    )
                    if len(ids) == self.NUM_OBJECTS:
                        break

                # #instances = detector_postprocess(instances, raw_height, raw_width)

                roi_features = feature_pooled[ids].detach()
                roi_features = roi_features.permute(1, 0)
                out_features.append(roi_features)
                # #print(instances)

            return torch.stack(out_features)
