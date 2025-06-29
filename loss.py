import torch 
import torch.nn as nn
from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super().__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        # Get IOUs for both predicted boxes
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        # Find which predicted box has higher IoU
        iou_maxes, best_box = torch.max(ious, dim=0)
        exists_box = target[..., 20].unsqueeze(3)  # Iobj_i

        # ============================== #
        #     FOR BOX COORDINATES        #
        # ============================== #

        box_predictions = exists_box * (
            best_box * predictions[..., 26:30] +
            (1 - best_box) * predictions[..., 21:25]
        )
        box_targets = exists_box * target[..., 21:25]

        # Take sqrt on widths/heights **without in-place modification**
        box_pred_wh = torch.sqrt(torch.abs(box_predictions[..., 2:4].clone()) + 1e-6)
        box_target_wh = torch.sqrt(torch.abs(box_targets[..., 2:4].clone()) + 1e-6)

        # Combine with xy coords (no sqrt)
        box_pred = torch.cat([box_predictions[..., :2], box_pred_wh], dim=-1)
        box_target = torch.cat([box_targets[..., :2], box_target_wh], dim=-1)

        box_loss = self.mse(
            torch.flatten(box_pred, end_dim=-2),
            torch.flatten(box_target, end_dim=-2)
        )

        # ============================== #
        #     FOR OBJECT LOSS            #
        # ============================== #   

        pred_box_conf = (
            best_box * predictions[..., 25:26] +
            (1 - best_box) * predictions[..., 20:21]
        )

        object_loss = self.mse(
            torch.flatten(exists_box * pred_box_conf),
            torch.flatten(exists_box * target[..., 20:21])
        )

        # ============================== #
        #     FOR NO OBJECT LOSS         #
        # ============================== #

        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        ) + self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        # ============================== #
        #        FOR CLASS LOSS          #
        # ============================== #

        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :self.C], end_dim=-2),
            torch.flatten(exists_box * target[..., :self.C], end_dim=-2)
        )

        # ============================== #
        #           FINAL LOSS           #
        # ============================== #

        loss = (
            self.lambda_coord * box_loss +
            object_loss +
            self.lambda_noobj * no_object_loss +
            class_loss
        )

        return loss
