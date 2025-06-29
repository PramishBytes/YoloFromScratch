import torch
import torchvision
import numpy as np
import cv2
import torchvision.transforms.functional as F

# ----------------------------------------
# Intersection over Union
# ----------------------------------------
def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2

        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    elif box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]

        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


# ----------------------------------------
# Non-Max Suppression
# ----------------------------------------
def non_max_suppression(bboxes, iou_threshold=0.5, threshold=0.4, box_format="corners"):
    """
    bboxes: list of boxes -> [class, conf, x1, y1, x2, y2]
    """
    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    selected_boxes = []

    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0] or intersection_over_union(
                torch.tensor(chosen_box[2:]), torch.tensor(box[2:]), box_format
            ) < iou_threshold
        ]
        selected_boxes.append(chosen_box)

    return selected_boxes


# ----------------------------------------
# Convert cell predictions to box predictions
# ----------------------------------------
def cellboxes_to_boxes(predictions, S, B, C):
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, S, S, C + B * 5)

    bboxes1 = predictions[..., 21:25]
    bboxes2 = predictions[..., 26:30]
    scores = torch.cat(
        [predictions[..., 20:21], predictions[..., 25:26]], dim=-1
    )
    best_box = scores.argmax(-1).unsqueeze(-1)

    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(S).repeat(predictions.shape[0], S, 1).unsqueeze(-1)

    x = 1 / S * (best_boxes[..., 0:1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w = 1 / S * best_boxes[..., 2:3]
    h = 1 / S * best_boxes[..., 3:4]

    converted_bboxes = torch.cat((x, y, w, h), dim=-1)
    return converted_bboxes.reshape(batch_size, S * S, 4)


# ----------------------------------------
# Draw Bounding Boxes
# ----------------------------------------
def draw_bounding_boxes(image, boxes, class_labels, color=(255, 0, 0), thickness=2):
    """
    image: np.ndarray (H, W, 3)
    boxes: list of [class, confidence, x1, y1, x2, y2]
    """
    h, w = image.shape[:2]
    for box in boxes:
        cls, conf, x1, y1, x2, y2 = box
        x1 = int(x1 * w)
        y1 = int(y1 * h)
        x2 = int(x2 * w)
        y2 = int(y2 * h)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        label = f"{class_labels[int(cls)]}: {conf:.2f}"
        cv2.putText(image, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image


# ----------------------------------------
# Mean Average Precision (mAP)
# ----------------------------------------
def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, num_classes=20):
    """
    pred_boxes: list [image_idx, class_pred, prob_score, x1, y1, x2, y2]
    true_boxes: same format
    """
    average_precisions = []

    for c in range(num_classes):
        detections = [box for box in pred_boxes if box[1] == c]
        ground_truths = [box for box in true_boxes if box[1] == c]

        gt_img_count = {}
        for gt in ground_truths:
            img_id = gt[0]
            gt_img_count[img_id] = gt_img_count.get(img_id, 0) + 1

        img_gt_matched = {k: torch.zeros(v) for k, v in gt_img_count.items()}
        detections.sort(key=lambda x: x[2], reverse=True)

        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))
        total_gt = len(ground_truths)

        for i, det in enumerate(detections):
            img_id = det[0]
            best_iou = 0
            best_gt_idx = -1

            for j, gt in enumerate(ground_truths):
                if gt[0] == img_id:
                    iou = intersection_over_union(
                        torch.tensor(det[3:]), torch.tensor(gt[3:]), box_format="corners"
                    )
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j

            if best_iou > iou_threshold:
                if img_gt_matched[img_id][best_gt_idx] == 0:
                    TP[i] = 1
                    img_gt_matched[img_id][best_gt_idx] = 1
                else:
                    FP[i] = 1
            else:
                FP[i] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + 1e-6)
        recalls = TP_cumsum / (total_gt + 1e-6)
        precisions = torch.cat([torch.tensor([1]), precisions])
        recalls = torch.cat([torch.tensor([0]), recalls])
        average_precision = torch.trapz(precisions, recalls)
        average_precisions.append(average_precision)

    return sum(average_precisions) / len(average_precisions)

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, bboxes):
        for t in self.transforms:
            image, bboxes = t(image, bboxes)
        return image, bboxes

class Resize:
    def __init__(self, size):
        self.size = size  # (width, height)

    def __call__(self, image, bboxes):
        image = F.resize(image, self.size)
        return image, bboxes

class ToTensor:
    def __call__(self, image, bboxes):
        image = F.to_tensor(image)  # Converts PIL to (C, H, W)
        return image, bboxes.clone().detach()
