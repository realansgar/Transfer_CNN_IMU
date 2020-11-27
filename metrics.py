import torch
from torch.functional import F
from config import DEVICE, MAX_BATCH_SIZE
from sklearn.metrics import confusion_matrix
import numpy as np

def class_confusion(pred_y, data_y, num_classes):
  tp = torch.zeros(num_classes, device=DEVICE)
  tn = torch.zeros(num_classes, device=DEVICE)
  fp = torch.zeros(num_classes, device=DEVICE)
  fn = torch.zeros(num_classes, device=DEVICE)

  for c in range(num_classes):
    selected = torch.where(pred_y == c)[0]
    not_selected = torch.where(pred_y != c)[0]

    tp[c] = torch.sum(data_y[selected] == c)
    tn[c] = torch.sum(data_y[not_selected] != c)
    fp[c] = torch.sum(data_y[selected] != c)
    fn[c] = torch.sum(data_y[not_selected] == c)
  
  return tp, tn, fp, fn


def class_precision_recall(pred_y, data_y, num_classes):
  tp, _, fp, fn = class_confusion(pred_y, data_y, num_classes)

  class_precision = tp / (tp + fp)
  class_recall = tp / (tp + fn)
  class_precision[torch.isnan(class_precision)] = 0
  class_recall[torch.isnan(class_recall)] = 0
  
  return class_precision, class_recall


def precision_recall(pred_y, data_y, num_classes, weighted=True):
  if weighted:
    class_weights = torch.bincount(data_y, minlength=num_classes) / float(len(data_y))
    class_weights = class_weights.to(DEVICE)
  else:
    class_weights = num_classes ** -1

  class_precision, class_recall = class_precision_recall(pred_y, data_y, num_classes)

  precision = torch.sum(class_weights * class_precision)
  recall = torch.sum(class_weights * class_recall)

  return precision, recall


def class_accuracy(pred_y, data_y, num_classes):
  tp, tn, fp, fn = class_confusion(pred_y, data_y, num_classes)
  return (tp + tn) / (tp + tn + fp + fn)


def accuracy(pred_y, data_y, num_classes, weighted=False):
  if weighted:
    class_weights = torch.bincount(data_y, minlength=num_classes) / float(len(data_y))
    class_weights = class_weights.to(DEVICE)
  else:
    class_weights = num_classes ** -1
  
  class_acc = class_accuracy(pred_y, data_y, num_classes)
  acc = torch.sum(class_weights * class_acc)

  return acc

def class_f1_score(pred_y, data_y, num_classes):
  class_precision, class_recall = class_precision_recall(pred_y, data_y, num_classes)
  class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall)
  class_f1[torch.isnan(class_f1)] = 0
  return class_f1

def f1_score(pred_y, data_y, num_classes, weighted=False):
  if weighted:
    class_weights = torch.bincount(data_y, minlength=num_classes) / float(len(data_y))
    class_weights = class_weights.to(DEVICE)
  else:
    class_weights = num_classes ** -1

  class_f1 = class_f1_score(pred_y, data_y, num_classes)
  weighted_class_f1 = class_weights * class_f1
  f1 = torch.sum(weighted_class_f1)
  
  return f1


def evaluate_net(net, criterion, batch, num_classes):
  data_x, data_y = batch
  data_x = data_x.to(DEVICE)
  data_y = data_y.to(DEVICE)
  net = net.to(DEVICE)

  net.eval()
  with torch.no_grad():
    chunks_y = []
    for chunk_x in torch.split(data_x, MAX_BATCH_SIZE, dim=0):
      chunk_y = net(chunk_x)
      chunks_y.append(chunk_y)
    raw_y = torch.cat(chunks_y, dim=0)
    loss = criterion(raw_y, data_y)
    prob_y = F.softmax(raw_y, dim=1)
    pred_y = torch.argmax(prob_y, dim=1)

    precision, recall = precision_recall(pred_y, data_y, num_classes)
    weighted_precision, weighted_recall = precision_recall(pred_y, data_y, num_classes, weighted=True)
    micro_acc = torch.sum(pred_y == data_y) / float(len(data_y))
    class_acc = class_accuracy(pred_y, data_y, num_classes)
    acc = accuracy(pred_y, data_y, num_classes)
    class_f1 = class_f1_score(pred_y, data_y, num_classes)
    weighted_acc = accuracy(pred_y, data_y, num_classes, weighted=True)
    f1 = f1_score(pred_y, data_y, num_classes)
    weighted_f1 = f1_score(pred_y, data_y, num_classes, weighted=True)
    confusion = np.transpose(confusion_matrix(data_y.cpu().numpy(), pred_y.cpu().numpy(), normalize="true", labels=range(num_classes)))

    return {
      "loss": loss,
      "class_accuracy": class_acc,
      "micro_accuracy": micro_acc,
      "accuracy": acc,
      "weighted_accuracy": weighted_acc,
      "precision": precision,
      "weighted_precision": weighted_precision,
      "recall": recall,
      "weighted_recall": weighted_recall,
      "class_f1": class_f1,
      "f1": f1,
      "weighted_f1": weighted_f1,
      "confusion": confusion
    }
    