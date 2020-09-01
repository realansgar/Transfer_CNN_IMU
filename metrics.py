import torch
from config import DEVICE

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


def confusion(pred_y, data_y, num_classes):
  class_conf = class_confusion(pred_y, data_y, num_classes)
  return tuple(map(torch.sum, class_conf))


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
    class_weights.to(DEVICE)
  else:
    class_weights = num_classes ** -1

  class_precision, class_recall = class_precision_recall(pred_y, data_y, num_classes)

  precision = torch.sum(class_weights * class_precision)
  recall = torch.sum(class_weights * class_recall)

  return precision, recall


def class_accuracy(pred_y, data_y, num_classes):
  tp, tn, fp, fn = class_confusion(pred_y, data_y, num_classes)
  return (tp + tn) / (tp + tn + fp + fn)


def accuracy(pred_y, data_y, num_classes, weighted=True):
  if weighted:
    class_weights = torch.bincount(data_y, minlength=num_classes) / float(len(data_y))
    class_weights.to(DEVICE)
  else:
    class_weights = num_classes ** -1
  
  class_acc = class_accuracy(pred_y, data_y, num_classes)
  acc = torch.sum(class_weights * class_acc)

  return acc


def f1_score(pred_y, data_y, num_classes, weighted=True):
  if weighted:
    class_weights = torch.bincount(data_y, minlength=num_classes) / float(len(data_y))
    class_weights.to(DEVICE)
  else:
    class_weights = num_classes ** -1

  class_precision, class_recall = class_precision_recall(pred_y, data_y, num_classes)
  class_f1 = 2 * class_weights * (class_precision * class_recall) / (class_precision + class_recall)
  class_f1[torch.isnan(class_f1)] = 0
  f1 = torch.sum(class_f1)
  
  return f1