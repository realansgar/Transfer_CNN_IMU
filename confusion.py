from argparse import ArgumentParser, FileType
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import config
from datasets import HARWindows
import metrics

# from https://github.com/DTrimarchi10/confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=False,
                          cbar=False,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=False,
                          figsize=None,
                          cmap='Blues',
                          title=None,
                          filepath=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.

    Arguments
    ---------
    cf:            confusion matrix to be passed in

    group_names:   List of strings that represent the labels row by row to be shown in each square.

    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'

    count:         If True, show the raw number in the confusion matrix. Default is True.

    normalize:     If True, show the proportions for each category. Default is True.

    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.

    xyticks:       If True, show x and y ticks. Default is True.

    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.

    sum_stats:     If True, display summary statistics below the figure. Default is True.

    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.

    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.

    '''


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:.1%}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.set(font_scale=2)
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)
    if filepath is not None:
        plt.savefig(filepath, orientation="landscape", bbox_inches="tight")
    else:
        plt.show()


def log_confusion(dataset, filepaths):
  for filepath in filepaths:
    eval_dict = torch.load(filepath, map_location=config.DEVICE)
    val_set_filepath = getattr(config, dataset + "_BASEPATH") + os.path.basename(eval_dict["config"]["VAL_SET_FILEPATH"])
    val_set = HARWindows(val_set_filepath)
    val_dataloader = DataLoader(val_set, batch_size=len(val_set))
    eval_val = metrics.evaluate_net(eval_dict["net"], torch.nn.CrossEntropyLoss(), next(iter(val_dataloader)), eval_dict["config"]["NUM_CLASSES"])
    np.set_printoptions(precision=4, linewidth=200, suppress=True)
    print(eval_dict["config"]["NAME"])
    print(f"wF1: {eval_dict['best_val']['weighted_f1']}")
    print(eval_val["confusion"])
    save_filepath = None
    if args.s:
      save_filepath = os.path.splitext(filepath)[0] + ".pdf"
    make_confusion_matrix(eval_val["confusion"], figsize=(12,7), categories=getattr(config, dataset + "_LABEL_NAMES").values(), filepath=save_filepath)


if __name__ == "__main__":
  parser = ArgumentParser(description="Display logs and test saved model")
  parser.add_argument("dataset", help="the dataset to use")
  parser.add_argument("files", type=FileType("r"), nargs="*", help="the files to log or test")
  parser.add_argument("-s", action="store_true", help="saves the confusion matrix to a pdf")
  args = parser.parse_args()
  f_paths = [file.name for file in args.files]
  log_confusion(args.dataset, f_paths)
