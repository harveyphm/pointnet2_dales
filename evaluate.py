from glob import glob
import os.path as osp
import os
import numpy as np
from tqdm.auto import tqdm


import torch


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import  ConfusionMatrixDisplay
from models import create_model
from data.loader import convertData


@torch.no_grad()
def evaluate(subset = 10):
        
    model.eval()
    all_test_files = os.listdir(osp.abspath(TEST_DIR))
    if subset is not None: 
        all_test_files = np.random.choice(all_test_files, 10)
        
    outs= []
    ys  = []
    
    for i, test_file in tqdm(enumerate(all_test_files), total = len(all_test_files)):
        data = convertData( os.path.join(TEST_DIR, test_file))
        data = data.to(device)
        outs.append(   model(data).argmax(dim = 1).cpu().numpy().astype(np.uint8))
        ys.append(data.y.cpu().numpy().astype(np.uint8))
    
    outs = np.hstack(outs, dtype = np.uint8)
    ys = np.hstack(ys, dtype = np.uint8)
    
    confusions = confusion_matrix(ys, outs)
    
    ConfusionMatrixDisplay.from_predictions( ys, outs, normalize = "true", display_labels=["Unknown", "Ground", "Vegetation", "Car", "Truck", "Powerline", "Fences", "Poles", "Building"])
    
    TP = np.diagonal(confusions, axis1=-2, axis2=-1)
    TP_plus_FP = np.sum(confusions, axis=-1)
    TP_plus_FN = np.sum(confusions, axis=-2)

    # Compute precision and recall. This assume that the second to last axis counts the truths (like the first axis of
    # a confusion matrix), and that the last axis counts the predictions (like the second axis of a confusion matrix)
    precision = TP / (TP_plus_FN + 1e-6)
    recall = TP / (TP_plus_FP + 1e-6)

    # Compute Accuracy
    acc = np.sum(TP, axis=-1) / (np.sum(confusions, axis=(-2, -1)) + 1e-6)

    # Compute F1 score
    f1 = 2 * TP / (TP_plus_FP + TP_plus_FN + 1e-6)

    # Compute IoU
    iou = f1 / (2 - f1)

    return confusions, precision, recall, f1, iou, acc


if __name__ == "__main__":
    TILES_SIZE = 15
    TEST_DIR = f"dales_tiled/tiles_{TILES_SIZE}/test/" 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(
    sampling_ratio = [0.5]*3, 
    ball_radius = [0.2 , 0.2*2 , 0.2*2],
    act = 'relu' ).to(device)
    checkpoint = torch.load('models/deep_relu.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    confusions, precision, recall, f1, iou, acc = evaluate()
    print(f"Accuracy: {acc.mean():.4f} - IoU: {iou.mean():.4f}")
    print(f"Precision: {precision.mean():.4f} - Recall: {recall.mean():.4f}")
    print(f"F1: {f1.mean():.4f}")
    print("Confusion Matrix: ")
    print(confusions)
    plt.show()