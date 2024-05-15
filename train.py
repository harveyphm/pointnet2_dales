from glob import glob
import os.path as osp
import os
import numpy as np
from tqdm.auto import tqdm


import torch
import torch.nn.functional as F
from torchmetrics.functional import jaccard_index

from models import PointNet
from data.loader import convertData

## Config ``````````````````````````````````````````
NUM_CLASS = 9
NUM_FEATURES = 3
LOAD_CHECKPOINT = False
TILES_SIZE = 15
TRAIN_DIR = f"dales_tiled/tiles_{TILES_SIZE}/train/" 
TEST_DIR = f"dales_tiled/tiles_{TILES_SIZE}/test/" 
## `````````````````````````````````````````````````

# Model and optimizer
torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PointNet(NUM_CLASS, NUM_FEATURES).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


if LOAD_CHECKPOINT:
    #Load from checkpoint
    checkpoint = torch.load('models/deep_relu.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']


#Load data
all_train_files = os.listdir(osp.abspath(TRAIN_DIR))
train_batches = []
for laspath in tqdm(all_train_files, total = len(all_train_files)):
    las_data = convertData( os.path.join(TRAIN_DIR, laspath))
    if las_data.pos.shape[0] > 1:
        train_batches.append(las_data)


# Training loop
def train(verbose = True):
    model.train()

    total_loss = correct_nodes = total_nodes = 0
    accuracy = 0.0
    ious = []
    
    train_log = tqdm(enumerate(train_batches), total = len(train_batches), position = 1)
    
    for i, data in train_log:
        try:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct_nodes += out.argmax(dim=1).eq(data.y).sum().item()
            total_nodes += data.num_nodes

            accuracy += correct_nodes / total_nodes
            iou = jaccard_index(out.argmax(dim=-1), data.y,
                                    num_classes=model.num_classes, task='multiclass')
            ious.append(iou)

            torch.cuda.empty_cache() 

            if i  % 10 == 0:
                if verbose:
                    train_log.set_description_str(f' Loss: {total_loss / 10:.4f} -'
                          f' Train Acc: {correct_nodes / total_nodes:.4f} -'
                            f' Train IoU: {iou:.4f} ')
                total_loss = correct_nodes = total_nodes = 0
        except:
            print(data)
        

    accuracy /= len(all_train_files)
    iou = torch.tensor(ious, device=device)
    return accuracy, total_loss, iou

# Testing loop
@torch.no_grad()
def test(test_dir,subset = None, verbose  = False):
    model.eval()
    
    total_loss = correct_nodes = total_nodes = 0
    accuracy = 0.0
    ious = []
    
    all_test_files = os.listdir(osp.abspath(test_dir))
    
    if subset is not None: 
        all_test_files = np.random.choice(all_test_files, 10)
        
    
    for i, test_file in enumerate(all_test_files):
        data = convertData( os.path.join(test_dir, test_file))
        data = data.to(device)
        out = model(data)
        
        loss = F.nll_loss(out, data.y)
        total_loss += loss.item()
        
        correct_nodes += out.argmax(dim=1).eq(data.y).sum().item()
        total_nodes += data.num_nodes
        
        accuracy += correct_nodes / total_nodes
        
        iou = jaccard_index(out.argmax(dim=-1), data.y,
                                num_classes=model.num_classes, task='multiclass')
        ious.append(iou)

    iou = torch.tensor(ious, device=device)
    accuracy /= len(all_test_files)
    total_loss /= len(all_test_files)
    return accuracy,  total_loss, float(iou.mean())  # Global IoU.


# Get inital performance of the models 
test_acc, test_loss, test_iou = test(TEST_DIR, subset = 10)
print(f'Initial Test Accuracy: {test_acc:.4f} - Test Loss: {test_loss:.4f} - Test IoU: {test_iou:.4f}')
print('Training...')


try:
    best_iou = checkpoint['test_iou']
    print(f"Loading previous test iou from check point: {best_iou}")

except:
    print('Initialize test IoU = 0.0') 
    best_iou = 0.0

try:
    print(f"Loading previous accuracy from check point: {checkpoint['accuracy']}")
    best_acc = checkpoint['accuracy']
except:
    print('Initialize accuracy = 0.0') 
    best_acc = 0.0

epoch_log = tqdm(range(1, 31), total = len(range(1, 31)), position = 0)
epoch_log.set_description_str(f'Epoch: 1, Acc: {best_acc:.2f} Test IoU: {best_iou:.4f}')

for epoch in epoch_log:
    acc, loss, train_iou = train()
    test_acc, test_loss, test_iou = test(TEST_DIR, subset = 10)
    
    epoch_log.set_description_str(f'Epoch: {epoch:02d}, Acc: {test_acc:.2f} Test IoU: {test_iou:.4f}')
    if test_iou>best_iou or test_acc> best_acc: 
        best_iou = test_iou if best_iou < test_iou else best_iou
        best_acc = acc if best_acc < acc else best_acc
        #Save model
        print(f"Saving best model with IoU:{test_iou:.4f} and Acc:{acc:.2f}")
        torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'accuracy': best_acc, 
                 'test_iou': best_iou}, 
            'models/best.pt')