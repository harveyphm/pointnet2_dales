import laspy as lp
import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data



def convertData(laspath, debug = False, transform_bool = True ):
    
    pc = lp.read(laspath)
    
    # Define preprocessing steps
    transform = T.Compose([
        T.RandomJitter(0.01),
        T.RandomRotate(15, axis=0),
        T.RandomRotate(15, axis=1),
        T.RandomRotate(15, axis=2)
        ])
    pre_transform = T.NormalizeScale()
    
    # Get imformation from las file
    coords = np.vstack((pc.x, pc.y, pc.z)).transpose()    
    scales = pc.header.scales
    offsets= pc.header.offsets    
    
    labels = pc.classification.array
    features = pc.intensity if np.max(pc.intensity) > 0 else np.ones_like(pc.intensity, dtype = np.uint8)
    if len(features.shape) <2: 
        features = features[:, np.newaxis]
    num_classes = len(np.unique(labels))
    
    data = Data(x = torch.from_numpy(coords).type(torch.FloatTensor),  
                    pos = torch.from_numpy(coords).type(torch.FloatTensor), 
                    y = torch.from_numpy(labels), 
                    num_classes=num_classes,
                    num_features = coords.shape[-1],
                    # transform = transform, 
                    # pre_transform= pre_transform,
               batch = torch.from_numpy(np.zeros_like(features,dtype = np.int64)).flatten()
               )
    if transform_bool: 
        return transform( pre_transform(data))
    else:
        return data