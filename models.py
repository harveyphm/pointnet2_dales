import torch
from torch_geometric.nn import MLP, PointNetConv, fps, global_max_pool, radius,knn_interpolate



class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch
    
class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch
    
class FPModule(torch.nn.Module):
    def __init__(self, k, nn):
        super().__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip
    

class PointNet(torch.nn.Module):
    def __init__(self, num_classes , num_features, ratio = [0.5, 0.5, 0.5], r = [0.2, 0.2, 0.2],  act = 'relu'):
        super().__init__()
        self.num_classes = num_classes
        self.num_feature = num_features
        # Input channels account for both pos and node features.
        self.sa1_module = SAModule(ratio[0], r[0], MLP([3 + num_features, 64, 64, 128], act = act ))
        self.sa2_module = SAModule(ratio[1], r[1], MLP([128 + 3, 128, 128, 256], act = act))
        self.sa3_module = SAModule(ratio[2], r[2], MLP([256 + 3, 256, 256, 512], act = act))
        self.sa4_module = GlobalSAModule(MLP([512 + 3, 512, 512, 1024], act = act))
        
        self.fp4_module = FPModule(1, MLP([1024 + 512, 512 , 512], act = act))
        self.fp3_module = FPModule(3, MLP([512 + 256, 256, 256], act = act))
        self.fp2_module = FPModule(3, MLP([256 + 128, 256, 256], act = act))
        self.fp1_module = FPModule(3, MLP([256 + num_features, 128, 128, 128], act = act))

        self.mlp = MLP([128, 128, 128, num_classes], act = "sigmoid", norm=None)

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        sa4_out = self.sa4_module(*sa3_out)
        
        fp4_out = self.fp4_module(*sa4_out, *sa3_out)
        fp3_out = self.fp3_module(*fp4_out, *sa2_out)
        fp2_out = self.fp2_module(*fp3_out, *sa1_out)
        x, _, _ = self.fp1_module(*fp2_out, *sa0_out)

        return self.mlp(x).log_softmax(dim=-1)
    

def create_model( sampling_ratio = [0.25, 0.25, 0.25 ], ball_radius = [0.2, 0.4, 0.4], act = 'relu', num_classes = 9, num_features = 1): 
    return PointNet(ratio = sampling_ratio, 
                   r = ball_radius, 
                   num_classes = num_classes, 
                   num_features = num_features, 
                   act = act)