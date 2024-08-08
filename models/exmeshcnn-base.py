import torch.nn as nn
import torch
from exmeshcnn.layers.geodesic import Geodesic
from exmeshcnn.layers.geometric import Geometric
from exmeshcnn.layers.meshconv import MeshConv

class ExMeshCNN(nn.Module):
    """
    ed: edge feature
    fa: face feature
    ad: adjacent face list
    """
    def __init__(self, tag_data, params):
        super().__init__()
        
        # Convolutional block declarations
        self.conv_e = Geodesic(128,64)
        self.conv_f = Geometric(128,64)
        self.conv1 = MeshConv(128,128)
        self.conv2 = MeshConv(128,256)
        self.conv3 = MeshConv(256,256)
        self.conv4 = MeshConv(256,512)

        # Fully Connected block declarations
        self.fcn_branches = []
        self.avg_pool_branches = []

        for _, values in tag_data.items():
            self.fn = nn.Sequential(
            nn.Conv1d(in_channels=512 , out_channels=values["classes"] - 1, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(values["classes"] - 1))

            # self.fcn_branches.append(act_fn)
            self.avgp = nn.AdaptiveAvgPool1d(1)

    def forward(self, ed, fa, ad):
        ed = self.conv_e(ed)
        fa = self.conv_f(fa)
        fe = torch.cat([ed,fa],dim=1)
        fe = self.conv1(fe, ad)
        fe = self.conv2(fe, ad)
        fe = self.conv3(fe, ad)
        fe = self.conv4(fe, ad)
        
        # preds = []
        # for fn, avgp in zip(self.fcn_branches, self.avg_pool_branches):
        fe_out = self.fn(fe)
        fe_out = self.avgp(fe_out)
        fe_out = fe_out.view(fe_out.size(0), -1)
            # preds.append(fe_out)

        return torch.stack((fe_out,))


def get_model(tag_data, device, opt_sel = None, options = None):

    params = None
    if options is not None:
        pass

    model = ExMeshCNN(tag_data, params).to(device)

    return model