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
        
        # Input block
        # - Input and middle layers
        self.conv_e = Geodesic(params["geod_mid_channel"],
                               params["geod_out_channel"])
        self.conv_f = Geometric(params["geom_mid_channel"],
                                params["geom_out_channel"])
        
        conv_in_channel = params["geod_out_channel"] + params["geom_out_channel"]

        # Mesh Convolution block
        # - How many layers
        conv_layer_count = params["conv_layer_num"]

        # - How many channels in each layer?
        self.meshconv_layers = nn.ModuleList()
        for layer in range(conv_layer_count):
            conv_out_channel = params[f"conv{layer}_out_channel"]
            self.meshconv_layers.append(MeshConv(conv_in_channel, conv_out_channel))
            
            conv_in_channel = conv_out_channel

        fn_in_channel = params[f"conv{conv_layer_count - 1}_out_channel"]

        # Pooling bridge
        self.pool_bridge = nn.Sequential(nn.Conv1d(in_channels=conv_in_channel , out_channels=fn_in_channel, kernel_size=1, stride=1, bias=False),
                                         nn.BatchNorm1d(fn_in_channel),
                                         nn.AdaptiveAvgPool1d(1))

        # Fully Connected block
        self.fn_layers = nn.ModuleDict()
        for tag, values in tag_data.items():

            # - How many layers
            fn_layer_count = params["fn_layer_num"]

            act_fn_list = nn.ModuleList()

            for layer in range(fn_layer_count):
                fn_out_channel = params[f"fn{layer}_out_channel"]
                act_fn_list.append(nn.Linear(fn_in_channel, fn_out_channel))
                act_fn_list.append(nn.ReLU())

                fn_in_channel = fn_out_channel

            if values["classes"] > 2:
                fn_out_channel = values["classes"]
            else:
                fn_out_channel = 1
    
            act_fn_list.append(nn.Linear(fn_in_channel, fn_out_channel))

            self.fn_layers[tag] = nn.Sequential(*act_fn_list)

    def forward(self, ed, fa, ad):
        ed = self.conv_e(ed)
        fa = self.conv_f(fa)
        fe = torch.cat([ed,fa],dim=1)

        # Pass data into Mesh Convolutions fields...
        for meshconv in self.meshconv_layers:
            fe = meshconv(fe, ad)

        # ...then through the pooling bridge...
        fe = self.pool_bridge(fe)
        fe = fe.view(fe.size(0), -1)

        # ...and finally into FN city.
        preds = {}
        for tag, fn_branch in self.fn_layers.items():
            fe_out = fn_branch(fe)
            preds[tag] = fe_out

        return preds


def get_model(tag_data, trial = None, params = None):

    if trial is not None:
        params = {}
        # Input block
        params["geod_mid_channel"] = trial.suggest_categorical("geod_mid_channel", [16, 32, 64, 128, 256, 512])
        params["geod_out_channel"] = trial.suggest_categorical("geod_out_channel", [16, 32, 64, 128, 256, 512])
        params["geom_mid_channel"] = trial.suggest_categorical("geom_mid_channel", [16, 32, 64, 128, 256, 512])
        params["geom_out_channel"] = trial.suggest_categorical("geom_out_channel", [16, 32, 64, 128, 256, 512])
        
        # Mesh Convolution block
        params["conv_layer_num"] = trial.suggest_int("conv_layer_num", 2, 6)

        for layer in range(params["conv_layer_num"]):
            params[f"conv{layer}_out_channel"]  = trial.suggest_categorical(f"conv{layer}_out_channel", [16, 32, 64, 128, 256, 512])
        
        # Fully Connected block
        params["fn_layer_num"] = trial.suggest_int("fn_layer_num", 1, 5)

        for layer in range(params["fn_layer_num"]):
            params[f"fn{layer}_out_channel"]  = trial.suggest_categorical(f"fn{layer}_out_channel", [8, 16, 32, 64, 128, 256])
        

    model = ExMeshCNN(tag_data, params)

    return model