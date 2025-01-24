import numpy as np
import pandas as pd
import torch
import os

from sklearn.preprocessing import OneHotEncoder

class SingleImgDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, mode):
        self.root_dir = root_dir
        class_names = sorted(os.listdir(self.root_dir))
        self.classnames = class_names
        self.mode = mode
        
        self.filepaths = []
        for class_name in class_names:
            path = os.path.join(root_dir, class_name, mode)
            obj_name = np.sort(os.listdir(path))
            for obj in obj_name:
                obj_path = os.path.join(path, obj)
                self.filepaths.append(obj_path)
        
    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        class_name = path.split('/')[2]
        class_id = self.classnames.index(class_name)
        
        d_1 = np.load(self.filepaths[idx], allow_pickle=True)
        ed = torch.from_numpy(d_1.item()['edge'])
        fa = torch.from_numpy(d_1.item()['face'])
        ad = d_1.item()['adj']
        return (class_id, ed , fa, ad)


class PSDataset(torch.utils.data.Dataset):
    """
    Dataset loader for the Pubic Symphysis data
    """

    def __init__(self, x_names : np.array, y_tags : pd.DataFrame, path : str, tag_data : dict, npy_name : str = "npy", ):
        self.bone_id = x_names
        self.path = path
        self.npy_name = npy_name
        self.tags = {}

        for key, _ in tag_data.items():
            act_tag = y_tags[key].to_list()

            # if val["classes"] > 2:
            #     tag_encoder = OneHotEncoder(categories = list(range(val["classes"])))
            #     act_tag = tag_encoder.fit_transform(np.reshape(act_tag, (-1,1)))

            self.tags[key] = act_tag

    def __len__(self):
        return len(self.bone_id)
    
    def __getitem__(self, idx):

        # Load the "mesh"
        pseudo_mesh = np.load(os.path.join(self.path, self.npy_name, f"{self.bone_id[idx]}.npy"), allow_pickle=True)
        edge_feat = torch.from_numpy(pseudo_mesh.item()['edge'])
        face_feat = torch.from_numpy(pseudo_mesh.item()['face'])
        adyacent_faces = pseudo_mesh.item()['adj']

        # Get the tags:
        # Structure of dictionary is
        # { "tag_1" : <value_at_idx>,
        #   "tag_2" : <value_at_idx>,
        #    ...
        #   "tag_n" : <value_at_idx> }
        bone_tag = {}

        for key, val in self.tags.items():
            bone_tag[key] = val[idx]

        return edge_feat, face_feat, adyacent_faces, bone_tag, self.bone_id[idx]
    