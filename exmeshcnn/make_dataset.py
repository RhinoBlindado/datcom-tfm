import trimesh as tm
import torch
import numpy as np
import os
from utils.functions import *

import argparse
import glob
from tqdm import tqdm

# # Data pre-processing towards making the entire training easier.
# dataset = "./dataset"
# database = './Nodule-98-30K-Remesh' # The number of fases for each mesh must be the same with each other. SWs such as MeshLab can be used.

# # ModelNet40, Manifold40: 1024
# # Cube, Shrec: 500
# target_face_num = 20000 

# # database = database+"_"+str(target_face_num)
# # database = 


# class_names = np.sort(os.listdir(database))
# modes = ["train", "test"]

def to_npy(dataset_path, target_face_num, force_overwrite = False):

    obj_files = glob.glob(os.path.join(dataset_path, "obj", "*.obj"))

    pbar = tqdm(initial=0, total=len(obj_files), unit=" objs")

    for obj in obj_files:

        obj_fname = os.path.split(obj)[-1].split(".")[0]
        npy_path = os.path.join(dataset_path, "npy", obj_fname) + ".npy"

        if(os.path.isfile(npy_path) and not force_overwrite):
            print("Mesh {} already has NPY. Skip.".format(obj_fname))
            pbar.update(1)
            continue

        mesh = tm.load_mesh(obj, process=False)
        
        verts = mesh.vertices
        faces = mesh.faces
        
        # normalization
        centroid = mesh.centroid
        verts = verts - centroid
        max_len = np.max(verts[:, 0]**2 + verts[:, 1]**2 + verts[:, 2]**2)
        verts /= np.sqrt(max_len)
        mesh = tm.Trimesh(verts, faces, process=False)

        verts = mesh.vertices
        faces = mesh.faces
        norms = mesh.vertex_normals
        
        faces_t = torch.from_numpy(faces)
        verts_t = torch.from_numpy(verts)

        face_adj = np.copy(mesh.face_adjacency)
        adjs = torch.from_numpy(face_adj)
        adj_list = get_adj_nm(adjs)

        if adj_list is None:
            print("Mesh {} has problems. Skip.".format(obj_fname))
            pbar.update(1)
            continue  
        else:
            adj_list = adj_list.long()
        
        # extract edge feature
        norm_t = torch.from_numpy(norms[faces_t[adj_list[:,0]]])
        in_face = faces_t[adj_list].clone()
        size = len(in_face)
        edges_t = torch.stack([get_edges(in_face[i], verts_t) for i in range(size)])
        edges_t = edges_t.reshape(-1,3,6)
        face_centroid = torch.mean(verts_t[faces_t[adj_list[:,0]]],dim=1)
        facen=face_centroid.unsqueeze(1).repeat(1,3,1)
        facened = facen - verts_t[faces_t[adj_list[:,0]]]
        edge_feature = torch.cat([edges_t, facened, norm_t],dim=2)
        edge_feature = np.float16(edge_feature.detach().numpy())
        
        # extract face feature
        adj_list = adj_list.detach().numpy()
        normals = mesh.face_normals[adj_list]
        faces_t = faces_t.detach().numpy()
        verts_t = verts_t.detach().numpy()
        points = verts_t[faces_t[adj_list]].reshape(-1,4,9)
        face_feature = np.concatenate((points, normals), axis=2)
        
        if (len(edge_feature) == target_face_num) and (len(adj_list) == target_face_num):
            d1={'edge':edge_feature, 'adj':adj_list, 'face':face_feature}
            np.save(npy_path, d1)
        else:
            print("Discarded, mesh has {} edge features and {} adjacents, needed is {}".format(len(edge_feature), len(adj_list), target_face_num))
        pbar.update(1)

    pbar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset folder. Inside there must be a CSV file named 'dataset.csv' and a 'obj' folder containing the meshes.")
    parser.add_argument("--num-faces", type=int, required=True, help="Number of desired faces to use.")
    parser.add_argument("--force", action="store_true", help="If set, any NPY file already existing will be overwritten.")

    args = parser.parse_args()

    to_npy(args.dataset, args.num_faces, args.force)