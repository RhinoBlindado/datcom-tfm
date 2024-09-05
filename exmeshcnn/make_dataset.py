
import multiprocessing as mp
import argparse
import glob
import time

import trimesh as tm
import torch
import numpy as np
import os
from utils.functions import *

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

def to_npy(obj, dataset_folder, target_face_num, force_overwrite, npy_folder_name):

    start_t = time.time()
    # obj_files = glob.glob(os.path.join(obj_folder, "*.obj"))

    # pbar = tqdm(initial=0, total=len(obj_files), unit="objs")
    
    # for obj in obj_files:

    obj_fname = os.path.split(obj)[-1].split(".")[0]
    print(f"[{obj_fname}]: Start...")
    npy_folder_path = os.path.join(dataset_folder, npy_folder_name)
    npy_path = os.path.join(npy_folder_path, obj_fname) + ".npy"

    os.makedirs(npy_folder_path, exist_ok=True)

    if(os.path.isfile(npy_path) and not force_overwrite):
        end_t = time.time()
        time_delta = end_t - start_t
        print(f"[{obj_fname}]: already has NPY. Skip. Took {time_delta:.2f} s")
        # pbar.update(1)
        return

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
        end_t = time.time()
        time_delta = end_t - start_t
        print(f"[{obj_fname}]: Has problems. Skip. Took {time_delta:.2f} s")
        # pbar.update(1)
        return
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
        end_t = time.time()
        time_delta = end_t - start_t
        print(f"[{obj_fname}]: Saved, took {time_delta:.2f} s")
    else:
        end_t = time.time()
        time_delta = end_t - start_t
        print(f"[{obj_fname}]: Discarded, mesh has {len(edge_feature)} edge features and {len(adj_list)} adjacents, needed is {target_face_num}. Took {time_delta:.2f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--obj-folder", type=str, required=True, help="")
    parser.add_argument("--dataset-out-folder", type=str, required=True, help="")
    parser.add_argument("--npy-out-folder-name", type=str, default="npy", help="")
    parser.add_argument("--num-faces", type=int, required=True, help="Number of desired faces to use.")
    parser.add_argument("--force", action="store_true", help="If set, any NPY file already existing will be overwritten.")
    parser.add_argument("--cpus", default=4, help="Number of processes to use, default is 4.")

    args = parser.parse_args()

    obj_files = glob.glob(os.path.join(args.obj_folder, "*.obj"))

    if args.cpus > 1:
        args_pool=[(obj, args.dataset_out_folder, args.num_faces, args.force, args.npy_out_folder_name) for obj in obj_files]
        pool = mp.Pool(processes=4)
        pool.starmap(to_npy, args_pool, chunksize=None)
        pool.close()
        pool.join()
    else:
        for obj in obj_files:
            to_npy(obj,
                args.dataset_out_folder,
                args.num_faces,
                force_overwrite=args.force,
                npy_folder_name=args.npy_out_folder_name)