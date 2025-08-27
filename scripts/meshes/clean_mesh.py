"""

"""

import os
import glob
import argparse

import pymeshlab
import tqdm


parser = argparse.ArgumentParser()

parser.add_argument("--input")
parser.add_argument("--output")

args = parser.parse_args()

in_folder = args.input
out_folder = args.output

if os.path.isdir(in_folder):
    meshes = glob.glob(os.path.join(in_folder, "*.obj"))
else:
    meshes = [in_folder]

pbar = tqdm.tqdm(initial=0, total=len(meshes), unit=" meshes")

for m in meshes:

    fname = os.path.split(m)[1]
    out_path = os.path.join(out_folder, fname)

    if os.path.exists(out_path):
        pbar.update(1)
        continue
    else:
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(m)

        # Delete disconnected geometry, that has less than 25% of the triangles.
        ms.meshing_remove_connected_component_by_diameter(mincomponentdiag = pymeshlab.PercentageValue(25))
        ms.meshing_remove_unreferenced_vertices()

        # Delete non-manifold geometry
        ms.meshing_repair_non_manifold_edges()
        ms.meshing_repair_non_manifold_vertices()

        # Delete degenerate geometry
        try:
            ms.meshing_remove_t_vertices(method = 1)
        except Exception:
            print(f"Couldn't remove T-verts from mesh {fname}")
        ms.meshing_remove_null_faces()

        # Delete repeated/overlapping geometry.
        ms.meshing_remove_duplicate_vertices()
        ms.meshing_merge_close_vertices(threshold = pymeshlab.PercentageValue(0.1))

        fname = os.path.split(m)[1]
        ms.save_current_mesh(out_path)
        os.remove(out_path + ".mtl")
    pbar.update(1)

pbar.close()