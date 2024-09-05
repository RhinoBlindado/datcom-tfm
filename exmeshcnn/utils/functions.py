import torch

def get_adj_nm(adjs_): # integrate all the adjacent face lists to make a unified structure including each face and its three adjacent faces.
    re = torch.cat([adjs_[:,1].unsqueeze(1), adjs_[:,0].unsqueeze(1)],dim=1)
    ag = torch.cat([adjs_, re],dim=0)
    tt = ag[ag[:,0].sort()[1]]
    t1 = tt[:,0].unique(return_inverse =True)[1]
    t2 = (tt[:,0].unique(return_counts =True)[1] == 3).nonzero(as_tuple=False)[:,0].unsqueeze(1)
    # uid = ((t1 == t2) == True).nonzero(as_tuple=False)[:,1]

    # Batched version
    # Initialize a list to store the result
    result = []

    # Loop over each element in t2
    for i, value in enumerate(t2.squeeze()):  # Remove the extra dimension from t2 and get the index and value
        indices = (t1 == value).nonzero(as_tuple=True)[0]  # Get indices where t1 matches the value
        for index in indices:
            result.append([i, index.item()])

    uid_batched = torch.Tensor(result).to(torch.int64)[:,1]
    # print(torch.equal(uid, uid_batched))

    aa= tt[uid_batched].reshape(-1,6)
    add=torch.unique(aa, dim=1)

    return add
    
def get_edges(face_, verts_): # looking for an edge path
    tf = face_[0]
    tt = face_[1:]
    edge = []
    for i in range(3):
        t2 = tf[i].unsqueeze(0).repeat(3,1)
        iid = (tt== t2).nonzero(as_tuple = False)
        pp = tt[iid[:,0]].flatten().unique(return_counts=True)[0]
        pi = ((pp != tf[0]) & (pp != tf[1]) & (pp != tf[2])).nonzero(as_tuple=False)[:,0]
        pp[pi]
        e1 = tf[i].unsqueeze(0).repeat(2,1)
        e2 = pp[pi].reshape(-1,1)
        e1 = verts_[e1]
        e2 = verts_[e2]
        edge.append((e1 - e2).squeeze(1))
    return torch.stack(edge,dim=0)

