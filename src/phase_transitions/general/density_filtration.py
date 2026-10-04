from sklearn.neighbors import NearestNeighbors
import numpy as np

def density_filtration(point_cloud, k=200, p=0.30):

    nbrs = NearestNeighbors(n_neighbors=k+1)
    nbrs.fit(point_cloud)

    distances, _ = nbrs.kneighbors(point_cloud)

    # distancia al vecino número k
    rho = distances[:, k]

    n_keep = int(len(point_cloud)*p)

    keep = np.argsort(rho)[:n_keep]

    return point_cloud[keep]