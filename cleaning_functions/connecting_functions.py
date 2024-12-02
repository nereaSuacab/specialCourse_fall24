import numpy as np
import tifffile as tiff
from skimage.morphology import skeletonize  # Import for skeletonization
from skan import Skeleton, summarize
from joblib import Parallel, delayed
import sys
from scipy.spatial.distance import cdist

def Bresenham3D(x1, y1, z1, x2, y2, z2):
    ListOfPoints = []
    ListOfPoints.append((x1, y1, z1))
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    dz = abs(z2 - z1)
    if (x2 > x1):
        xs = 1
    else:
        xs = -1
    if (y2 > y1):
        ys = 1
    else:
        ys = -1
    if (z2 > z1):
        zs = 1
    else:
        zs = -1
 
    # Driving axis is X-axis"
    if (dx >= dy and dx >= dz):        
        p1 = 2 * dy - dx
        p2 = 2 * dz - dx
        while (x1 != x2):
            x1 += xs
            if (p1 >= 0):
                y1 += ys
                p1 -= 2 * dx
            if (p2 >= 0):
                z1 += zs
                p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz
            ListOfPoints.append((x1, y1, z1))
 
    # Driving axis is Y-axis"
    elif (dy >= dx and dy >= dz):       
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        while (y1 != y2):
            y1 += ys
            if (p1 >= 0):
                x1 += xs
                p1 -= 2 * dy
            if (p2 >= 0):
                z1 += zs
                p2 -= 2 * dy
            p1 += 2 * dx
            p2 += 2 * dz
            ListOfPoints.append((x1, y1, z1))
 
    # Driving axis is Z-axis"
    else:        
        p1 = 2 * dy - dz
        p2 = 2 * dx - dz
        while (z1 != z2):
            z1 += zs
            if (p1 >= 0):
                y1 += ys
                p1 -= 2 * dz
            if (p2 >= 0):
                x1 += xs
                p2 -= 2 * dz
            p1 += 2 * dy
            p2 += 2 * dx
            ListOfPoints.append((x1, y1, z1))
    return ListOfPoints

def calculate_centroid(blob_labels, labels_out):
    centroids = {}
    for blob in np.unique(blob_labels):
        # Encuentra las coordenadas de todos los píxeles de ese blob
        coords = np.array(np.where(labels_out == blob)).T
        # Calcula el centroide como el promedio de las coordenadas
        centroid = np.mean(coords, axis=0)
        centroids[blob] = centroid
    return centroids

def find_closest_blobs(centroids):
    blobs = list(centroids.keys())
    distances = cdist([centroids[blob] for blob in blobs], [centroids[blob] for blob in blobs])
    np.fill_diagonal(distances, np.inf)  # Evitar que los blobs se conecten consigo mismos
    closest_pairs = np.unravel_index(np.argmin(distances), distances.shape)
    return blobs[closest_pairs[0]], blobs[closest_pairs[1]]

def find_closest_blobs(centroids):
    blobs = list(centroids.keys())
    distances = cdist([centroids[blob] for blob in blobs], [centroids[blob] for blob in blobs])
    np.fill_diagonal(distances, np.inf)  # Evitar que los blobs se conecten consigo mismos
    closest_pairs = np.unravel_index(np.argmin(distances), distances.shape)
    return blobs[closest_pairs[0]], blobs[closest_pairs[1]]

# Paso 3: Conectar los blobs más cercanos con una línea recta 3D
def connect_blobs_3d(labels_out, pixel_blob_mapping, skeleton_cleaned):
    import numpy as np
    from scipy.spatial import distance

    # Copia del esqueleto para modificaciones
    skel = np.copy(skeleton_cleaned)

    # Calcular centroides de todos los blobs
    centroids = calculate_centroid(labels_out, labels_out)

    # Encontrar el par de blobs más cercanos
    blob1, blob2 = find_closest_blobs(centroids)
    print(f"Connecting blobs {blob1} and {blob2}")

    # Coordenadas de todos los puntos en cada blob
    coords_blob1 = np.array(np.where(labels_out == blob1)).T
    coords_blob2 = np.array(np.where(labels_out == blob2)).T

    # Calcular las distancias entre cada par de puntos
    distances = distance.cdist(coords_blob1, coords_blob2, 'euclidean')
    min_idx = np.unravel_index(np.argmin(distances), distances.shape)

    # Seleccionar los puntos más cercanos
    p1 = coords_blob1[min_idx[0]]
    p2 = coords_blob2[min_idx[1]]

    print(f"Closest points: {p1} in blob {blob1}, {p2} in blob {blob2}")

    # Calcular la línea de Bresenham entre los puntos (en 3D)
    line_points = Bresenham3D(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2])

    # Conectar los blobs en el mapeo de píxeles
    for point in line_points:
        pixel_blob_mapping[tuple(point)] = blob1  # O usar una mezcla de ambos blobs
        skel[tuple(point)] = 1
    
    return skel
