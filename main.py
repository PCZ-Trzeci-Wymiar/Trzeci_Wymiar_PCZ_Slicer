import meshio
import numpy as np
import matplotlib.pyplot as plt

# ===========================
# 1. Wczytaj plik STL
# ===========================
mesh = meshio.read("stojak.stl")  # zamień na swoją ścieżkę
vertices = mesh.points
triangles = mesh.cells_dict["triangle"]

# ===========================
# 2. Parametry warstw
# ===========================
layer_height = 0.2  # mm
z_min = np.min(vertices[:, 2])
z_max = np.max(vertices[:, 2])
layers = np.arange(z_min + layer_height/2, z_max, layer_height)

# ===========================
# 3. Funkcja przecięcia trójkąta z płaszczyzną Z
# ===========================
def triangle_plane_intersection(v0, v1, v2, z):
    points = []
    for edge in [(v0, v1), (v1, v2), (v2, v0)]:
        p1, p2 = edge
        if (p1[2] - z) * (p2[2] - z) < 0:
            t = (z - p1[2]) / (p2[2] - p1[2])
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            points.append([x, y])
        elif p1[2] == z:
            points.append([p1[0], p1[1]])
        elif p2[2] == z:
            points.append([p2[0], p2[1]])
    if len(points) == 2:
        return points
    return None

# ===========================
# 4. Tworzenie segmentów dla warstw
# ===========================
def slice_layer(z):
    segments = []
    for tri in triangles:
        v0, v1, v2 = vertices[tri]
        seg = triangle_plane_intersection(v0, v1, v2, z)
        if seg:
            segments.append(seg)
    return segments

# ===========================
# 5. Łączenie segmentów w kontury (prosty algorytm)
# ===========================
def build_contours(segments):
    contours = []
    segments = [np.array(s) for s in segments]
    while segments:
        current = list(segments.pop(0))
        changed = True
        while changed:
            changed = False
            for i, s in enumerate(segments):
                if np.allclose(current[-1], s[0]):
                    current.append(s[1])
                    segments.pop(i)
                    changed = True
                    break
                elif np.allclose(current[-1], s[1]):
                    current.append(s[0])
                    segments.pop(i)
                    changed = True
                    break
        contours.append(np.array(current))
    return contours

# ===========================
# 6. Wizualizacja warstw
# ===========================
for i, z in enumerate(layers):
    segments = slice_layer(z)
    contours = build_contours(segments)
    
    plt.figure(figsize=(5,5))
    for contour in contours:
        plt.plot(contour[:,0], contour[:,1], '-o')
    plt.title(f"Layer {i} at Z={z:.2f}")
    plt.axis('equal')
    plt.show()
