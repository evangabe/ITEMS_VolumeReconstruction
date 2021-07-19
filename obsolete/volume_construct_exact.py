# surface volume reconstruction from point cloud
# finds surface points in point cloud

import numpy as np
import shapely
from shapely import geometry, wkt
from shapely.geometry import LinearRing
from shapely.ops import unary_union
from shapely.ops import polygonize
from shapely.ops import nearest_points
from shapely.affinity import translate
from scipy.spatial import Delaunay
from scipy import signal
from scipy import misc
from scipy.signal import fftconvolve
from scipy.signal import argrelextrema
import math
from math import floor, ceil, sqrt
from descartes import PolygonPatch
import pylab as pl
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import pickle
import sys
import os

voxels = np.load('src/vox_files/data_vox_DG_top.npy')  # np.load("data_vox_DG_temporal.npy")
pt_name = "hello_CA1" #sys.argv[1]  # "DG_clean_xyz_norm"
output_filename = pt_name + ".xyz"
alpha = 0.04  # shape construction smoothness factor
k_val = 10  # length of normal
slide = 10 # int(sys.argv[2])  # step size
thickness = 30 # int(sys.argv[3])  # 1/2 window length
figures = 0  # 0 == no figures; 1 == figures


def alpha_shape(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set of points. (using shapely)
    @param points: Iterable container of points.
    @param alpha: alpha value to influence the gooeyness of the border. Smaller
                  numbers don't fall inward as much as larger numbers. Too large,
                  and you lose everything!

    @return unary_union(triangles): shapely polygon
    @return edge_points: boundary points
    """
    if len(points) < 4:
        # When you have a triangle, there is no sense in computing an alpha shape.
        return geometry.MultiPoint(list(points)).convex_hull

    def add_edge(edges, edge_points, coords, i, j):
        """Add a line between the i-th and j-th points, if not in the list already"""
        if (i, j) in edges or (j, i) in edges:
            # already added
            return
        edges.add((i, j))
        edge_points.append(coords[[i, j]])

    tri = Delaunay(points)
    edges = set()
    edge_points = []
    # loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]

        # Lengths of sides of triangle
        a = math.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = math.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = math.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)

        # Semiperimeter of triangle
        s = (a + b + c) / 2.0

        # Area of triangle by Heron's formula
        if s * (s - a) * (s - b) * (s - c) <= 0:
            continue
        area = math.sqrt(s * (s - a) * (s - b) * (s - c))
        if area == 0:
            continue
        circum_r = a * b * c / (4.0 * area)
        if circum_r < 1.0 / alpha:
            add_edge(edges, edge_points, points, ia, ib)
            add_edge(edges, edge_points, points, ib, ic)
            add_edge(edges, edge_points, points, ic, ia)
    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    return unary_union(triangles), edge_points


def plot_polygon(polygon):
    """
    Plot the shapely polygon
    @param polygon: shapely polygon.

    @return fig: plot. Use pl.show() to visualize.
    """
    fig = pl.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim([0, 6000])
    ax.set_ylim([0, 7000])
    patch = PolygonPatch(polygon, fc='#999999', ec='#000000', fill=True, zorder=-1)
    ax.add_patch(patch)
    return fig


def find_slice(dataxy0, alpha):
    """
    Extracting slice of points to generate shape
    @param pts: point cloud.
    @param axis: axis to slice.
    @param val: z value used to extract the slice of points.
    @param alpha: variable to determine smoothness of shape. The closer to zero, the smoother.

    @return shape: shapely polygon.
    """
    # Extracting polygon shape
    # print(dataxy0)
    shape0_, edge_points0 = alpha_shape(dataxy0, alpha=alpha)  # creates polygon(s)

    if shape0_.geom_type == 'MultiPolygon':
        remove = []
        for s in range(len(shape0_)):
            # removes shapes with boundaries that contained less than 10 points
            if len(np.array(shape0_[s].exterior)) < 10:
                remove.append(s)
        counts = 0
        include = list(range(len(shape0_)))
        remove = list(dict.fromkeys(remove))
        for p in range(len(remove)):
            include.remove(remove[p])
        new_shapes = [None] * len(include)
        for t in range(len(include)):
            new_shapes[counts] = shape0_[include[t]]
            counts = counts + 1
        shape = shapely.geometry.MultiPolygon(new_shapes)
    else:
        # checks if there is an interior boundary and preserves it
        ordered_y = np.array(dataxy0[:, 1])
        ordered_y.sort()
        mid_idx = int(len(ordered_y) / 2)
        mid_y = ordered_y[mid_idx]  # calculates median
        idx0 = np.all([dataxy0[:, 1] >= mid_y], axis=0)  # split ring in middle
        dataxy00 = dataxy0[:, 0:2][idx0]
        shape00, edge_points00 = alpha_shape(dataxy00, alpha=alpha)
        idx01 = np.all([dataxy0[:, 1] <= mid_y + 50], axis=0)  # split ring in middle
        dataxy01 = dataxy0[:, 0:2][idx01]
        shape01, edge_points01 = alpha_shape(dataxy01, alpha=alpha)
        shape = shape00.union(shape01)
    return shape


def normal_CCW(before_pt, after_pt, k_value):
    """
    calculating 2D normals for point on exterior boundary
    @param before_pt: point previous in the shape.
    @param after_pt: point after in the shape.
    @param k_value: length of the normal.

    @return norm_x: x component of normal.
    @return norm_y: y component of normal.
    """
    if (before_pt[0] - after_pt[0]) == 0:  # vertical
        norm_y = 0
        if before_pt[1] - after_pt[1] > 0:  # pointed down
            norm_x = 0 - k_value
        else:  # pointed up
            norm_x = k_value
    elif (before_pt[1] - after_pt[1]) == 0:  # horizontal
        norm_x = 0
        if before_pt[0] - after_pt[0] > 0:  # pointed left
            norm_y = k_value
        else:  # pointed right
            norm_y = 0 - k_value
    elif before_pt[0] - after_pt[0] > 0:  # angled to the left
        slope = (before_pt[1] - after_pt[1]) / (before_pt[0] - after_pt[0])
        norm_slope = -(1 / slope)
        angle = math.atan(abs(norm_slope))  # in radians
        x_norm = k_value * math.cos(angle)
        y_norm = k_value * math.sin(angle)
        norm_y = y_norm
        # determining up or down
        if before_pt[1] - after_pt[1] > 0:  # angled toward bottom left
            norm_x = 0 - x_norm
        else:  # angled toward upper left
            norm_x = x_norm
    else:  # angled to the right
        slope = (before_pt[1] - after_pt[1]) / (before_pt[0] - after_pt[0])
        norm_slope = -(1 / slope)
        angle = math.atan(abs(norm_slope))  # in radians
        x_norm = k_value * math.cos(angle)
        y_norm = k_value * math.sin(angle)
        norm_y = 0 - y_norm
        # determining up or down
        if before_pt[1] - after_pt[1] > 0:  # angled toward bottom right
            norm_x = 0 - x_norm
        else:  # angled toward upper right
            norm_x = x_norm
    return norm_x, norm_y


def normal_CW(before_pt, after_pt, k_value):
    """
    calculating 2D normals for point on interior boundary
    @param before_pt: point previous in the shape.
    @param after_pt: point after in the shape.
    @param k_value: length of the normal.

    @return norm_x: x component of normal.
    @return norm_y: y component of normal.
    """
    if (before_pt[0] - after_pt[0]) == 0:  # vertical
        norm_y = 0
        if before_pt[1] - after_pt[1] > 0:  # pointed down
            norm_x = k_value
        else:  # pointed up
            norm_x = 0 - k_value
    elif (before_pt[1] - after_pt[1]) == 0:  # horizontal
        norm_x = 0
        if before_pt[0] - after_pt[0] > 0:  # pointed left
            norm_y = 0 - k_value
        else:  # pointed right
            norm_y = k_value
    elif before_pt[0] - after_pt[0] > 0:  # angled to the left
        slope = (before_pt[1] - after_pt[1]) / (before_pt[0] - after_pt[0])
        norm_slope = -(1 / slope)
        angle = math.atan(abs(norm_slope))  # in radians
        x_norm = k_value * math.cos(angle)
        y_norm = k_value * math.sin(angle)
        norm_y = 0 - y_norm
        # determining up or down
        if before_pt[1] - after_pt[1] > 0:  # angled toward bottom left
            norm_x = x_norm
        else:  # angled toward upper left
            norm_x = 0 - x_norm
    else:  # angled to the right
        slope = (before_pt[1] - after_pt[1]) / (before_pt[0] - after_pt[0])
        norm_slope = -(1 / slope)
        angle = math.atan(abs(norm_slope))  # in radians
        x_norm = k_value * math.cos(angle)
        y_norm = k_value * math.sin(angle)
        norm_y = y_norm
        # determining up or down
        if before_pt[1] - after_pt[1] > 0:  # angled toward bottom right
            norm_x = x_norm
        else:  # angled toward upper right
            norm_x = 0 - x_norm
    return norm_x, norm_y

z_list = np.unique(voxels[:, 2])
x_list = np.unique(voxels[:, 0])
min_x = int(min(x_list))
max_x = int(max(x_list))

# save all values into dictionaries
pts_dict = {}
norm_dict = {}
print('starting z')
# take exterior points for slices taken along z axis
for z in z_list:
    # Extracting points on the slice
    idx = np.all([voxels[:, 2] == z], axis=0)
    pts = voxels[:, 0:2][idx]
    slice_shape = find_slice(pts, alpha)
    if slice_shape.geom_type == 'MultiPolygon':
        for shape in slice_shape:
            if LinearRing(shape.exterior.coords).is_ccw:
                points = list(shape.exterior.coords)
            else:
                points = list(shape.exterior.coords)[::-1]  # reverses order so its CCW
            z_repeat = [z] * len(points)
            points = [sub + (val,) for sub, val in zip(points, z_repeat)]
            # note points that need to be saved
            for ct, k in enumerate(points):
                pts_dict[str(k)] = 1
                if ct == 0:
                    norm = normal_CCW(points[-2], points[ct + 1], k_val)
                elif ct == len(points) - 1:
                    norm = normal_CCW(points[ct - 1], points[1], k_val)
                else:
                    norm = normal_CCW(points[ct - 1], points[ct + 1], k_val)
                norm_dict[str(k)] = (norm[0], norm[1], 0)
            extract_interior = [list(interior.coords) for interior in shape.interiors]
            if extract_interior:
                if not LinearRing(shape.exterior.coords).is_ccw:
                    extract_interior = extract_interior[0]
                else:
                    extract_interior = extract_interior[0][::-1]
                z_repeat = [z] * len(extract_interior)
                in_pts = [sub + (val,) for sub, val in zip(extract_interior, z_repeat)]
                for ct, k in enumerate(in_pts):
                    pts_dict[str(k)] = 1
                    if ct == 0:
                        norm = normal_CW(in_pts[-2], in_pts[ct + 1], k_val)
                    elif ct == len(in_pts) - 1:
                        norm = normal_CW(in_pts[ct - 1], in_pts[1], k_val)
                    else:
                        norm = normal_CW(in_pts[ct - 1], in_pts[ct + 1], k_val)
                    norm_dict[str(k)] = (norm[0], norm[1], 0)
    else:
        if LinearRing(slice_shape.exterior.coords).is_ccw:
            points = list(slice_shape.exterior.coords)
        else:
            points = list(slice_shape.exterior.coords)[::-1]  # reverses order so its CCW
        z_repeat = [z] * len(points)
        points = [sub + (val,) for sub, val in zip(points, z_repeat)]
        for ct, k in enumerate(points):
            pts_dict[str(k)] = 1
            if ct == 0:
                norm = normal_CCW(points[-2], points[ct + 1], k_val)
            elif ct == len(points) - 1:
                norm = normal_CCW(points[ct - 1], points[1], k_val)
            else:
                norm = normal_CCW(points[ct - 1], points[ct + 1], k_val)
            norm_dict[str(k)] = (norm[0], norm[1], 0)
        extract_interior = [list(interior.coords) for interior in slice_shape.interiors]
        if extract_interior:
            if not LinearRing(slice_shape.exterior.coords).is_ccw:
                extract_interior = extract_interior[0]
            else:
                extract_interior = extract_interior[0][::-1]
            z_repeat = [z] * len(extract_interior)
            in_pts = [sub + (val,) for sub, val in zip(extract_interior, z_repeat)]
            for ct, k in enumerate(in_pts):
                pts_dict[str(k)] = 1
                if ct == 0:
                    norm = normal_CW(in_pts[-2], in_pts[ct + 1], k_val)
                elif ct == len(in_pts) - 1:
                    norm = normal_CW(in_pts[ct - 1], in_pts[1], k_val)
                else:
                    norm = normal_CW(in_pts[ct - 1], in_pts[ct + 1], k_val)
                norm_dict[str(k)] = (norm[0], norm[1], 0)
print("total points: ", len(list(pts_dict.keys())))

print('starting x')
# take exterior points on slices along the x axis (using step and 1/2 window length sizes)
x_dict = {}
for k in range(min_x+int(slide*0.5), max_x-int(slide*0.5), slide):
    dis = abs(voxels[:, 0] - k)
    idx = np.all([dis <= thickness], axis=0)
    pts = voxels[:, 0:3][idx]
    for i in pts:
        x_dict[str(i[1:3])] = i[0]
    if len(pts) > 2:
        if len(pts) > 3:
            slice_shape = find_slice(pts[:, 1:3], alpha)
        else:
            slice_shape = geometry.Polygon(pts[:, 1:3])
        if slice_shape.geom_type in ['MultiPolygon', 'GeometryCollection']:
            for shape in slice_shape:
                if LinearRing(shape.exterior.coords).is_ccw:
                    points = list(shape.exterior.coords)
                else:
                    points = list(shape.exterior.coords)[::-1]  # reverses order so its
                for ct, j in enumerate(points):
                    pt_list = np.array([j[0], j[1]])
                    x_value = x_dict[str(pt_list)]
                    pts_dict[str((x_value, pt_list[0], pt_list[1]))] = 1
                    if ct == 0:
                        norm = normal_CCW(points[-2], points[ct + 1], k_val)
                    elif ct == len(points) - 1:
                        norm = normal_CCW(points[ct - 1], points[1], k_val)
                    else:
                        norm = normal_CCW(points[ct - 1], points[ct + 1], k_val)
                    norm_dict[str((x_value, pt_list[0], pt_list[1]))] = (0, norm[0], norm[1])
                extract_interior = [list(interior.coords) for interior in shape.interiors]
                if extract_interior:
                    if not LinearRing(shape.exterior.coords).is_ccw:
                        extract_interior = extract_interior[0]
                    else:
                        extract_interior = extract_interior[0][::-1]
                    for ct, j in enumerate(extract_interior):
                        pt_list = np.array([j[0], j[1]])
                        x_value = x_dict[str(pt_list)]
                        pts_dict[str((x_value, pt_list[0], pt_list[1]))] = 1
                        if ct == 0:
                            norm = normal_CW(extract_interior[-2], extract_interior[ct + 1], k_val)
                        elif ct == len(extract_interior) - 1:
                            norm = normal_CW(extract_interior[ct - 1], extract_interior[1], k_val)
                        else:
                            norm = normal_CW(extract_interior[ct - 1], extract_interior[ct + 1], k_val)
                        norm_dict[str((x_value, pt_list[0], pt_list[1]))] = (0, norm[0], norm[1])
        elif slice_shape.geom_type == 'Polygon':
            if LinearRing(slice_shape.exterior.coords).is_ccw:
                points = list(slice_shape.exterior.coords)
            else:
                points = list(slice_shape.exterior.coords)[::-1]  # reverses order so its CCW
            for ct, j in enumerate(points):
                pt_list = np.array([j[0], j[1]])
                x_value = x_dict[str(pt_list)]
                pts_dict[str((x_value, pt_list[0], pt_list[1]))] = 1
                if ct == 0:
                    norm = normal_CCW(points[-2], points[ct + 1], k_val)
                elif ct == len(points) - 1:
                    norm = normal_CCW(points[ct - 1], points[1], k_val)
                else:
                    norm = normal_CCW(points[ct - 1], points[ct + 1], k_val)
                norm_dict[str((x_value, pt_list[0], pt_list[1]))] = (0, norm[0], norm[1])
            extract_interior = [list(interior.coords) for interior in slice_shape.interiors]
            if extract_interior:
                if not LinearRing(slice_shape.exterior.coords).is_ccw:
                    extract_interior = extract_interior[0]
                else:
                    extract_interior = extract_interior[0][::-1]
                for ct, j in enumerate(extract_interior):
                    pt_list = np.array([j[0], j[1]])
                    x_value = x_dict[str(pt_list)]
                    pts_dict[str((x_value, pt_list[0], pt_list[1]))] = 1
                    if ct == 0:
                        norm = normal_CW(extract_interior[-2], extract_interior[ct + 1], k_val)
                    elif ct == len(extract_interior) - 1:
                        norm = normal_CW(extract_interior[ct - 1], extract_interior[1], k_val)
                    else:
                        norm = normal_CW(extract_interior[ct - 1], extract_interior[ct + 1], k_val)
                    norm_dict[str((x_value, pt_list[0], pt_list[1]))] = (0, norm[0], norm[1])
        x_dict.clear()

# writing to file
points_saved = list(pts_dict.keys())
print(len(points_saved))
print('saving to file')
with open(pt_name + '_dict.pkl', 'wb') as f:
    pickle.dump(norm_dict, f, pickle.HIGHEST_PROTOCOL)
f = open(output_filename, "w+")  # output point cloud with corresponding normals
for k in range(len(points_saved)):
    pt_eval = eval(points_saved[k])
    norm = norm_dict[points_saved[k]]
    f.write("%f %f %f %f %f %f\n" % (pt_eval[0], pt_eval[1], pt_eval[2], norm[0], norm[1], norm[2]))
f.close()


"""
z_list = np.unique(voxels[:, 2])
x_list = np.unique(voxels[:, 0])
min_x = int(min(x_list))
max_x = int(max(x_list))
print(z_list)
# save all values into dictionaries
pts_dict = {}
norm_dict = {}
print('starting z')
# take exterior points for slices taken along z axis
for z in z_list:
    # Extracting points on the slice
    idx = np.all([voxels[:, 2] == z], axis=0)
    pts = voxels[:, 0:2][idx]
    if len(pts) < 2:
        slice_shape = find_slice(pts, alpha)
        if slice_shape.geom_type in ['GeometryCollection','MultiPolygon']:
            for shape in slice_shape:
                if LinearRing(shape.exterior.coords).is_ccw:
                    points = list(shape.exterior.coords)
                else:
                    points = list(shape.exterior.coords)[::-1]  # reverses order so its CCW
                z_repeat = [z] * len(points)
                points = [sub + (val,) for sub, val in zip(points, z_repeat)]
                # note points that need to be saved
                for ct, k in enumerate(points):
                    pts_dict[str(k)] = 1
                    if ct == 0:
                        norm = normal_CCW(points[-2], points[ct + 1], k_val)
                    elif ct == len(points) - 1:
                        norm = normal_CCW(points[ct - 1], points[1], k_val)
                    else:
                        norm = normal_CCW(points[ct - 1], points[ct + 1], k_val)
                    norm_dict[str(k)] = (norm[0], norm[1], 0)
                extract_interior = [list(interior.coords) for interior in shape.interiors]
                if extract_interior:
                    if not LinearRing(shape.exterior.coords).is_ccw:
                        extract_interior = extract_interior[0]
                    else:
                        extract_interior = extract_interior[0][::-1]
                    z_repeat = [z] * len(extract_interior)
                    in_pts = [sub + (val,) for sub, val in zip(extract_interior, z_repeat)]
                    for ct, k in enumerate(in_pts):
                        pts_dict[str(k)] = 1
                        if ct == 0:
                            norm = normal_CW(in_pts[-2], in_pts[ct + 1], k_val)
                        elif ct == len(in_pts) - 1:
                            norm = normal_CW(in_pts[ct - 1], in_pts[1], k_val)
                        else:
                            norm = normal_CW(in_pts[ct - 1], in_pts[ct + 1], k_val)
                        norm_dict[str(k)] = (norm[0], norm[1], 0)
        else:
            if LinearRing(slice_shape.exterior.coords).is_ccw:
                points = list(slice_shape.exterior.coords)
            else:
                points = list(slice_shape.exterior.coords)[::-1]  # reverses order so its CCW
            z_repeat = [z] * len(points)
            points = [sub + (val,) for sub, val in zip(points, z_repeat)]
            for ct, k in enumerate(points):
                pts_dict[str(k)] = 1
                if ct == 0:
                    norm = normal_CCW(points[-2], points[ct + 1], k_val)
                elif ct == len(points) - 1:
                    norm = normal_CCW(points[ct - 1], points[1], k_val)
                else:
                    norm = normal_CCW(points[ct - 1], points[ct + 1], k_val)
                norm_dict[str(k)] = (norm[0], norm[1], 0)
            extract_interior = [list(interior.coords) for interior in slice_shape.interiors]
            if extract_interior:
                if not LinearRing(slice_shape.exterior.coords).is_ccw:
                    extract_interior = extract_interior[0]
                else:
                    extract_interior = extract_interior[0][::-1]
                z_repeat = [z] * len(extract_interior)
                in_pts = [sub + (val,) for sub, val in zip(extract_interior, z_repeat)]
                for ct, k in enumerate(in_pts):
                    pts_dict[str(k)] = 1
                    if ct == 0:
                        norm = normal_CW(in_pts[-2], in_pts[ct + 1], k_val)
                    elif ct == len(in_pts) - 1:
                        norm = normal_CW(in_pts[ct - 1], in_pts[1], k_val)
                    else:
                        norm = normal_CW(in_pts[ct - 1], in_pts[ct + 1], k_val)
                    norm_dict[str(k)] = (norm[0], norm[1], 0)
print("total points: ", len(list(pts_dict.keys())))

print('starting x')
# take exterior points on slices along the x axis (using step and 1/2 window length sizes)
x_dict = {}
for k in range(min_x+int(slide*0.5), max_x-int(slide*0.5), slide):
    dis = abs(voxels[:, 0] - k)
    idx = np.all([dis <= thickness], axis=0)
    pts = voxels[:, 0:3][idx]
    for i in pts:
        x_dict[str(i[1:3])] = i[0]
    if len(pts) > 2:
        if len(pts) > 3:
            slice_shape = find_slice(pts[:, 1:3], alpha)
        else:
            slice_shape = geometry.Polygon(pts[:, 1:3])
        if slice_shape.geom_type in ['GeometryCollection','MultiPolygon']:
            for shape in slice_shape:
                if LinearRing(shape.exterior.coords).is_ccw:
                    points = list(shape.exterior.coords)
                else:
                    points = list(shape.exterior.coords)[::-1]  # reverses order so its
                for ct, j in enumerate(points):
                    pt_list = np.array([j[0], j[1]])
                    x_value = x_dict[str(pt_list)]
                    pts_dict[str((x_value, pt_list[0], pt_list[1]))] = 1
                    if ct == 0:
                        norm = normal_CCW(points[-2], points[ct + 1], k_val)
                    elif ct == len(points) - 1:
                        norm = normal_CCW(points[ct - 1], points[1], k_val)
                    else:
                        norm = normal_CCW(points[ct - 1], points[ct + 1], k_val)
                    norm_dict[str((x_value, pt_list[0], pt_list[1]))] = (0, norm[0], norm[1])
                extract_interior = [list(interior.coords) for interior in shape.interiors]
                if extract_interior:
                    if not LinearRing(shape.exterior.coords).is_ccw:
                        extract_interior = extract_interior[0]
                    else:
                        extract_interior = extract_interior[0][::-1]
                    for ct, j in enumerate(extract_interior):
                        pt_list = np.array([j[0], j[1]])
                        x_value = x_dict[str(pt_list)]
                        pts_dict[str((x_value, pt_list[0], pt_list[1]))] = 1
                        if ct == 0:
                            norm = normal_CW(extract_interior[-2], extract_interior[ct + 1], k_val)
                        elif ct == len(extract_interior) - 1:
                            norm = normal_CW(extract_interior[ct - 1], extract_interior[1], k_val)
                        else:
                            norm = normal_CW(extract_interior[ct - 1], extract_interior[ct + 1], k_val)
                        norm_dict[str((x_value, pt_list[0], pt_list[1]))] = (0, norm[0], norm[1])
        elif slice_shape.geom_type == 'Polygon':
            if LinearRing(slice_shape.exterior.coords).is_ccw:
                points = list(slice_shape.exterior.coords)
            else:
                points = list(slice_shape.exterior.coords)[::-1]  # reverses order so its CCW
            for ct, j in enumerate(points):
                pt_list = np.array([j[0], j[1]])
                x_value = x_dict[str(pt_list)]
                pts_dict[str((x_value, pt_list[0], pt_list[1]))] = 1
                if ct == 0:
                    norm = normal_CCW(points[-2], points[ct + 1], k_val)
                elif ct == len(points) - 1:
                    norm = normal_CCW(points[ct - 1], points[1], k_val)
                else:
                    norm = normal_CCW(points[ct - 1], points[ct + 1], k_val)
                norm_dict[str((x_value, pt_list[0], pt_list[1]))] = (0, norm[0], norm[1])
            extract_interior = [list(interior.coords) for interior in slice_shape.interiors]
            if extract_interior:
                if not LinearRing(slice_shape.exterior.coords).is_ccw:
                    extract_interior = extract_interior[0]
                else:
                    extract_interior = extract_interior[0][::-1]
                for ct, j in enumerate(extract_interior):
                    pt_list = np.array([j[0], j[1]])
                    x_value = x_dict[str(pt_list)]
                    pts_dict[str((x_value, pt_list[0], pt_list[1]))] = 1
                    if ct == 0:
                        norm = normal_CW(extract_interior[-2], extract_interior[ct + 1], k_val)
                    elif ct == len(extract_interior) - 1:
                        norm = normal_CW(extract_interior[ct - 1], extract_interior[1], k_val)
                    else:
                        norm = normal_CW(extract_interior[ct - 1], extract_interior[ct + 1], k_val)
                    norm_dict[str((x_value, pt_list[0], pt_list[1]))] = (0, norm[0], norm[1])
        x_dict.clear()
    else:
        print("1 or 0 points in slice, k =", k)

# writing to file
points_saved = list(pts_dict.keys())
print(len(points_saved))
print('saving to file')
with open(pt_name + '_dict.pkl', 'wb') as f:
    pickle.dump(norm_dict, f, pickle.HIGHEST_PROTOCOL)
f = open(output_filename, "w+")  # output point cloud with corresponding normals
for k in range(len(points_saved)):
    print(points_saved[k])
    pt_eval = eval(points_saved[k])
    norm = norm_dict[points_saved[k]]
    f.write("%f %f %f %f %f %f\n" % (pt_eval[0], pt_eval[1], pt_eval[2], norm[0], norm[1], norm[2]))
f.close()
"""