# surface volume reconstruction from point cloud

import numpy as np
import mahotas
import shapely
from shapely import geometry, wkt
from shapely.geometry import mapping, shape, LinearRing, Point
from shapely.ops import unary_union
from shapely.ops import polygonize
from shapely.ops import nearest_points
from shapely.affinity import translate
from scipy.spatial import Delaunay
from scipy import signal
from scipy import misc
from scipy.signal import fftconvolve
from scipy.signal import argrelextrema
from time import time
import math
from math import floor, ceil, sqrt
import rasterio
from rasterio import Affine, features
from rasterio.plot import show
from descartes import PolygonPatch
import pylab as pl
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

section_name = 'top'  # input("DG Section Name: ")
file_path = "src/vox_files/data_vox_DG_" + section_name.lower() + ".npy"
t_slice = 0
t_start = 40
file_ID = 2
slice_basis_file_ID = "src/intermediate_slice_outputs/slice_basis_logit2.txt"
file_type = "a04t07"
voxels = np.load(file_path)
output_filename = "logit_test_" + str(file_ID) + "_" + section_name.lower() + "_" + "full" + ".xyz"
intermediate_sections = np.loadtxt(slice_basis_file_ID, dtype=np.int64)  # number of intermediate slices added to the original point cloud
alpha = 0.04  # shape construction smoothness factor
k_val = 1  # length of normal
axis = 2  # axis to take slices on
figures = 0   # 0 == no figures; 1 == figures

total_time = time()

def alpha_shape(points, alpha):
    time_alpha = time()
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
        print(circum_r)
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

def find_slice(pts, axis, val, alpha):
    """
    Extracting slice of points to generate shape
    @param pts: point cloud.
    @param axis: axis to slice.
    @param val: z value used to extract the slice of points.
    @param alpha: variable to determine smoothness of shape. The closer to zero, the smoother.

    @return shape: shapely polygon.
    """
    # Extracting points on the slice
    idx = np.all([pts[:, axis] == val], axis=0)  # 3354 connects together
    dataxy0 = pts[:, 0:2][idx]
    # Extracting polygon shape
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
        mid_y = ((np.max(dataxy0[:, 1]) - np.min(dataxy0[:, 1])) / 2) + np.min(dataxy0[:, 1])
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


def z_normal(point, k, shape_current, shape_below, z_below, shape_above, z_above):
    """
    calculating 3D normal
    @param point: point [x,y,z] to calculate normal for.
    @param k: length of normal.
    @param shape_current: shape the point is on.
    @param shape_below: distance transform map of shape directly before the current.
    @param z_below: z axis value for below shape.
    @param shape_above: distance transform map of shape directly after the current.
    @param z_above: z axis value for above shape.

    @return normals: [x, y, z] component of normal.
    """
    """if point[2] == 3384:
        plot_polygon(shape_current)
        plt.scatter(x=point[0], y=point[1], c='r')
        plt.show()"""

    if_interior = 0
    point_xy = Point(point[0], point[1])
    if shape_current.geom_type == 'MultiPolygon':
        # find which shape the point lies on
        shape_current_dis = []
        for s in range(len(shape_current)):
            nearest = nearest_points(shape_current[s], point_xy)[0]
            min_distance = point_xy.distance(nearest)
            shape_current_dis.append(min_distance)
        closest_pt_idx = shape_current_dis.index(min(shape_current_dis))
        shape_current = shape_current[closest_pt_idx]
    if shape_current.geom_type == "Polygon":
        exterior_ring = shape_current.exterior
        if exterior_ring.intersects(point_xy):  # point is on exterior
            shape_pts = np.asarray(shape_current.exterior)
            find_x = np.where(shape_pts[:, 0] == point[0])
            find_y = np.where(shape_pts[:, 1] == point[1])
            ts = np.intersect1d(find_x, find_y)
            if len(ts) == 0:
                print('Fail0')
                return [False, False, False]
            t = ts[0]
            if t == 0:  # first point in array
                first_pt = shape_pts[len(shape_pts) - 2]
                last_pt = shape_pts[t + 1]
            elif t == len(shape_pts) - 1:  # last point in array
                first_pt = shape_pts[t - 1]
                last_pt = shape_pts[1]
            else:
                first_pt = shape_pts[t - 1]
                last_pt = shape_pts[t + 1]
        elif len([np.array(interior.coords) for interior in shape_current.interiors]) > 0:  # point is on interior
            in_pts = [np.array(interior.coords) for interior in shape_current.interiors]
            interior_ring = LinearRing(in_pts[0])
            if interior_ring.intersects(point_xy):
                if_interior = 1
                shape_pts = [np.array(interior.coords) for interior in shape_current.interiors]
                shape_pts = np.asarray(shape_pts[0])
                find_x = np.where(shape_pts[:, 0] == point[0])
                find_y = np.where(shape_pts[:, 1] == point[1])
                ts = np.intersect1d(find_x, find_y)
                if len(ts) == 0:
                    print('intersects interior but not in')
                    plot_polygon(shape_current)
                    plt.scatter(x=point[0], y=point[1], c='r')
                    plt.show()
                    return [False, False, False]
                t = ts[0]
                if t == 0:  # first point in array
                    first_pt = shape_pts[len(shape_pts) - 2]
                    last_pt = shape_pts[t + 1]
                elif t == len(shape_pts) - 1:  # last point in array
                    first_pt = shape_pts[t - 1]
                    last_pt = shape_pts[1]
                else:
                    first_pt = shape_pts[t - 1]
                    last_pt = shape_pts[t + 1]
            else:  # ERROR: point is not on shape
                print('Fail1')
                return [False, False, False]
        else:  # ERROR: point is not on shape
            print('Fail2')
            return [False, False, False]
    else:  # ERROR: point is not on shape
        print('Fail3')
        return [False, False, False]

    # print(point, shape_current, shape_below, z_below, shape_above, z_above)
    if not shape_below:  # this is bottom slice, so no below point
        if shape_above.geom_type == 'MultiPolygon':
            shape_above_dis = []
            for s in range(len(shape_above)):
                nearest = nearest_points(shape_above[s], point_xy)[0]
                min_distance = point_xy.distance(nearest)
                shape_above_dis.append(min_distance)
            closest_pt_idx = shape_above_dis.index(min(shape_above_dis))
            shape_above = shape_above[closest_pt_idx]
        above_pt = nearest_points(shape_above, point_xy)[0]
        plane_points = np.asarray([[first_pt[0], first_pt[1], point[2]], [last_pt[0], last_pt[1], point[2]],
                                   [above_pt.x, above_pt.y, z_above]])
    elif not shape_above:  # this is the top slice, so no above point
        if shape_below.geom_type == 'MultiPolygon':
            shape_below_dis = []
            for s in range(len(shape_below)):
                nearest = nearest_points(shape_below[s], point_xy)[0]
                min_distance = point_xy.distance(nearest)
                shape_below_dis.append(min_distance)
            closest_pt_idx = shape_below_dis.index(min(shape_below_dis))
            shape_below = shape_below[closest_pt_idx]
        below_pt = nearest_points(shape_below, point_xy)[0]
        plane_points = np.asarray([[first_pt[0], first_pt[1], point[2]], [last_pt[0], last_pt[1], point[2]],
                                   [below_pt.x, below_pt.y, z_below]])
    else:  # point is on middle slice, so below and above both exist
        if shape_below.geom_type == 'MultiPolygon':
            shape_below_dis = []
            for s in range(len(shape_below)):
                nearest = nearest_points(shape_below[s], point_xy)[0]
                min_distance = point_xy.distance(nearest)
                shape_below_dis.append(min_distance)
            closest_pt_idx = shape_below_dis.index(min(shape_below_dis))
            shape_below = shape_below[closest_pt_idx]
        if shape_above.geom_type == 'MultiPolygon':
            shape_above_dis = []
            for s in range(len(shape_above)):
                nearest = nearest_points(shape_above[s], point_xy)[0]
                min_distance = point_xy.distance(nearest)
                shape_above_dis.append(min_distance)
            closest_pt_idx = shape_above_dis.index(min(shape_above_dis))
            shape_above = shape_above[closest_pt_idx]
        below_pt = nearest_points(shape_below, point_xy)[0]
        above_pt = nearest_points(shape_above, point_xy)[0]
        plane_points = np.asarray([[first_pt[0], first_pt[1], point[2]], [last_pt[0], last_pt[1], point[2]],
                                   [below_pt.x, below_pt.y, z_below], [above_pt.x, above_pt.y, z_above]])

    # use points surround point of interest to create plane
    # Compute covariance matrix
    p_centroid = np.mean(plane_points, axis=0)
    C = np.zeros((3, 3))
    for ii in range(len(plane_points)):
        p_diff = plane_points[ii] - p_centroid
        C += np.dot(p_diff[:, np.newaxis], p_diff[:, np.newaxis].T)
    C /= len(plane_points)
    # Perform SVD
    u, s, v = np.linalg.svd(C)
    # Normal is the 3rd eigenvector
    normals = v[:, 2]

    # correct orientation
    if if_interior == 0:
        normal_x, normal_y = normal_CCW(first_pt, last_pt, k)
        if not shape_below:  # this is bottom slice, so no below point (so z negative)
            if normals[2] > 0:
                normals[2] = - normals[2]
        elif not shape_above:  # this is the top slice, so no above point
            if normals[2] < 0:
                normals[2] = - normals[2]
        elif point_xy.within(shape_below) and point_xy.within(shape_above):  # local indent
            twod_vex = np.array([normal_x, normal_y, 0])
            normal_rev = -1 * normals
            norm_dot = np.dot(twod_vex, normals)
            norm_angle = np.arccos(norm_dot)
            rev_dot = np.dot(twod_vex, normal_rev)
            rev_angle = np.arccos(rev_dot)
            if rev_angle < norm_angle:
                normals[2] = - normals[2]
        elif not point_xy.within(shape_below) and not point_xy.within(shape_above):  # local peak
            twod_vex = np.array([normal_x, normal_y, 0])
            normal_rev = -1 * normals
            norm_dot = np.dot(twod_vex, normals)
            norm_angle = np.arccos(norm_dot)
            rev_dot = np.dot(twod_vex, normal_rev)
            rev_angle = np.arccos(rev_dot)
            if rev_angle < norm_angle:
                normals[2] = - normals[2]
        elif point_xy.within(shape_below) and not point_xy.within(shape_above):  # / (so z positive)
            if normals[2] < 0:
                normals[2] = - normals[2]
        elif not point_xy.within(shape_below) and point_xy.within(shape_above):  # \ (so z negative)
            if normals[2] > 0:
                normals[2] = - normals[2]
        normals[0] = normal_x
        normals[1] = normal_y
    else:  # for interior points
        normal_x, normal_y = normal_CW(first_pt, last_pt, k)
        if not shape_below:  # this is bottom slice, so no below point (so z negative)
            if normals[2] > 0:
                normals[2] = - normals[2]
        elif not shape_above:  # this is the top slice, so no above point
            if normals[2] < 0:
                normals[2] = - normals[2]
        elif point_xy.within(shape_below) and point_xy.within(shape_above):  # local indent
            twod_vex = np.array([normal_x, normal_y, 0])
            normal_rev = -1 * normals
            norm_dot = np.dot(twod_vex, normals)
            norm_angle = np.arccos(norm_dot)
            rev_dot = np.dot(twod_vex, normal_rev)
            rev_angle = np.arccos(rev_dot)
            if rev_angle < norm_angle:
                normals[2] = - normals[2]
        elif not point_xy.within(shape_below) and not point_xy.within(shape_above):  # local peak
            twod_vex = np.array([normal_x, normal_y, 0])
            normal_rev = -1 * normals
            norm_dot = np.dot(twod_vex, normals)
            norm_angle = np.arccos(norm_dot)
            rev_dot = np.dot(twod_vex, normal_rev)
            rev_angle = np.arccos(rev_dot)
            if rev_angle < norm_angle:
                normals[2] = - normals[2]
        elif point_xy.within(shape_below) and not point_xy.within(shape_above):  # / (so z negative)
            if normals[2] > 0:
                normals[2] = - normals[2]
        elif not point_xy.within(shape_below) and point_xy.within(shape_above):  # \ (so z positive)
            if normals[2] < 0:
                normals[2] = - normals[2]
        normals[0] = normal_x
        normals[1] = normal_y
    mag = math.sqrt((normals[0] ** 2) + (normals[1] ** 2) + normals[2] ** 2)
    normals = (normals / mag) * k

    return normals


def average_polygon(list_shapes, num_shapes):
    """
    Cross correlation and distance transform to generate average polygon between two shapes
    @param list_shapes: two shapely polygons/multipolygons.
        @param num_shapes: array of the weight for each shape.

        @return final_poly_simp: average polygon.
    """
    shapes = shapely.geometry.MultiPolygon(list_shapes)
    max_shape = unary_union(shapes)

    # setting up for the cross-correlation to align shapes
    minx, miny, maxx, maxy = max_shape.bounds
    minx_2, miny_2, maxx_2, maxy_2 = list_shapes[1].bounds
    find_delta = 1
    dx = dy = floor(find_delta)  # grid resolution
    lenx = dx * (ceil(maxx / dx) - floor(minx / dx))
    leny = dy * (ceil(maxy / dy) - floor(miny / dy))
    assert lenx % dx == 0.0
    assert leny % dy == 0.0
    nx = int(lenx / dx)
    ny = int(leny / dy)
    nx_2 = int(dx * (ceil(maxx_2 / dx) - floor(minx_2 / dx)) / dx)
    ny_2 = int(dy * (ceil(maxy_2 / dy) - floor(miny_2 / dy)) / dy)
    gt = Affine(dx, 0.0, dx * floor(minx / dx), 0.0, -dy, dy * ceil(maxy / dy))
    gt_2 = Affine(dx, 0.0, dx * floor(minx_2 / dx), 0.0, -dy, dy * ceil(maxy_2 / dy))
    pa0 = np.zeros((ny, nx), bool)
    r0 = features.rasterize([shapes[0]], (ny, nx), transform=gt)
    pa0[r0 > 0] += True
    pa0 = np.invert(pa0)
    pa1 = np.zeros((ny_2, nx_2), bool)
    r1 = features.rasterize([shapes[1]], (ny_2, nx_2), transform=gt_2)
    pa1[r1 > 0] += True
    pa1 = np.invert(pa1)
    # cross-correlation to align the shapes
    c = signal.fftconvolve(pa0, pa1[::-1], mode='same')
    # FIND:  add gaussian filter to smooth and reduce

    # added: find local maxes of 'c' greater than 50% of global max
    local_maxes = sorted(c[argrelextrema(c,np.greater)], reverse=True)
    local_maxes = set(local_maxes[:int(len(local_maxes)*0.5)])
    # maintain idx where max is found
    local_max = []
    pm = np.zeros(c.shape, 'B')
    t0 = time()
    for loc_max in local_maxes:
        pm[c == loc_max] = 1
        local_max.append((loc_max, np.where(c == loc_max)))
    print('pm time: ', time() - t0, 's')
    #print('PM count: ', np.count_nonzero(pm == 1))

    max_intersection_area = -1
    point = []
    threshold = 0.85
    pts_loc = []
    for sh, val in features.shapes(pm, transform=gt):
        if val == 1:
            point.append(shape(sh))
            pts = np.array(shape(sh).exterior.coords)
            pts = np.mean(pts[0:3], axis=0)
            pts_loc.append(pts)
    if not any(point):
        raise ValueError("could not find any optimal shift")

    print(' Pts_loc: ', len(pts_loc), ' Loc_max: ', len(local_max))
    local_max = local_max[:len(pts_loc)]

    ct = 0
    for (loc_max, i) in local_max:
        pt_loc = np.asarray(pts_loc[ct])
        x_shift = pt_loc[0]
        y_shift = pt_loc[1]
        x_shift = x_shift - ((minx_2 + maxx_2) / 2)
        y_shift = y_shift - ((miny_2 + maxy_2) / 2)

        this_shape = translate(list_shapes[1], xoff=x_shift, yoff=y_shift, zoff=0.0)

        ct += 1
        if this_shape.intersection(list_shapes[0]).area > max_intersection_area:
            max_intersection_area = this_shape.intersection(list_shapes[0]).area
            shift_to = [x_shift, y_shift]
            best_loc_max = loc_max
        else:
            this_shape = translate(this_shape, xoff=-x_shift, yoff=-y_shift, zoff=0.0)
        # added: threshold to reduce computation time
        if (max_intersection_area / list_shapes[0].area) > threshold:
            break

    # print(best_loc_max, 'best max @ index: ', max_idx, '/', len(local_max))

    """
    for i, (sh, val) in enumerate(features.shapes(pm, transform=gt)):
        if val == 1:
            point.append(shape(sh))
            pts = np.array(shape(sh).exterior.coords)
            xy_mean = np.mean(pts[0:3],axis=0)
            x_shift = xy_mean[0] - ((minx_2 + maxx_2) / 2)
            y_shift = xy_mean[1] - ((miny_2 + maxy_2) / 2)

            this_shape = translate(list_shapes[1], xoff=x_shift, yoff=y_shift, zoff=0.0)

            if this_shape.intersection(list_shapes[0]).area > max_intersection_area:
                max_intersection_area = this_shape.intersection(list_shapes[0]).area
                shift_to = [x_shift, y_shift]
                best_loc_max = c[np.where(features.shapes(pm, transform=gt) == sh)]
                print(best_loc_max, 'best max')
                max_intersection_idx = i
            else:
                this_shape = translate(this_shape, xoff=-x_shift, yoff=-y_shift, zoff=0.0)
            # added: threshold to reduce computation time
            if (max_intersection_area / list_shapes[0].area) > threshold:
                break
    """

    list_shapes[1] = translate(list_shapes[1], xoff=shift_to[0], yoff=shift_to[1], zoff=0.0)

    # calculate average polygon
    shapes = shapely.geometry.MultiPolygon(list_shapes)
    max_shape = unary_union(shapes)
    minx, miny, maxx, maxy = max_shape.bounds
    find_delta = 1
    dx = dy = floor(find_delta)
    lenx = dx * (ceil(maxx / dx) - floor(minx / dx))
    leny = dy * (ceil(maxy / dy) - floor(miny / dy))
    assert lenx % dx == 0.0
    assert leny % dy == 0.0
    nx = int(lenx / dx)
    ny = int(leny / dy)
    gt = Affine(dx, 0.0, dx * floor(minx / dx), 0.0, -dy, dy * ceil(maxy / dy))
    pa0 = np.zeros((ny, nx), bool)
    r0 = features.rasterize([shapes[0]], (ny, nx), transform=gt)
    pa0[r0 > 0] += True
    pa0 = np.invert(pa0)
    pa1 = np.zeros((ny, nx), bool)
    r1 = features.rasterize([shapes[1]], (ny, nx), transform=gt)
    pa1[r1 > 0] += True
    pa1 = np.invert(pa1)
    dmap0 = mahotas.distance(pa0)
    dmap0 = dmap0 / np.max(dmap0)
    dmap1 = mahotas.distance(pa1)
    dmap1 = dmap1 / np.max(dmap1)
    # weighting shapes by area
    if num_shapes[0] > num_shapes[1]:  # shapes[0].area > shapes[1].area:
        num_shapes_0 = 10 * num_shapes[0]  # * round(shapes[0].area / shapes[1].area)
        num_shapes_1 = num_shapes[1]
    elif num_shapes[0] < num_shapes[1]:
        num_shapes_0 = num_shapes[0]
        num_shapes_1 = 10 * num_shapes[0]  # * round(shapes[1].area / shapes[0].area)
    else:
        num_shapes_0 = num_shapes[0]
        num_shapes_1 = num_shapes[1]
    dmap_avg = ((num_shapes_0 * dmap0) + (num_shapes_1 * dmap1)) / (num_shapes[0] + num_shapes[1])

    # extract polygon from distance transform
    thresh = 0.0007 * np.amax(dmap_avg)
    pm = np.zeros(dmap_avg.shape, 'B')
    pm[dmap_avg < thresh] = 1
    poly = []
    poly_in = []
    poly_shapes = []
    interior_pts = 0
    for sh, val in features.shapes(pm, transform=gt):
        if val == 1:
            poly_shapes.append(shape(sh))
            pts = np.array(shape(sh).exterior.coords)
            poly.append(pts)
            extract_interior = [np.array(interior.coords) for interior in shape(sh).interiors]
            if extract_interior:
                poly_in.append(extract_interior)
                interior_pts = 1
    if not any(poly_shapes):
        thresh = 0.001 * np.amax(dmap_avg)
        pm = np.zeros(dmap_avg.shape, 'B')
        pm[dmap_avg < thresh] = 1
        poly = []
        poly_in = []
        poly_shapes = []
        interior_pts = 0
        for sh, val in features.shapes(pm, transform=gt):
            if val == 1:
                poly_shapes.append(shape(sh))
                pts = np.array(shape(sh).exterior.coords)
                poly.append(pts)
                extract_interior = [np.array(interior.coords) for interior in shape(sh).interiors]
                if extract_interior:
                    poly_in.append(extract_interior)
                    interior_pts = 1
        if not any(poly_shapes):
            return []
            # raise ValueError("could not find any shapes")
    if interior_pts == 1:
        poly = np.asarray(poly[0])
        poly_in = np.asarray(poly_in[0])
        final_poly = shapely.geometry.Polygon(poly, poly_in)
        final_poly_simp = final_poly.simplify(4)
    else:
        poly = np.asarray(poly)
        if len(poly) > 1:
            rings = []
            for q in range(len(poly)):
                rings.append(geometry.Polygon(poly[q]))
            final_poly = shapely.geometry.MultiPolygon(rings)
            final_poly_simp = final_poly.simplify(4)
        else:
            final_poly = shapely.geometry.Polygon(poly[0])
            final_poly_simp = final_poly.simplify(4)

    # plt.figure()
    # c_plt = plt.imshow(np.log10(dmap_avg))
    # plt.colorbar(c_plt)
    # plt.show()

    percent_shift = num_shapes[1] / sum(num_shapes)
    final_poly_simp = translate(final_poly_simp, xoff=-shift_to[0] * percent_shift, yoff=-shift_to[1] * percent_shift, zoff=0.0)

    """plot_polygon(final_poly_simp)
    first = np.array(list_shapes[0].exterior)
    second = np.array(list_shapes[1].exterior)
    plt.plot(first[:, 0], first[:, 1], "r")
    plt.plot(second[:, 0], second[:, 1], "b")
    plt.show()"""

    return final_poly_simp


def average_transition_polygon(shape0, shape1, list_info):
    """
    Cross correlation and distance transform to generate average polygon between two shapes where one is a Multipolygon
    @param shape0: polygon.
    @param shape1: polygon.
    @param list_info: array of the weight for each shape.

    @return final_poly_simp: average polygon.
    """
    shapes_for_intermediate = []
    if shape0.geom_type == 'MultiPolygon':
        shapes_for_intermediate.append(shape1)
        for x in range(len(shape0)):
            shapes_for_intermediate.append(shape0[x])
    else:
        shapes_for_intermediate.append(shape0)
        for x in range(len(shape1)):
            shapes_for_intermediate.append(shape1[x])
    shapes = shapely.geometry.MultiPolygon(shapes_for_intermediate)
    max_shape = unary_union(shapes)

    # setting up for the cross-correlation to align shapes
    minx, miny, maxx, maxy = max_shape.bounds
    minx_2, miny_2, maxx_2, maxy_2 = shape1.bounds
    find_delta = 1
    dx = dy = floor(find_delta)  # grid resolution
    lenx = dx * (ceil(maxx / dx) - floor(minx / dx))
    leny = dy * (ceil(maxy / dy) - floor(miny / dy))
    assert lenx % dx == 0.0
    assert leny % dy == 0.0
    nx = int(lenx / dx)
    ny = int(leny / dy)
    nx_2 = int(dx * (ceil(maxx_2 / dx) - floor(minx_2 / dx)) / dx)
    ny_2 = int(dy * (ceil(maxy_2 / dy) - floor(miny_2 / dy)) / dy)
    gt = Affine(dx, 0.0, dx * floor(minx / dx), 0.0, -dy, dy * ceil(maxy / dy))
    gt_2 = Affine(dx, 0.0, dx * floor(minx_2 / dx), 0.0, -dy, dy * ceil(maxy_2 / dy))

    if shape0.geom_type == 'MultiPolygon':
        pa0_mag = np.zeros((ny, nx))
        pa0 = np.zeros((ny, nx), bool)
        for x in range(len(shape0)):
            r0 = features.rasterize([shape0[x]], (ny, nx), transform=gt)
            pa0_mag[r0 > 0] += 1
        pa0[pa0_mag > 0] += True
        pa0 = np.invert(pa0)
        pa1 = np.zeros((ny_2, nx_2), bool)
        r1 = features.rasterize([shape1], (ny_2, nx_2), transform=gt_2)
        pa1[r1 > 0] += True
        pa1 = np.invert(pa1)
    else:  # shape1.geom_type == 'MultiPolygon'
        pa0 = np.zeros((ny, nx), bool)
        r0 = features.rasterize([shape0], (ny, nx), transform=gt)
        pa0[r0 > 0] += True
        pa0 = np.invert(pa0)
        pa1_mag = np.zeros((ny_2, nx_2))
        pa1 = np.zeros((ny_2, nx_2), bool)
        for x in range(len(shape1)):
            r1 = features.rasterize([shape1[x]], (ny_2, nx_2), transform=gt_2)
            pa1_mag[r1 > 0] += 1
        pa1[pa1_mag > 0] += True
        pa1 = np.invert(pa1)

    # cross-correlation to align the shapes
    c = signal.fftconvolve(pa0, pa1[::-1], mode='same')
    local_max = sorted(c[argrelextrema(c,np.greater)], reverse=True)
    best_shape = []
    max_intersection_area = -1
    max_intersection_idx = -1

    threshold = 0.85
    t = time()
    for loc_max in local_max:
        pm = np.zeros(c.shape, 'B')
        pm[c == loc_max] = 1
        pts_loc = []
        point = []

        for sh, val in features.shapes(pm, transform=gt):
            if val == 1:
                point.append(shape(sh))
                pts = np.array(shape(sh).exterior.coords)
                pts_loc.append(pts)

        if not any(point):
            raise ValueError("could not find any optimal shift")
        pts_loc = np.asarray(pts_loc[0])
        x_shift = np.mean(pts_loc[:, 0])
        y_shift = np.mean(pts_loc[:, 1])
        x_shift = x_shift - ((minx_2 + maxx_2) / 2)
        y_shift = y_shift - ((miny_2 + maxy_2) / 2)

        this_shape = translate(shape1, xoff=x_shift, yoff=y_shift, zoff=0.0)

        # possibly add a threshold percent area to reduce computation time
        if this_shape.intersection(shape0).area > max_intersection_area:
            max_intersection_area = this_shape.intersection(shape0).area
            best_shape = this_shape
            shift_to = [x_shift, y_shift]
            max_intersection_idx = local_max.index(loc_max)
        if (max_intersection_area / shape0.area) > threshold:
            break
    shape1 = translate(best_shape, xoff=shift_to[0], yoff=shift_to[1], zoff=0.0)


    # calculate average polygon
    list_shapes = []
    which = 0
    if shape0.geom_type == 'MultiPolygon':
        list_shapes.append(shape1)
        for x in range(len(shape0)):
            list_shapes.append(shape0[x])
    else:
        list_shapes.append(shape0)
        for x in range(len(shape1)):
            list_shapes.append(shape1[x])
        which = 1
    shapes = shapely.geometry.MultiPolygon(list_shapes)
    max_shape = unary_union(shapes)
    minx, miny, maxx, maxy = max_shape.bounds
    find_delta = 1
    dx = dy = floor(find_delta)
    lenx = dx * (ceil(maxx / dx) - floor(minx / dx))
    leny = dy * (ceil(maxy / dy) - floor(miny / dy))
    assert lenx % dx == 0.0
    assert leny % dy == 0.0
    nx = int(lenx / dx)
    ny = int(leny / dy)
    gt = Affine(dx, 0.0, dx * floor(minx / dx), 0.0, -dy, dy * ceil(maxy / dy))
    if shape0.geom_type == 'MultiPolygon':
        pa0_mag = np.zeros((ny, nx))
        pa0 = np.zeros((ny, nx), bool)
        for x in range(len(shape0)):
            r0 = features.rasterize([shape0[x]], (ny, nx), transform=gt)
            pa0_mag[r0 > 0] += 1
        pa0[pa0_mag > 0] += True
        pa0 = np.invert(pa0)
        pa1 = np.zeros((ny, nx), bool)
        r1 = features.rasterize([shape1], (ny, nx), transform=gt)
        pa1[r1 > 0] += True
        pa1 = np.invert(pa1)
    else:  # shape1.geom_type == 'MultiPolygon'
        pa0 = np.zeros((ny, nx), bool)
        r0 = features.rasterize([shape0], (ny, nx), transform=gt)
        pa0[r0 > 0] += True
        pa0 = np.invert(pa0)
        pa1_mag = np.zeros((ny, nx))
        pa1 = np.zeros((ny, nx), bool)
        for x in range(len(shape1)):
            r1 = features.rasterize([shape1[x]], (ny, nx), transform=gt)
            pa1_mag[r1 > 0] += 1
        pa1[pa1_mag > 0] += True
        pa1 = np.invert(pa1)
    dmap0 = mahotas.distance(pa0)
    dmap0 = dmap0 / np.max(dmap0)
    dmap1 = mahotas.distance(pa1)
    dmap1 = dmap1 / np.max(dmap1)
    # weighting shapes by size
    if list_info[0] > list_info[1]:
        num_shapes_0 = 10 * list_info[0] * round(shape0.area / shape1.area)
        num_shapes_1 = list_info[1]
    elif list_info[0] < list_info[1]:
        num_shapes_0 = list_info[0]
        num_shapes_1 = 10 * list_info[0] * round(shape1.area / shape0.area)
    else:
        num_shapes_0 = list_info[0]
        num_shapes_1 = list_info[1]
    dmap_avg = ((num_shapes_0 * dmap0) + (num_shapes_1 * dmap1)) / (list_info[0] + list_info[1])
    # extract polygon from distance transform
    thresh = 0.0007 * np.amax(dmap_avg)
    pm = np.zeros(dmap_avg.shape, 'B')
    pm[dmap_avg < thresh] = 1
    poly = []
    poly_in = []
    poly_shapes = []
    interior_pts = 0
    for sh, val in features.shapes(pm, transform=gt):
        if val == 1:
            poly_shapes.append(shape(sh))
            pts = np.array(shape(sh).exterior.coords)
            poly.append(pts)
            extract_interior = [np.array(interior.coords) for interior in shape(sh).interiors]
            if extract_interior:
                poly_in.append(extract_interior)
                interior_pts = 1
            else:
                poly_in.append([[0, 0, 0]])
    if not any(poly_shapes):
        raise ValueError("could not find any shapes")
    poly = np.asarray(poly)
    if interior_pts == 1:
        if len(poly) > 1:
            list_poly = []
            poly_in = np.asarray(poly_in)
            for w in range(len(poly)):
                if len(poly_in[w]) == 1:
                    a_poly = shapely.geometry.Polygon(poly[w])
                else:
                    a_poly = shapely.geometry.Polygon(poly[w], poly_in[w])
                list_poly.append(a_poly)
            final_poly = shapely.geometry.MultiPolygon(list_poly)
            final_poly_simp = final_poly.simplify(0.1)
        else:
            poly_in = np.asarray(poly_in)
            final_poly = shapely.geometry.Polygon(poly[0], poly_in[0])
            final_poly_simp = final_poly.simplify(4)
    else:
        if len(poly) > 1:
            rings = []
            for q in range(len(poly)):
                rings.append(geometry.Polygon(poly[q]))
            final_poly = shapely.geometry.MultiPolygon(rings)
            final_poly_simp = final_poly.simplify(4)
        else:
            final_poly = shapely.geometry.Polygon(poly[0])
            final_poly_simp = final_poly.simplify(4)
    # shift into alignment
    percent_shift = list_info[1] / sum(list_info)
    final_poly_simp = translate(final_poly_simp, xoff=-x_shift * percent_shift, yoff=-y_shift * percent_shift, zoff=0.0)

    return final_poly_simp


def simple_slice(list_polygons, list_info, slices_all, z, figs):
    """
    Create intermediate slice with two polygons.
    @param list_polygons: list of two polygons.
    @param list_info: array of the weight for each shape.
    @param slices_all: [z, shape]
    @param z: z value of slice.
    @param figs: bool for figure.

    @return poly_pts: polygon boundary points.
    @return poly_in_pts: polygon interior points
    """
    avg_poly = average_polygon(list_polygons, list_info)
    #plot_polygon(avg_poly)

    if isinstance(avg_poly, list):
        return [], [], slices_all
    if avg_poly.geom_type == 'MultiPolygon':
        poly_pts_all = []
        poly_in_pts_all = []
        all_polygons = []
        all_interiors = []
        for q in range(len(avg_poly)):
            if avg_poly[q].exterior.is_ccw:
                poly_pts = np.asarray(
                    avg_poly[q].exterior)  # boundary points in CCW direction (first and last point are repeats)
            else:
                poly_pts = np.flip(np.asarray(avg_poly[q].exterior), 0)
            # checks the distance between points. if distance is greater than 50, a midpoint is added
            w = 0
            while w < len(poly_pts) - 1:
                pt1 = poly_pts[w, :]
                pt2 = poly_pts[w + 1, :]
                distance = math.sqrt(((pt1[0] - pt2[0]) ** 2) + ((pt1[1] - pt2[1]) ** 2))
                if distance > 50:
                    x_dis = (pt1[0] - pt2[0]) / 2
                    y_dis = (pt1[1] - pt2[1]) / 2
                    mid_pt = [pt1[0] - x_dis, pt1[1] - y_dis]
                    poly_pts = np.insert(poly_pts, w + 1, np.array(mid_pt), 0)
                else:
                    w = w + 1
            for s in range(len(poly_pts)):
                poly_pts_all.append([poly_pts[s][0], poly_pts[s][1]])
            all_polygons.append(geometry.Polygon(poly_pts))

            extract_interior = [np.array(interior.coords) for interior in avg_poly[q].interiors]
            if extract_interior:
                poly_in_pts_ = np.asarray(extract_interior)
                poly_in_pts = poly_in_pts_[0]
                # checks the distance between points. if distance is greater than 50, a midpoint is added
                w = 0
                while w < len(poly_in_pts) - 1:
                    pt1 = poly_in_pts[w, :]
                    pt2 = poly_in_pts[w + 1, :]
                    distance = math.sqrt(((pt1[0] - pt2[0]) ** 2) + ((pt1[1] - pt2[1]) ** 2))
                    if distance > 50:
                        x_dis = (pt1[0] - pt2[0]) / 2
                        y_dis = (pt1[1] - pt2[1]) / 2
                        mid_pt = [pt1[0] - x_dis, pt1[1] - y_dis]
                        poly_in_pts = np.insert(poly_in_pts, w + 1, np.array(mid_pt), 0)
                    else:
                        w = w + 1
                all_interiors.append(poly_in_pts)
                poly_in_pts_all.append(poly_in_pts)
            # visualize polygons
            if figs > 0:
                plot_polygon(avg_poly)
                pl.show()
        if len(all_interiors) > 0:
            avg_poly_new = geometry.MultiPolygon(all_polygons, all_interiors)
            if not any(slices_all[:, 0] == z):  # if z value not in slices_all
                slices_all = np.append(slices_all, [[z, avg_poly_new]], axis=0)
            else:
                z_find = np.where(slices_all[:, 0] == z)
                z_index = z_find[0][0]
                poly_at_slice = slices_all[z_index, 1]
                multi_new = unary_union([poly_at_slice, avg_poly_new])
                slices_all[z_index, 1] = multi_new
        else:
            avg_poly_new = geometry.MultiPolygon(all_polygons)
            if not any(slices_all[:, 0] == z):  # if z value not in slices_all
                slices_all = np.append(slices_all, [[z, avg_poly_new]], axis=0)
            else:
                z_find = np.where(slices_all[:, 0] == z)
                z_index = z_find[0][0]
                poly_at_slice = slices_all[z_index, 1]
                multi_new = unary_union([poly_at_slice, avg_poly_new])
                slices_all[z_index, 1] = multi_new
        poly_pts = np.array(poly_pts_all)
        poly_in_pts = np.array(poly_in_pts_all)
        if len(poly_in_pts) > 0:
            poly_in_pts = poly_in_pts[0]
    else:
        if avg_poly.exterior.is_ccw:
            poly_pts = np.asarray(avg_poly.exterior)  # boundary points in CCW direction (first and last point are repeats)
        else:
            poly_pts = np.flip(np.asarray(avg_poly.exterior), 0)
        # checks the distance between points. if distance is greater than 50, a midpoint is added
        w = 0
        while w < len(poly_pts) - 1:
            pt1 = poly_pts[w, :]
            pt2 = poly_pts[w + 1, :]
            distance = math.sqrt(((pt1[0] - pt2[0]) ** 2) + ((pt1[1] - pt2[1]) ** 2))
            if distance > 100:
                x_dis = (pt1[0] - pt2[0]) / 2
                y_dis = (pt1[1] - pt2[1]) / 2
                mid_pt = [pt1[0] - x_dis, pt1[1] - y_dis]
                poly_pts = np.insert(poly_pts, w + 1, np.array(mid_pt), 0)
            else:
                w = w + 1

        extract_interior = [np.array(interior.coords) for interior in avg_poly.interiors]
        if extract_interior:
            poly_in_pts_ = np.asarray(extract_interior)
            poly_in_pts = poly_in_pts_[0]
            check_ring = LinearRing(poly_in_pts)
            reduced = check_ring.simplify(6)
            poly_in_pts = np.asarray(reduced.coords)
            # checks to see if exterior is correctly going CCW
            if not check_ring.is_ccw:
                poly_in_pts = np.flip(poly_in_pts, 0)
            # checks the distance between points. if distance is greater than 50, a midpoint is added
            w = 0
            while w < len(poly_in_pts) - 1:
                pt1 = poly_in_pts[w, :]
                pt2 = poly_in_pts[w + 1, :]
                distance = math.sqrt(((pt1[0] - pt2[0]) ** 2) + ((pt1[1] - pt2[1]) ** 2))
                if distance > 200:
                    x_dis = (pt1[0] - pt2[0]) / 2
                    y_dis = (pt1[1] - pt2[1]) / 2
                    mid_pt = [pt1[0] - x_dis, pt1[1] - y_dis]
                    poly_in_pts = np.insert(poly_in_pts, w + 1, np.array(mid_pt), 0)
                else:
                    w = w + 1
            avg_poly_new = geometry.Polygon(poly_pts, holes=[poly_in_pts[::-1]])
            if not any(slices_all[:, 0] == z):  # if z value not in slices_all
                slices_all = np.append(slices_all, [[z, avg_poly_new]], axis=0)
            else:
                z_find = np.where(slices_all[:, 0] == z)
                z_index = z_find[0][0]
                poly_at_slice = slices_all[z_index, 1]
                multi_new = unary_union([poly_at_slice, avg_poly_new])
                slices_all[z_index, 1] = multi_new
        else:
            poly_in_pts = []
            avg_poly_new = geometry.Polygon(poly_pts)
            if not any(slices_all[:, 0] == z):  # if z value not in slices_all
                slices_all = np.append(slices_all, [[z, avg_poly_new]], axis=0)
            else:
                z_find = np.where(slices_all[:, 0] == z)
                z_index = z_find[0][0]
                poly_at_slice = slices_all[z_index, 1]
                multi_new = unary_union([poly_at_slice, avg_poly_new])
                slices_all[z_index, 1] = multi_new

    # visualize polygons
    if figs > 0:
        plot_polygon(avg_poly_new)
        pl.show()

    """plt.plot(poly_pts[:,0], poly_pts[:,1])
    plt.show()"""
    return poly_pts, poly_in_pts, slices_all


def transition_slice(shape0, shape1, list_ratio, slices_all, z, figs):
    """
    Create intermediate slice with a polygon and multipolygon.
    @param shape0: polygon/multipolygon.
    @param shape1: polygon/multipolygon.
    @param list_ratio: array of the weight for each shape.
    @param slices_all: [z, shape]
    @param z: z value of slice.
    @param figs: bool for figure.

    @return poly_pts: polygon boundary points.
    @return poly_in_pts: polygon interior points
    """
    avg_poly = average_transition_polygon(shape0, shape1, list_ratio)
    if avg_poly.geom_type == 'MultiPolygon':
        poly_pts_all = []
        poly_in_pts_all = []
        all_polygons = []
        all_interiors = []
        for q in range(len(avg_poly)):
            if avg_poly[q].exterior.is_ccw:
                poly_pts = np.asarray(
                    avg_poly[q].exterior)  # boundary points in CCW direction (first and last point are repeats)
            else:
                poly_pts = np.flip(np.asarray(avg_poly[q].exterior), 0)
            # checks the distance between points. if distance is greater than 50, a midpoint is added
            w = 0
            while w < len(poly_pts) - 1:
                pt1 = poly_pts[w, :]
                pt2 = poly_pts[w + 1, :]
                distance = math.sqrt(((pt1[0] - pt2[0]) ** 2) + ((pt1[1] - pt2[1]) ** 2))
                if distance > 50:
                    x_dis = (pt1[0] - pt2[0]) / 2
                    y_dis = (pt1[1] - pt2[1]) / 2
                    mid_pt = [pt1[0] - x_dis, pt1[1] - y_dis]
                    poly_pts = np.insert(poly_pts, w + 1, np.array(mid_pt), 0)
                else:
                    w = w + 1
            for s in range(len(poly_pts)):
                poly_pts_all.append([poly_pts[s][0], poly_pts[s][1]])
            all_polygons.append(geometry.Polygon(poly_pts))

            extract_interior = [np.array(interior.coords) for interior in avg_poly[q].interiors]
            if extract_interior:
                poly_in_pts_ = np.asarray(extract_interior)
                poly_in_pts = poly_in_pts_[0]
                # checks the distance between points. if distance is greater than 50, a midpoint is added
                w = 0
                while w < len(poly_in_pts) - 1:
                    pt1 = poly_in_pts[w, :]
                    pt2 = poly_in_pts[w + 1, :]
                    distance = math.sqrt(((pt1[0] - pt2[0]) ** 2) + ((pt1[1] - pt2[1]) ** 2))
                    if distance > 50:
                        x_dis = (pt1[0] - pt2[0]) / 2
                        y_dis = (pt1[1] - pt2[1]) / 2
                        mid_pt = [pt1[0] - x_dis, pt1[1] - y_dis]
                        poly_in_pts = np.insert(poly_in_pts, w + 1, np.array(mid_pt), 0)
                    else:
                        w = w + 1
                all_interiors.append(poly_in_pts)
                poly_in_pts_all.append(poly_in_pts)
            # visualize polygons
            if figs > 0:
                plot_polygon(avg_poly)
                pl.show()
        if len(all_interiors) > 0:
            avg_poly_new = geometry.MultiPolygon(all_polygons, all_interiors)
            if not any(slices_all[:, 0] == z):  # if z value not in slices_all
                slices_all = np.append(slices_all, [[z, avg_poly_new]], axis=0)
            else:
                z_find = np.where(slices_all[:, 0] == z)
                z_index = z_find[0][0]
                poly_at_slice = slices_all[z_index, 1]
                multi_new = unary_union([poly_at_slice, avg_poly_new])
                slices_all[z_index, 1] = multi_new
        else:
            avg_poly_new = geometry.MultiPolygon(all_polygons)
            if not any(slices_all[:, 0] == z):  # if z value not in slices_all
                slices_all = np.append(slices_all, [[z, avg_poly_new]], axis=0)
            else:
                z_find = np.where(slices_all[:, 0] == z)
                z_index = z_find[0][0]
                poly_at_slice = slices_all[z_index, 1]
                multi_new = unary_union([poly_at_slice, avg_poly_new])
                slices_all[z_index, 1] = multi_new
        poly_pts_all = np.array(poly_pts_all)
        poly_in_pts_all = np.array(poly_in_pts_all)
        if len(poly_in_pts_all) > 0:
            poly_in_pts_all = poly_in_pts_all[0]
    else:  # only one polygon
        if avg_poly.exterior.is_ccw:
            poly_pts = np.asarray(avg_poly.exterior)  # boundary points in CCW direction (first and last point are repeats)
        else:
            poly_pts = np.flip(np.asarray(avg_poly.exterior), 0)
        # checks the distance between points. if distance is greater than 50, a midpoint is added
        w = 0
        while w < len(poly_pts) - 1:
            pt1 = poly_pts[w, :]
            pt2 = poly_pts[w + 1, :]
            distance = math.sqrt(((pt1[0] - pt2[0]) ** 2) + ((pt1[1] - pt2[1]) ** 2))
            if distance > 50:
                x_dis = (pt1[0] - pt2[0]) / 2
                y_dis = (pt1[1] - pt2[1]) / 2
                mid_pt = [pt1[0] - x_dis, pt1[1] - y_dis]
                poly_pts = np.insert(poly_pts, w + 1, np.array(mid_pt), 0)
            else:
                w = w + 1
        poly_pts_all = poly_pts

        poly_in_pts_all = []
        extract_interior = [np.array(interior.coords) for interior in avg_poly.interiors]
        if extract_interior:
            poly_in_pts_ = np.asarray(extract_interior)
            poly_in_pts = poly_in_pts_[0]
            # checks the distance between points. if distance is greater than 50, a midpoint is added
            w = 0
            while w < len(poly_in_pts) - 1:
                pt1 = poly_in_pts[w, :]
                pt2 = poly_in_pts[w + 1, :]
                distance = math.sqrt(((pt1[0] - pt2[0]) ** 2) + ((pt1[1] - pt2[1]) ** 2))
                if distance > 50:
                    x_dis = (pt1[0] - pt2[0]) / 2
                    y_dis = (pt1[1] - pt2[1]) / 2
                    mid_pt = [pt1[0] - x_dis, pt1[1] - y_dis]
                    poly_in_pts = np.insert(poly_in_pts, w + 1, np.array(mid_pt), 0)
                else:
                    w = w + 1
            poly_in_pts_all = poly_in_pts
            # visualize polygons
            if figs > 0:
                plot_polygon(avg_poly)
                pl.show()
            avg_poly_new = geometry.Polygon(poly_pts_all, [poly_in_pts])
            if not any(slices_all[:, 0] == z):  # if z value not in slices_all
                slices_all = np.append(slices_all, [[z, avg_poly_new]], axis=0)
            else:
                z_find = np.where(slices_all[:, 0] == z)
                z_index = z_find[0][0]
                poly_at_slice = slices_all[z_index, 1]
                multi_new = unary_union([poly_at_slice, avg_poly_new])
                slices_all[z_index, 1] = multi_new
        else:
            avg_poly_new = geometry.Polygon(poly_pts_all)
            if not any(slices_all[:, 0] == z):  # if z value not in slices_all
                slices_all = np.append(slices_all, [[z, avg_poly_new]], axis=0)
            else:
                z_find = np.where(slices_all[:, 0] == z)
                z_index = z_find[0][0]
                poly_at_slice = slices_all[z_index, 1]
                multi_new = unary_union([poly_at_slice, avg_poly_new])
                slices_all[z_index, 1] = multi_new
    return poly_pts_all, poly_in_pts_all, slices_all


intermediate_pts = []
all_outer = []
all_inner = []
slices = np.unique(voxels[:, 2])
number_of_slices = len(slices)
poly = find_slice(voxels, axis, slices[0], alpha)
slices_all = np.array([[slices[0], poly]])

# generates all the z values we want the intermediate slices at
intermediate = []
for k in np.arange(t_start, number_of_slices - 1):
    difference = slices[k + 1] - slices[k]
    for s in range(1, intermediate_sections[k] + 1):
        intermediate.append(slices[k] + (difference * s) / (intermediate_sections[k] + 1))
# intermediate contains the z value of the intermediate slice

# generate the two polygons adjacent to the intermediate slices and find average polygon
# change this number to change number of slices to run through
#for k in range(number_of_slices - 1):

for k in np.arange(t_start, number_of_slices - 1):
#for k in range(number_of_slices - 1):
    print(k, '-', k + 1)
    for t in range(intermediate_sections[k]):
        print('intermediate: ', t)
        # existing bottom slice
        shape0 = find_slice(voxels, axis, slices[k], alpha)
        if shape0.geom_type == 'MultiPolygon':
            if len(shape0) == 1:
                shape0 = shape0[0]
                # add points of the bottom slice to the point cloud
                poly_pts = np.asarray(shape0.exterior)
                for b in range(len(poly_pts)):
                    all_outer.append([poly_pts[b, 0], poly_pts[b, 1], slices[k]])
                extract_interior = [np.array(interior.coords) for interior in shape0.interiors]
                if extract_interior:
                    poly_in_pts = np.asarray(extract_interior)[0]
                    if len(np.shape(poly_in_pts)) == 1:
                        for s in range(len(poly_in_pts)):
                            poly_in_pts_it = poly_in_pts[s]
                            for b in range(len(poly_in_pts_it)):
                                all_inner.append([poly_in_pts_it[b, 0], poly_in_pts[b, 1], slices[k]])
                    else:
                        for b in range(len(poly_in_pts)):
                            all_inner.append([poly_in_pts[b, 0], poly_in_pts[b, 1], slices[k]])
            else:  # add points from the 'MultiPolygon
                for q in range(len(shape0)):
                    # add points of the bottom slice to the point cloud
                    poly_pts = np.asarray(shape0[q].exterior)
                    for b in range(len(poly_pts)):
                        all_outer.append([poly_pts[b, 0], poly_pts[b, 1], slices[k]])
                    extract_interior = [np.array(interior.coords) for interior in shape0[q].interiors]
                    if extract_interior:
                        poly_in_pts = np.asarray(extract_interior)[0]
                        if len(np.shape(poly_in_pts)) == 1:
                            for s in range(len(poly_in_pts)):
                                poly_in_pts_it = poly_in_pts[s]
                                for b in range(len(poly_in_pts_it)):
                                    all_inner.append([poly_in_pts_it[b, 0], poly_in_pts_it[b, 1], slices[k]])
                        else:
                            for b in range(len(poly_in_pts)):
                                all_inner.append([poly_in_pts[b, 0], poly_in_pts[b, 1], slices[k]])
        else:  # shape0.geom_type == 'Polygon'
            # add points of the bottom slice to the point cloud
            poly_pts = np.asarray(shape0.exterior)
            for b in range(len(poly_pts)):
                all_outer.append([poly_pts[b, 0], poly_pts[b, 1], slices[k]])
            extract_interior = [np.array(interior.coords) for interior in shape0.interiors]
            if extract_interior:
                poly_in_pts = np.asarray(extract_interior)[0]
                if len(np.shape(poly_in_pts)) == 1:
                    for s in range(len(poly_in_pts)):
                        poly_in_pts_it = poly_in_pts[s]
                        for b in range(len(poly_in_pts_it)):
                            all_inner.append([poly_in_pts_it[b, 0], poly_in_pts_it[b, 1], slices[k]])
                else:
                    for b in range(len(poly_in_pts)):
                        all_inner.append([poly_in_pts[b, 0], poly_in_pts[b, 1], slices[k]])
        if not any(slices_all[:, 0] == slices[k]):
            slices_all = np.append(slices_all, [[slices[k], shape0]], axis=0)

        # existing top slice
        shape1 = find_slice(voxels, axis, slices[k + 1], alpha)
        if shape1.geom_type == 'MultiPolygon':
            if len(shape1) == 1:
                shape1 = shape1[0]
        if figures > 1:
            plot_polygon(shape0)
            plot_polygon(shape1)
            pl.show()

        if shape0.geom_type == 'MultiPolygon' or shape1.geom_type == 'MultiPolygon':
            # both slices contain multiple polygons
            if shape0.geom_type == 'MultiPolygon' and shape1.geom_type == 'MultiPolygon':
                if len(shape0) == len(shape1):
                    time_avgply = time()
                    for x in range(len(shape0)):
                        # find slices for intermediate
                        num_of_top = t + 1
                        num_of_bot = (intermediate_sections[k] + 1) - num_of_top
                        # print(num_of_bot, num_of_top)
                        shapes_for_intermediate = [shape0[x], shape1[x]]
                        # print(len(shapes_for_intermediate))

                        # finds average polygon, extracts points

                        outer, inner, slices_all = simple_slice(shapes_for_intermediate, [num_of_bot, num_of_top], slices_all, intermediate[sum(intermediate_sections[t_start:k]) + t], figures) # sum(intermediate_sections[:k]) + t
                        if len(outer) == 0:
                            print("Yes")
                            continue
                        else:
                            for b in range(len(outer)):
                                all_outer.append([outer[b, 0], outer[b, 1], intermediate[sum(intermediate_sections[t_start:k]) + t]])
                            if len(np.shape(inner)) == 1:
                                for s in range(len(inner)):
                                    poly_in_pts_it = inner[s]
                                    for b in range(len(poly_in_pts_it)):
                                        all_inner.append([poly_in_pts_it[b, 0], poly_in_pts_it[b, 1], intermediate[sum(intermediate_sections[t_start:k]) + t]])
                            else:
                                for b in range(len(inner)):
                                    all_inner.append([inner[b, 0], inner[b, 1], intermediate[sum(intermediate_sections[t_start:k]) + t]])
                    print("Avg Poly time: ",time() - time_avgply)
                else:
                    # slices contain unequal number of polygons
                    print("currently bypassing: unequal MultiPolygon")
            else:
                # slices contain unequal number of polygons
                num_of_top = t + 1
                num_of_bot = (intermediate_sections[k] + 1) - num_of_top
                # finds average polygon, extracts points
                outer, inner, slices_all = transition_slice(shape0, shape1, [num_of_bot, num_of_top], slices_all, intermediate[sum(intermediate_sections[t_start:k]) + t], figures)
                if len(outer) == 0:
                    continue
                else:
                    for b in range(len(outer)):
                        all_outer.append([outer[b, 0], outer[b, 1], intermediate[sum(intermediate_sections[t_start:k]) + t]])
                    if len(np.shape(inner)) == 1:
                        for s in range(len(inner)):
                            poly_in_pts_it = inner[s]
                            for b in range(len(poly_in_pts_it)):
                                all_inner.append([poly_in_pts_it[b, 0], poly_in_pts_it[b, 1], intermediate[sum(intermediate_sections[t_start:k]) + t]])
                    else:
                        for b in range(len(inner)):
                            all_inner.append([inner[b, 0], inner[b, 1], intermediate[sum(intermediate_sections[t_start:k]) + t]])

        # both slices contain only one polygon
        else:
            # find slices for intermediate
            num_of_top = t + 1
            num_of_bot = (intermediate_sections[k] + 1) - num_of_top
            shapes_for_intermediate = [shape0, shape1]

            # finds average polygon, extracts points
            outer, inner, slices_all = simple_slice(shapes_for_intermediate, [num_of_bot, num_of_top], slices_all, intermediate[sum(intermediate_sections[t_start:k]) + t], figures)
            if len(outer) == 0:
                continue
            else:
                for b in range(len(outer)):
                    all_outer.append([outer[b, 0], outer[b, 1], intermediate[sum(intermediate_sections[t_start:k]) + t]])
                if len(np.shape(inner)) == 1:
                    for s in range(len(inner)):
                        poly_in_pts_it = inner[s]
                        for b in range(len(poly_in_pts_it)):
                            all_inner.append([poly_in_pts_it[b, 0], poly_in_pts_it[b, 1], intermediate[sum(intermediate_sections[t_start:k]) + t]])
                else:
                    for b in range(len(inner)):
                        all_inner.append([inner[b, 0], inner[b, 1], intermediate[sum(intermediate_sections[t_start:k]) + t]])
# add bottom slice
shape0 = find_slice(voxels, axis, slices[-1], alpha)
slices_all = np.append(slices_all, [[slices[-1], shape0]], axis=0)
# add points of the bottom slice to the point cloud
poly_pts = np.asarray(shape0.exterior)

for b in range(len(poly_pts)):
    all_outer.append([poly_pts[b, 0], poly_pts[b, 1], slices[-1]])
extract_interior = [np.array(interior.coords) for interior in shape0.interiors]
if extract_interior:
    poly_in_pts = np.asarray(extract_interior)
    if len(np.shape(poly_in_pts)) == 1:
        for s in range(len(poly_in_pts)):
            poly_in_pts_it = poly_in_pts[s]
            for b in range(len(poly_in_pts_it)):
                all_inner.append([poly_in_pts_it[b, 0], poly_in_pts_it[b, 1], slices[-1]])
    else:
        for b in range(len(poly_in_pts)):
            all_inner.append([poly_in_pts[b, 0], poly_in_pts[b, 1], slices[-1]])

slices_all = slices_all[slices_all[:,0].argsort()]

if len(all_inner) > 1:
    all_pts = np.concatenate((all_outer, all_inner), axis=0).tolist()
else:
    all_pts = all_outer

shape_info = open('collect_all_polygons.txt', "w+")
for k in range(len(slices_all)):
    """plot_polygon(slices_all[k][1])
    pl.show()"""
    shape_info.write(str(slices_all[k][1]) + "\n")
shape_info.close()
x = len(all_pts)
print('normal calculation')
k = 0
pt_norm = []
while x > 0:
    z_find = np.where(slices_all[:, 0] == all_pts[k][2])
    z_index = z_find[0][0]
    if z_index == 0:
        # bottom slice
        slices_current = slices_all[z_index][1]
        slices_below = []
        slices_above = slices_all[z_index + 1][1]
        z_norm = z_normal(all_pts[k], k_val, slices_current, slices_below, 0, slices_above, slices_all[z_index + 1, 0])
        if z_norm[0] == False:
            all_pts.remove(all_pts[k])
        else:
            pt_norm.append(z_norm)
            k = k + 1
    elif z_index == len(slices_all) - 1:
        # top slice
        slices_current = slices_all[z_index][1]
        slices_below = slices_all[z_index - 1][1]
        slices_above = []
        z_norm = z_normal(all_pts[k], k_val, slices_current, slices_below, slices_all[z_index - 1, 0], slices_above, 0)
        if z_norm[0] == False:
            all_pts.remove(all_pts[k])
        else:
            pt_norm.append(z_norm)
            k = k + 1
    else:  # everything else
        slices_current = slices_all[z_index][1]
        slices_below = slices_all[z_index - 1][1]
        slices_above = slices_all[z_index + 1][1]
        z_norm = z_normal(all_pts[k], k_val, slices_current, slices_below, slices_all[z_index - 1, 0], slices_above, slices_all[z_index + 1, 0])
        if z_norm[0] == False:
            all_pts.remove(all_pts[k])
        else:
            pt_norm.append(z_norm)
            k = k + 1
    x = x - 1

#fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
#ax1.pcolormesh(dmap_avg)
#plt.plot()

f = open(output_filename, "w+")  # output point cloud with corresponding normals
for k in range(len(all_pts)):
    f.write("%f %f %f %f %f %f\n" % (all_pts[k][0], all_pts[k][1], all_pts[k][2], pt_norm[k][0], pt_norm[k][1], pt_norm[k][2]))
f.close()
total_elapsed = time() - total_time
print("Total Time Elapsed: ", total_elapsed)