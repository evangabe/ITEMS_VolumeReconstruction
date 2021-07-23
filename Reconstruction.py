# surface volume reconstruction from point cloud
# finds surface points in point cloud
from cProfile import run

import numpy as np
import scipy.spatial
from numpy import array, average, all, flip, where, insert, load, empty, median
import shapely
from shapely import speedups
from shapely import geometry, wkt
from shapely.geometry import LinearRing, MultiLineString, Polygon, MultiPolygon
from shapely.ops import unary_union, polygonize, nearest_points, cascaded_union
from shapely.affinity import translate
import scipy
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
from time import time
import sys
import os
from open3d import geometry, utility
from open3d.visualization import draw_geometries

alpha = 0.04
k_val = 10

class Reconstruction:
    # STATIC ATTRIBUTES
    fileID = "1"
    out_file_ext = ".xyz"
    zx_progress = 0
    out_file_tag = ""

    # Input filename as "... .npy"
    def __init__(self, in_file, out_file="", figures = 0):
        self.voxels = load(in_file)  # load("data_vox_DG_temporal.npy")
        if self.voxels is None or len(self.voxels) is 0:
            print("File does not contain any points.")
        self.slide = 10  # int(sys.argv[2])  # int(sys.argv[2])  # step size
        self.thickness = 30  # int(sys.argv[3])  # int(sys.argv[3])  # 1/2 window length
        self.figures = figures

        if out_file is "":
            self.out_file = "volrecon_s" + str(self.slide) + "_t" + str(self.thickness) + ".xyz"
        elif ".xyz" in out_file:
            self.out_file = out_file
        elif "." in out_file:
            self.out_file = out_file.split('.')[0] + ".xyz"
        else:
            self.out_file = out_file + ".xyz"


    # PUBLIC INSTANCE METHODS

    def set_slide(self, slide):
        self.slide = slide

    def set_thickness(self, thickness):
        self.thickness = thickness

    def set_figures(self, figures):
        self.figures = figures

    def set_voxels(self, filename):
        self.voxels = load(filename)

    def plot_pc(self):
        pcd = geometry.PointCloud()
        pcd.points = utility.Vector3dVector(self.voxels[:, :3])
        draw_geometries([pcd])

    def plot_polygon(self, polygon):
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

    # RECONSTRUCTION HELPER METHODS

    def alpha_shape(self, points, alpha):
        """
        Compute the alpha shape (concave hull) of a set of points. (using shapely)
        @param points: Iterable container of points.
        @param alpha: alpha value to influence the gooeyness of the border. Smaller
                      numbers don't fall inward as much as larger numbers. Too large,
                      and you lose everything!

        @return unary_union(triangles): shapely polygon
        @return edge_points: boundary points
        """

        def add_edge(edges, edge_points, coords, i, j):
            """Add a line between the i-th and j-th points, if not in the list already"""
            if (i, j) in edges or (j, i) in edges:
                # already added
                return
            # yield (i, j), coords[[i, j]]
            edges.add((i, j))
            edge_points.append(coords[[i, j]])

        #print(points)
        try:
            tri = Delaunay(points)
        except scipy.spatial.qhull.QhullError:
            plt.figure()
            plt.scatter(points[0, :], points[1, :])
            plt.title("QHULL Error Reached at Iteration #" + str(Reconstruction.zx_progress))
            plt.xlabel("1D Axis")
            plt.ylabel("2D Axis")
            plt.savefig(Reconstruction.out_file_tag + "_QHerr_slc_" + str(Reconstruction.zx_progress) + ".png")
            plt.show()
            return None, None

        edges = set()
        edge_points = []

        #print(dir(tri))
        # plotted = [[points[ia], points[ib], points[ic]] for ia, ib, ic in tri.vertices]

        # self.plot_polygon(plotted)
        # plt.show()
        # loop over triangles:
        # ia, ib, ic = indices of corner points of the triangle

        for ia, ib, ic in tri.vertices:
            pa = points[ia]
            pb = points[ib]
            pc = points[ic]

            # Lengths of sides of triangle
            a = sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
            b = sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
            c = sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)

            # Semiperimeter of triangle
            s = (a + b + c) / 2.0

            # Area of triangle by Heron's formula
            if s * (s - a) * (s - b) * (s - c) <= 0:
                continue
            area = sqrt(s * (s - a) * (s - b) * (s - c))
            if area == 0:
                continue
            circum_r = a * b * c / (4.0 * area)
            if circum_r < 1.0 / alpha:
                add_edge(edges, edge_points, points, ia, ib)
                add_edge(edges, edge_points, points, ib, ic)
                add_edge(edges, edge_points, points, ic, ia)
        m = MultiLineString(edge_points)
        triangles = unary_union(list(polygonize(m)))
        return triangles, edge_points

    def find_slice(self, dataxy0, alpha):
        """
        Extracting slice of points to generate shape
        @param pts: point cloud.
        @param axis: axis to slice.
        @param val: z value used to extract the slice of points.
        @param alpha: variable to determine smoothness of shape. The closer to zero, the smoother.

        @return shape: shapely polygon.
        """
        # Extracting polygon shape
        shape0_, edge_points0 = self.alpha_shape(dataxy0, alpha=alpha)  # creates polygon(s)
        if shape0_ is None or edge_points0 is None:
            return None

        type = shape0_.geom_type
        if type in ['MultiPolygon', 'GeometryCollection']:
            # ext_lens = array([len(array(shape.exterior)) for shape in shape0_])
            # shape = MultiPolygon(shape0_[where(ext_lens > 10)[0][0]:where(ext_lens > 10)[0][-1]])
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
            shape = MultiPolygon(new_shapes)
        else:
            # checks if there is an interior boundary and preserves it
            mid_y = median(dataxy0[:,1])  # calculates median
            idx0 = all([dataxy0[:,1] >= mid_y], axis=0)  # split ring in middle
            dataxy00 = dataxy0[:, 0:2][idx0]
            _, _, r, _, _ = scipy.stats.linregress(dataxy00[:, 0], dataxy00[:, 1])
            if r ** 2 == 0:
                return None
            shape00, edge_points00 = self.alpha_shape(dataxy00, alpha=alpha)
            if shape00 is None or edge_points00 is None:
                return None
            idx01 = all([dataxy0[:,1] <= mid_y + 50], axis=0)  # split ring in middle
            dataxy01 = dataxy0[:, 0:2][idx01]
            _, _, r, _, _ = scipy.stats.linregress(dataxy01[:, 0], dataxy01[:, 1])
            if r ** 2 == 0:
                return None
            shape01, edge_points01 = self.alpha_shape(dataxy01, alpha=alpha)
            if shape01 is None or edge_points01 is None:
                return None
            shape = shape00.union(shape01)

        return shape

    def normal_CCW(self, before_pt, after_pt, k_value):
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

    def normal_CW(self, before_pt, after_pt, k_value):
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

    # PRIMARY RECONSTRUCTION METHOD

    def reconstruct(self, slide=10, thickness=30):

        if slide is not self.slide or thickness is not self.thickness:
            self.set_slide(slide)
            self.set_thickness(thickness)

        z_list = np.unique(self.voxels[:, 2])
        x_list = np.unique(self.voxels[:, 0])
        min_x = int(min(x_list))
        max_x = int(max(x_list))

        t1 = time()
        # save all values into dictionaries
        pts_dict = {}
        norm_dict = {}
        print('starting z')
        # take exterior points for slices taken along z axis
        t_z = time()

        yield len(z_list) + len(x_list)

        for z in z_list:
            Reconstruction.zx_progress += 1
            yield Reconstruction.zx_progress
            # Extracting points on the slice
            idx = all([self.voxels[:, 2] == z], axis=0)
            pts = self.voxels[:, 0:2][idx]
            slice_shape = self.find_slice(pts, alpha)
            if not slice_shape:
                continue
            if Reconstruction.zx_progress % 10 == 0:
                print("Linear slice:", Reconstruction.zx_progress)
            if slice_shape.geom_type in ['MultiPolygon', 'GeometryCollection']:
                for shape in slice_shape:
                    if LinearRing(shape.exterior.coords).is_ccw:
                        points = array(shape.exterior.coords)
                    else:
                        points = array(shape.exterior.coords)[::-1]  # reverses order so its CCW
                    # note points that need to be saved
                    pt_list = [str((k[0], k[1], z)) for k in points]
                    for ct, k in enumerate(points):
                        pts_dict[pt_list[ct]] = 1
                        if ct == 0:
                            norm = self.normal_CCW(points[-2], points[ct + 1], k_val)
                        elif ct == len(points) - 1:
                            norm = self.normal_CCW(points[ct - 1], points[1], k_val)
                        else:
                            norm = self.normal_CCW(points[ct - 1], points[ct + 1], k_val)
                        norm_dict[pt_list[ct]] = (norm[0], norm[1], 0)
                    extract_interior = [array(interior.coords) for interior in shape.interiors]
                    if extract_interior:
                        if not LinearRing(shape.exterior.coords).is_ccw:
                            extract_interior = extract_interior[0]
                        else:
                            extract_interior = extract_interior[0][::-1]
                        pt_list = [str((k[0], k[1], z)) for k in extract_interior]
                        for ct, k in enumerate(extract_interior):
                            pts_dict[pt_list[ct]] = 1
                            if ct == 0:
                                norm = self.normal_CW(extract_interior[-2], extract_interior[ct + 1], k_val)
                            elif ct == len(extract_interior) - 1:
                                norm = self.normal_CW(extract_interior[ct - 1], extract_interior[1], k_val)
                            else:
                                norm = self.normal_CW(extract_interior[ct - 1], extract_interior[ct + 1], k_val)
                            norm_dict[pt_list[ct]] = (norm[0], norm[1], 0)
            else:
                if LinearRing(slice_shape.exterior.coords).is_ccw:
                    points = array(slice_shape.exterior.coords)
                else:
                    points = array(slice_shape.exterior.coords)[::-1]  # reverses order so its CCW
                pt_list = [str((k[0], k[1], z)) for k in points]
                for ct, k in enumerate(points):
                    pts_dict[pt_list[ct]] = 1
                    if ct == 0:
                        norm = self.normal_CCW(points[-2], points[ct + 1], k_val)
                    elif ct == len(points) - 1:
                        norm = self.normal_CCW(points[ct - 1], points[1], k_val)
                    else:
                        norm = self.normal_CCW(points[ct - 1], points[ct + 1], k_val)
                    norm_dict[pt_list[ct]] = (norm[0], norm[1], 0)
                extract_interior = [array(interior.coords) for interior in slice_shape.interiors]
                if extract_interior:
                    if not LinearRing(slice_shape.exterior.coords).is_ccw:
                        extract_interior = extract_interior[0]
                    else:
                        extract_interior = extract_interior[0][::-1]
                    pt_list = [str((k[0], k[1], z)) for k in extract_interior]
                    for ct, k in enumerate(extract_interior):
                        pts_dict[pt_list[ct]] = 1
                        if ct == 0:
                            norm = self.normal_CW(extract_interior[-2], extract_interior[1], k_val)
                        elif ct == len(extract_interior) - 1:
                            norm = self.normal_CW(extract_interior[ct - 1], extract_interior[1], k_val)
                        else:
                            norm = self.normal_CW(extract_interior[ct - 1], extract_interior[ct + 1], k_val)
                        norm_dict[pt_list[ct]] = (norm[0], norm[1], 0)
        #print("total points: ", len(array(pts_dict.keys())))
        print("Time z: ", time() - t_z)

        print('starting x')
        # take exterior points on slices along the x axis (using step and 1/2 window length sizes)
        x_dict = {}
        t_x = time()
        """
        Loop through windows in x
        """
        for k in range(min_x + int(slide * 0.5), max_x - int(slide * 0.5), slide):
            Reconstruction.zx_progress += 1
            yield Reconstruction.zx_progress

            # 1. Find distance from window's upper threshold for each x-value in voxels
            dis = abs(self.voxels[:, 0] - k)
            # 2. Take the indices in voxels whose distance is less than the window thickness
            idx = all([dis <= thickness], axis=0)
            # 3. Create a new array "pts" with x,y,z values corresponding to the above indices in voxels
            pts = self.voxels[:, 0:3][idx]
            if len(np.unique(pts[:,2])) < 2:  # or len(np.unique(pts[:,1])) < 3
                continue
            # 4. Map the y,z coordinates to the corresponding x value in "x_dict"
            for i in pts:
                x_dict[str(i[1:3])] = i[0]
            if len(pts) > 2:
                # 5a. After checking that there are enough coordinates for taking the slice, take the slice
                if len(pts) > 3:
                    slice_shape = self.find_slice(pts[:, 1:3], alpha)
                    if not slice_shape:
                        continue
                    if Reconstruction.zx_progress % 10 == 0:
                        print("Linear slice:", Reconstruction.zx_progress)
                # 5b. Otherwise, find a polygon or continue to next iteration
                else:
                    slice_shape = Polygon(pts[:, 1:3])
                type = slice_shape.geom_type
                if type in ['MultiPolygon', 'GeometryCollection']:
                    # 6a. For MultiPolygon or GeometryCollection slices:
                    #         *For each shape in the slice*
                    #     i.      Create a new array "points" containing the new coordinates of the slice
                    #     ii.     For each point, map original x with new y,z coordinates to logical True
                    #     iii.    Find the normal in the x,y,z coordinates and save the normals in a dictionary
                    #     iv.     Repeat i-iii. for interior coordinates
                    for shape in slice_shape:
                        if LinearRing(shape.exterior.coords).is_ccw:
                            points = array(shape.exterior.coords)
                        else:
                            points = array(shape.exterior.coords)[::-1]  # reverses order so its
                        pt_list = [str((x_dict[str(j[0:2])], j[0], j[1])) for j in points]
                        for ct, j in enumerate(points):
                            pts_dict[pt_list[ct]] = 1
                            if ct == 0:
                                norm = self.normal_CCW(points[-2], points[ct + 1], k_val)
                            elif ct == len(points) - 1:
                                norm = self.normal_CCW(points[ct - 1], points[1], k_val)
                            else:
                                norm = self.normal_CCW(points[ct - 1], points[ct + 1], k_val)
                            norm_dict[pt_list[ct]] = (0, norm[0], norm[1])
                        extract_interior = [array(interior.coords) for interior in shape.interiors]
                        if extract_interior:
                            if not LinearRing(shape.exterior.coords).is_ccw:
                                extract_interior = extract_interior[0]
                            else:
                                extract_interior = extract_interior[0][::-1]
                            pt_list = [str((x_dict[str(j[0:2])], j[0], j[1])) for j in extract_interior]
                            for ct, j in enumerate(extract_interior):
                                pts_dict[pt_list[ct]] = 1
                                if ct == 0:
                                    norm = self.normal_CW(extract_interior[-2], extract_interior[ct + 1], k_val)
                                elif ct == len(extract_interior) - 1:
                                    norm = self.normal_CW(extract_interior[ct - 1], extract_interior[1], k_val)
                                else:
                                    norm = self.normal_CW(extract_interior[ct - 1], extract_interior[ct + 1], k_val)
                                norm_dict[pt_list[ct]] = (0, norm[0], norm[1])
                elif type == 'Polygon':
                    
                # 6b. For Polygon slices:
                #     i.      Create a new array "points" containing the new coordinates of the slice
                #     ii.     For each point, map original x with new y,z coordinates to logical True
                #     iii.    Find the normal in the x,y,z coordinates and save the normals in a dictionary
                #     iv.     Repeat i-iii. for interior coordinates
                    
                    if LinearRing(slice_shape.exterior.coords).is_ccw:
                        points = array(slice_shape.exterior.coords)
                    else:
                        points = array(slice_shape.exterior.coords)[::-1]  # reverses order so its CCW
                    pt_list = [str((x_dict[str(j[0:2])], j[0], j[1])) for j in points]
                    for ct, j in enumerate(points):
                        pts_dict[pt_list[ct]] = 1
                        if ct == 0:
                            norm = self.normal_CCW(points[-2], points[1], k_val)
                        elif ct == len(points) - 1:
                            norm = self.normal_CCW(points[ct - 1], points[1], k_val)
                        else:
                            norm = self.normal_CCW(points[ct - 1], points[ct + 1], k_val)
                        norm_dict[pt_list[ct]] = (0, norm[0], norm[1])
                    extract_interior = [array(interior.coords) for interior in slice_shape.interiors]
                    if extract_interior:
                        if not LinearRing(slice_shape.exterior.coords).is_ccw:
                            extract_interior = extract_interior[0]
                        else:
                            extract_interior = extract_interior[0][::-1]
                        pt_list = [str((x_dict[str(j[0:2])], j[0], j[1])) for j in extract_interior]
                        for ct, j in enumerate(extract_interior):
                            pts_dict[pt_list[ct]] = 1
                            if ct == 0:
                                norm = self.normal_CW(extract_interior[-2], extract_interior[1], k_val)
                            elif ct == len(extract_interior) - 1:
                                norm = self.normal_CW(extract_interior[ct - 1], extract_interior[1], k_val)
                            else:
                                norm = self.normal_CW(extract_interior[ct - 1], extract_interior[ct + 1], k_val)
                            norm_dict[pt_list[ct]] = (0, norm[0], norm[1])
                x_dict.clear()
        print("Time x: ", time() - t_x)
        t2 = time() - t1
        print("Total Time: ", t2)

        # writing to file
        points_saved = list(pts_dict.keys())
        #print(points_saved)

        print('saving to file')
        with open(self.out_file[:-4] + '_dict.pkl', 'wb') as f:
            pickle.dump(norm_dict, f, pickle.HIGHEST_PROTOCOL)
        with open(self.out_file, "w+") as f: # output point cloud with corresponding normals
            for k in range(len(points_saved)):
                pt_eval = list(points_saved[k][1:-1].split(", "))
                norm = norm_dict["(" + str(points_saved[k][1:-1]) + ")"]
                f.write("%s %s %s %f %f %f\n" % (pt_eval[0], pt_eval[1], pt_eval[2], norm[0], norm[1], norm[2]))

def main():
    Reconstruction.out_file_tag = ["CA1_", "CA3_", "Top_", "Septal_", "Temporal_"]
    Reconstruction.out_file_tag = Reconstruction.out_file_tag[0]
    out_file = Reconstruction.out_file_tag + Reconstruction.fileID + Reconstruction.out_file_ext
    a = Reconstruction("src/vox_files/data_vox_CA1.npy", out_file=out_file)
    #b = Reconstruction("src/vox_files/data_vox_DG_top.npy", out_file=out_file)
    #c = Reconstruction("src/vox_files/data_vox_DG_septal.npy", out_file=out_file)
    #d = Reconstruction("src/vox_files/data_vox_DG_temporal.npy", out_file=out_file)
    #e = Reconstruction("src/vox_files/data_vox_CA3.npy", out_file=out_file)
    for i in a.reconstruct():
       continue
        # print(i)
#
if __name__ == '__main__':
     run('main()')