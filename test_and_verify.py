import torch
import numpy as np
import cupy as cp
from PIL import Image
from shapely.geometry import box, Polygon
import matplotlib.pyplot as plt
from LoadPoseGraph import load_d2pts_data, load_img_data, load_vins_pose_graph
from collections import defaultdict
import os
import cv2
from torch.utils.data import Dataset
from collections import defaultdict
from scipy.spatial.transform import Rotation as R
from ImageProjection import rotate_trans_merge

pos_graph_path = "../output/pose_graph/pose/stamped_traj_estimate_mono_pg.txt"
rgb_image_path = "../output/pose_graph/color_img/"
depth_image_path = "../output/pose_graph/depth/"
d3_points_path = "../output/pose_graph/3d_points/"
vins_pose_path = "../output/pose_graph/vins_result_loop.txt"
d2_points_path = "../output/pose_graph/key_points/"


fx = 616.591125488
fy = 616.796264648
cx = 324.219360351
cy = 239.427017211
K_ = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

def fit_line(points, origin_coor):
    points_diff = points[:, 0:2] - origin_coor
    k = points_diff[:, 1] / points_diff[:, 0]
    #print("The k is: ")
    #print(k.shape)
    b = points[:, 1]- k * points[:, 0]
    line = np.hstack((k.reshape(-1, 1), b.reshape(-1, 1)))
    return line

def line_point_extraction(points, image):
    n_u, n_v = (image.shape[1], image.shape[0])
    origin_coor = np.array([n_u/2, n_v/2]).astype(np.int)
    line = fit_line(points, origin_coor)
    points = points[:, 0:2].reshape(-1, 2)
    points_diff = (points[:, 0] - origin_coor[0]) / 2
    points_diff = points_diff.astype(np.int)

    x_coords = (origin_coor[0] + points_diff).reshape(-1 ,1)
    ext_r = np.ones((x_coords.shape[0], 1))
    x_coor_ext = np.hstack((x_coords, ext_r))
    #Compute [k b].T * [u, 1] = ku + b = v
    y_coords = np.matmul(line, x_coor_ext.T)
    y_coords = y_coords.astype(np.int)
    #Extract the diagonal elements
    y_coords = (y_coords[np.arange(y_coords.shape[0]), np.arange(y_coords.shape[0])]).reshape(-1, 1)
    line_points = np.hstack((x_coords, y_coords))
    return line_points

"""
Get the consecutive line points between every i-th key points
and origin points.
Step 1: Fit the line using original point and key points, the
output is a matrix, its shape is [key_points.shape[0], 2], and
the content is: [[k_n, b]....]
Step 2: Get the points between i-th key point and original point,
the output is a list, contains [p1_u, p2_u, ...., pm_u], it only contains
the u coordinate of the image.
Step 3: Using step 1 result multiply the Step 2 result, the i-th row
is the result_y of step 2. 
"""
def line_consecutive_extraction(points, image):
    n_u, n_v = (image.shape[1], image.shape[0])
    origin_coor = np.array([n_u/2, n_v/2]).astype(np.int)
    line = fit_line(points, origin_coor)
    points = points[:, 0:2].reshape(-1, 2)
    gap = 0
    points_fit = np.array([0])
    for i in range(points.shape[0]):
        #Get the points between i-th key point to origin point
        point_end_u = points[i][0]
        if(origin_coor[0] > point_end_u):
            gap = -1
            point_u_list = np.arange(origin_coor[0] + 1, point_end_u, gap)
        if(origin_coor[0] <= point_end_u):
            gap = 1
            point_u_list = np.arange(origin_coor[0] - 1, point_end_u, gap)

        if (len(point_u_list) == 0):
            print("The length of list is 0!")
            continue

        point_u_list = point_u_list.reshape(-1, 1)
        ext_r = np.ones((point_u_list.shape[0], 1))
        point_u_array = np.hstack((point_u_list, ext_r))
        point_v_list = np.matmul(line, point_u_array.T)
        point_v_list = point_v_list[i, :].reshape(-1, 1)
        points_single_fit = np.hstack((point_u_list, point_v_list))
        if(i == 0):
            points_fit = points_single_fit
        else:
            points_fit = np.vstack((points_fit, points_single_fit))

        points_fit = points_fit.astype(np.int)

    return points_fit

def show_image_points(color_img, points, num_iter):
    line_points = line_consecutive_extraction(points, color_img)
    center_point_x = int(color_img.shape[1]/2)
    center_point_y = int(color_img.shape[0]/2)
    cv2.circle(color_img, (center_point_x, center_point_y), 1, (255, 0, 0), 10)

    for i in range(points.shape[0]):
        point_x = int(points[i][0])
        point_y = int(points[i][1])
        cv2.circle(color_img, (point_x, point_y), 1, (0, 0, 255), 4)

    for j in range(line_points.shape[0]):
        line_point_x = line_points[j][0]
        line_point_y = line_points[j][1]
        cv2.circle(color_img, (line_point_x, line_point_y), 1, (0, 255, 0), 4)

    cv2.imwrite("../line_image_save/image_" + str(num_iter) + ".png", color_img)
    return

def image_coords_ops(img_data, K):
    img_h, img_w = img_data.shape
    nx, ny = (img_w, img_h)
    x = np.linspace(0, nx, nx)
    x = np.linspace(0, nx - 1, nx)
    y = np.linspace(0, ny - 1, ny)
    xv, yv = np.meshgrid(x, y)
    coords_uv = np.dstack((xv, yv)).reshape(-1, 2)
    coords_uv = coords_uv.astype(np.int)
    depth_vec = img_data[coords_uv[:, 1], coords_uv[:, 0]]/1000.0

    # filter out the zero depth region to avoid numerical error
    depth_vec[depth_vec == 0] = 1e-3
    coords_xy = coords_uv
    ext_c = np.ones((coords_xy.shape[0], 1))
    coords_xy = np.hstack((coords_xy, ext_c))
    coords_xyc = np.matmul(np.linalg.inv(K), coords_xy.T).T
    coords_xyz = np.hstack((coords_xyc[:, :2], depth_vec.reshape(-1, 1)))
    #print(coords_xyz)
    return coords_xyz

def reproject_pixel_iou(depth_img_list, num_iter, T_w_c_, K):
    depth_img_cur = depth_img_list[num_iter]
    T_w_cur = T_w_c_[num_iter]
    iou_list = []
    points_count = 0
    test_list = [0]
    point_xyz = image_coords_ops(depth_img_cur, K)
    ext_r = np.ones((point_xyz.shape[0], 1))
    point_xyz = np.hstack((point_xyz, ext_r))
    print("point_xyz after insert ext_r is: ")
    print(point_xyz)
    print("\n")
    for i in range(len(test_list)):
        T_w_ref = T_w_c_[i]
        T_ref_cur = np.matmul(np.linalg.inv(T_w_ref), T_w_cur)
        point_proj = np.matmul(T_ref_cur, point_xyz.T)
        print("The point_proj is: ")
        print(point_proj.T)
        print("\n")
        point_proj_norm = point_proj[0:3, :]/point_proj[2, :]
        print("The point_proj_norm is: ")
        print(point_proj_norm.T)

        #print("The point_proj_norm is: ")
        #print(point_proj_norm)
        #print("\n")
        point_proj_uv = np.matmul(K, point_proj_norm)
        point_proj_uv = point_proj_uv.astype(np.int).T

        print("The point_proj_uv: ")
        print(point_proj_uv)
        print("\n")
        point_proj_u = (point_proj_uv[:, 0] >= 0) & (point_proj_uv[:, 0] <= depth_img_cur.shape[1])
        point_proj_v = (point_proj_uv[:, 1] >= 0) & (point_proj_uv[:, 1] <= depth_img_cur.shape[0])
        point_proj_result = (point_proj_u == True) & (point_proj_v == True)

        intersect_area = np.sum(point_proj_result)
        union_area = depth_img_cur.shape[0] * depth_img_cur.shape[1] * 2 - intersect_area
        iou = intersect_area / union_area
        iou_list.append(iou)
    return iou_list

def reproject_calculate_iou(depth_img_list, color_img_list, num_iter, T_w_c_, fx_, fy_, cx_, cy_):
    color_img_cur = color_img_list[num_iter]
    depth_img_cur = depth_img_list[num_iter]
    T_w_cur = T_w_c_[num_iter]
    iou_list = []
    points_count = 0

    # corner_points represented by np.array([u, v, 1])
    corner_points = [np.array([0, 0, 1]), np.array([639, 0, 1]),
                     np.array([639, 479, 1]), np.array([0, 479, 1])]

    corner_point_uv_cur = [[0, 0], [639, 0], [639, 479], [0, 479]]
    ious = []
    num_image = 0
    for i in range(len(color_img_list)):
        num_image += 1
        print("The i-th image is: ")
        print(i)
        point_3d_cur = np.ones((4, 1))
        color_img_ref = color_img_list[i]
        T_w_ref = T_w_c_[i]  # T_w_ref = T_w_c2 T_w_cur = T_w_c1 We need T_c2_c1 = (T_w_c2)^-1 * T_w_c1
        T_ref_cur = np.matmul(np.linalg.inv(T_w_ref), T_w_cur)
        corner_point_proj_list = []
        for corner_point_uv in corner_points:
            corner_point_xy = np.ones((4, 1))
            corner_point_depth = depth_img_cur[int(corner_point_uv[1]), int(corner_point_uv[0])]
            corner_point_xy[0] = ((corner_point_uv[0] - cx_) * corner_point_depth) / fx_
            corner_point_xy[1] = ((corner_point_uv[1] - cy_) * corner_point_depth) / fy_
            corner_point_xy[2] = corner_point_depth

            corner_point_proj = np.matmul(T_ref_cur, corner_point_xy)

            print("The corner_point_xy is: ")
            print(corner_point_xy)
            print("THe corner_point_depth is: ")
            print(corner_point_depth)

            corner_point_proj_u = int(fx_ * (corner_point_proj[0] / corner_point_proj[2]) + cx_)
            corner_point_proj_v = int(fy_ * (corner_point_proj[1] / corner_point_proj[2]) + cy_)

            corner_point_proj_list.append([corner_point_proj_u, corner_point_proj_v])

        if (len(corner_point_proj_list) != 4):
            print("The length of the polygon list is not equal to 4!")
            return None
        #print("The corner_point_proj is: ")
        #print(corner_point_proj_list)
        polygon_cur_shape = Polygon(corner_point_proj_list)
        polygon_ref_shape = Polygon(corner_point_uv_cur)
        polygon_cur_shape = polygon_cur_shape.buffer(0.01)
        polygon_ref_shape = polygon_ref_shape.buffer(0.01)
        polygon_intersection = polygon_ref_shape.intersection(polygon_cur_shape).area
        polygon_union = polygon_ref_shape.union(polygon_cur_shape).area
        IOU = polygon_intersection / polygon_union
        print("The IOU is: ")
        print(IOU)
        ious.append(IOU)
    return ious

pose_data_list = load_vins_pose_graph(vins_pose_path)
pose_list = rotate_trans_merge(pose_data_list)
img_data_list = load_img_data(rgb_image_path, mode= "image")
depth_data_list = load_img_data(depth_image_path, mode = "depth")
#print(depth_data_list)
key_points_dict = load_d2pts_data(d2_points_path, 640, 480)
#print(len(key_points_dict["uv"][0]))
#image_points_trans(pose_list, d3pts_data_list, img_data_list, fx, fy, cx, cy)

#test = cp.arange(0, 4, 1)
#print(test)
for i in range(len(img_data_list)):
    print("The " + str(i) + " th image")
    color_image = img_data_list[i]
    key_points_ =  np.array(key_points_dict["uv"][i])
    key_points_ = key_points_.reshape(key_points_.shape[0], 3)
    key_points_ = key_points_.astype(np.int)
    show_image_points(color_image, key_points_, i)

#line_consecutive_extraction(key_points_, depth_image)
#ious = reproject_calculate_iou(depth_data_list, img_data_list, 66, pose_list, fx, fy, cx, cy)
#result_iou_list = []
#for i in range(len(depth_data_list)):
#    print("The " + str(i) + " th loop")
#iou_list = reproject_pixel_iou(depth_data_list, 1, pose_list, K_)
#iou_list = np.array(iou_list).reshape(-1, 1)
#result_iou_list.append(iou_list)
#print("The iou_list is: ")
#print(iou_list)
#print("The length of the iou list is: ")
#print(len(result_iou_list))
#x_ = np.arange(10, 10, -1)
#print(x_)
#k = 0.5
#b = 300
#y_ = k * x_ + b
#coor = np.vstack((x_, y_)).T
#print(coor)