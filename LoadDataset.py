import torch
#import cupy as cp
import numpy as np
import time
from PIL import Image
#from shapely.geometry import box, Polygon
from skimage.util.shape import view_as_windows
from LoadPoseGraph import load_d2pts_data, load_img_data
from collections import defaultdict
import h5py

import os
import cv2
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from scipy.spatial.transform import Rotation as R
from torchvision import transforms
#save_path = "../data_generate/"
#out_file_dir = save_path + "torch_data/"
fx = 585.0
fy = 585.0
cx = 320.0
cy = 240.0

#np.set_printoptions(threshold=np.inf)

transform_ = transforms.Compose([
    transforms.ToTensor()
])

camcalib_ = [fx, fy, cx, cy]

class LoadDataset(Dataset):
    def __init__(self, dataset_dir, save_data_dir, transform, patch_size, image_height, image_width, Tic, camcalib):
        self.patch_size = patch_size
        self.dataset_dir = dataset_dir
        self.save_data_dir = save_data_dir

        self.transform = transform
        self.img_dir = dataset_dir + "color_img/"
        self.img_file_list = []

        self.depth_dir = dataset_dir + "depth/"
        self.key_points_dir = dataset_dir + "key_points/"

        self.d3_points_dir = dataset_dir + "3d_points/"

        self.pose_list_dir = dataset_dir + "vins_result_loop.txt"
        self.pose_list = []
        self.T_w_c_ = []

        self.Tic = Tic
        self.img_height = int(image_height)
        self.img_width = int(image_width)
        self.fx = camcalib[0]
        self.fy = camcalib[1]
        self.cx = camcalib[2]
        self.cy = camcalib[3]
        self.K = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

        self.color_img_list = []
        self.depth_img_list = []

        self.key_points_extraction = defaultdict(list)
        self.key_points_array_list = []

        self.image_key_status_list = []
        # Read pose list from vins_result_loop.txt file
        with open(self.pose_list_dir) as p:
            pose_list = []
            for line in p.readlines():
                single_pose_data = {}
                temp_line = line.split(',')
                single_pose_data['timestamp'] = float(temp_line[0])
                single_pose_data['t_x'] = float(temp_line[1])
                single_pose_data['t_y'] = float(temp_line[2])
                single_pose_data['t_z'] = float(temp_line[3])
                single_pose_data['q_w'] = float(temp_line[4])
                single_pose_data['q_x'] = float(temp_line[5])
                single_pose_data['q_y'] = float(temp_line[6])
                single_pose_data['q_z'] = float(temp_line[7])
                pose_list.append(single_pose_data)

        print("Start to filter!")
        self.image_key_points_filter(pose_list)
        self.rotate_trans_merge()
        print("Filter finish!!")

        if len(self.color_img_list) != len(self.T_w_c_) or len(self.color_img_list) != len(self.depth_img_list):
            print("The number of parameter is not cosistent!")

        print("The length of pose is: ")
        print(len(self.color_img_list))
        print("The length of color image is: ")
        print(len(self.color_img_list))
    def __len__(self):
        return len(self.color_img_list)

    def get_image_length(self):
        return len(self.color_img_list)

    def get_depth_length(self):
        return len(self.depth_img_list)

    def get_key_points_num(self):
        return len(self.key_points_extraction["uv"])

    def get_single_key_points(self, i):
        return len(self.key_points_extraction["uv"][i])

    def fit_line(self, points, origin_coor):
        points_diff = points[:, 0:2] - origin_coor
        k = points_diff[:, 1] / points_diff[:, 0]
        # print("The k is: ")
        # print(k.shape)
        b = points[:, 1] - k * points[:, 0]
        line = np.hstack((k.reshape(-1, 1), b.reshape(-1, 1)))
        return line

    def line_consecutive_extraction(self, points):
        n_u, n_v = (self.img_width, self.img_height)
        origin_coor = np.array([n_u / 2, n_v / 2]).astype(np.int)
        origin_coor_x = int(n_u / 2)
        origin_coor_y = int(n_v / 2)
        line = self.fit_line(points, origin_coor)
        points = points[:, 0:2].reshape(-1, 2)
        gap = 0
        points_fit = np.array([0])
        for i in range(points.shape[0]):
            # Get the points between i-th key point to origin point
            point_end_u = points[i][0]
            if (origin_coor[0] > point_end_u):
                gap = -1
                point_u_list = np.arange(origin_coor_x + 1, point_end_u, gap, dtype=int)
            if (origin_coor[0] <= point_end_u):
                gap = 1
                point_u_list = np.arange(origin_coor_x - 1, point_end_u, gap, dtype=int)

            if (len(point_u_list) == 0):
                print("The length of list is 0!")
                continue

            point_u_list = point_u_list.reshape(-1, 1)
            ext_r = np.ones((point_u_list.shape[0], 1))
            point_u_array = np.hstack((point_u_list, ext_r))
            point_v_list = np.matmul(line, point_u_array.T)
            point_v_list = point_v_list[i, :].reshape(-1, 1)
            points_single_fit = np.hstack((point_u_list, point_v_list))
            if (i == 0):
                points_fit = points_single_fit
            else:
                points_fit = np.vstack((points_fit, points_single_fit))

            points_fit = points_fit.astype(np.int)

        return points_fit
    '''
    def image_coords_ops(self, img_data):
        img_h, img_w = img_data.shape
        nx, ny = (img_w, img_h)
        #x = np.linspace(0, nx - 1, nx)
        #y = np.linspace(0, ny - 1, ny)
        #xv, yv = np.meshgrid(x, y)
        x = cp.linspace(0, nx - 1, nx)
        y = cp.linspace(0, ny - 1, ny)
        xv, yv = cp.meshgrid(x, y)
        coords_uv = cp.dstack((xv, yv)).reshape(-1, 2)
        coords_uv = coords_uv.astype(cp.int)
        depth_vec = img_data[coords_uv[:, 1], coords_uv[:, 0]] / 1000.0
        # filter out the zero depth region to avoid numerical error
        depth_vec[depth_vec == 0] = 1e-10
        coords_xy = coords_uv
        ext_c = cp.ones((coords_xy.shape[0], 1))
        coords_xy = cp.hstack((coords_xy, ext_c))
        coords_xyz = coords_xy * depth_vec.reshape(-1, 1)
        coords_xyz = cp.matmul(cp.linalg.inv(cp.array(self.K)), coords_xyz.T).T
       # coords_xyc = np.matmul(np.linalg.inv(np.asarray(K)), coords_xy.T).T
       # coords_xyz = np.hstack((coords_xyc[:, :2], depth_vec.reshape(-1, 1)))
        return coords_xyz, coords_uv
        '''

    def image_coords_ops_numpy(self, img_data):
        img_h, img_w = img_data.shape
        nx, ny = (img_w, img_h)
        #x = np.linspace(0, nx - 1, nx)
        #y = np.linspace(0, ny - 1, ny)
        #xv, yv = np.meshgrid(x, y)
        x = np.linspace(0, nx - 1, nx)
        y = np.linspace(0, ny - 1, ny)
        xv, yv = np.meshgrid(x, y)
        coords_uv = np.dstack((xv, yv)).reshape(-1, 2)
        coords_uv = coords_uv.astype(np.int)
        depth_vec = img_data[coords_uv[:, 1], coords_uv[:, 0]] / 1000.0
        # filter out the zero depth region to avoid numerical error
        depth_vec[depth_vec == 0] = 1e-10
        depth_vec[depth_vec >= 60] = 1e-10

        coords_xy = coords_uv
        ext_c = np.ones((coords_xy.shape[0], 1))
        coords_xy = np.hstack((coords_xy, ext_c))
        coords_xyz = coords_xy * depth_vec.reshape(-1, 1)
        coords_xyz = np.matmul(np.linalg.inv(np.array(self.K)), coords_xyz.T).T
       # coords_xyc = np.matmul(np.linalg.inv(np.asarray(K)), coords_xy.T).T
       # coords_xyz = np.hstack((coords_xyc[:, :2], depth_vec.reshape(-1, 1)))
        return coords_xyz, coords_uv

    def boundary_filter(self, key_points_uv):
        key_points_u = (key_points_uv[:, 0]>=0) & (key_points_uv[:, 0] < self.img_width)
        key_points_v = (key_points_uv[:, 1]>=0) & (key_points_uv[:, 1] < self.img_height)
        key_points_uv_selection = np.where((key_points_u == True) & (key_points_v == True))
        key_points_uv_in = key_points_uv[key_points_uv_selection[0], :]
        return key_points_uv_in, key_points_uv_selection

    def z_buffer_filter(self, depth_img_ref, key_points_uv_in, key_points_xyz_proj, key_points_uv_selection):
        key_points_xyz_proj_transpose = key_points_xyz_proj.T
        #print("The shape of key_points_xyz_proj is: ")
        #print(key_points_xyz_proj_transpose.shape)
        key_points_xyz_proj_transpose = key_points_xyz_proj_transpose[key_points_uv_selection[0], :]
        depth_val_proj = key_points_xyz_proj_transpose[:, 2]
        #print("The depth_val_proj is: ")
        #print(depth_val_proj)
        #print("The shape of depth_val_proj is: ")
        #print(depth_val_proj.shape)
        depth_val_ref = depth_img_ref[key_points_uv_in[:, 1], key_points_uv_in[:, 0]]/1000.0
        #print("The depth val ref is: ")
        #print(depth_val_ref)
        #print("The shape of depth_val_ref is:")
        #print(depth_val_ref.shape)

        depth_buffer_idx = np.where(depth_val_ref >= (depth_val_proj - 0.05))
        #depth_buffer_idx = np.where(depth_buffer_bool == True)
        #print("The depth_buffter_idx is: ")
        #print(depth_buffer_idx)

        if len(depth_buffer_idx[0]) == 0:
            return np.empty(shape=(0, 3))
        else:
            #print("The key_points_uv_in is: ")
            return key_points_uv_in[depth_buffer_idx[0], :]


    def image_reproject_and_show(self, num_iter):
        depth_cur = self.depth_img_list[num_iter].copy()
        color_cur = self.color_img_list[num_iter].copy()
        T_w_cur = self.T_w_c_[num_iter]
        key_points_xyz, key_points_uv_ = self.image_coords_ops_numpy(depth_cur)
        #key_points_xyz = cp.asnumpy(key_points_xyz)
        ext_c = np.ones((key_points_xyz.shape[0], 1))
        key_points_xyz = np.hstack((key_points_xyz, ext_c))
        test = [0]
        iou_list = []
        intersect_area = 0
        file_dir = self.save_data_dir + "/reproject_save_unique/" + str(num_iter) + "/"
        #if not os.path.exists(file_dir):
        #    os.makedirs(file_dir)

        for i in range(len(self.color_img_list)):
            color_ref = self.color_img_list[i].copy()
            depth_ref = self.depth_img_list[i].copy()
            T_w_ref = self.T_w_c_[i]
            T_ref_cur = np.matmul(np.linalg.inv(T_w_ref), T_w_cur)
            key_points_xyz_proj = np.matmul(T_ref_cur, key_points_xyz.T).T
            key_points_xyz_idx = np.where(key_points_xyz_proj[:, 2] > 0)

            if(len(key_points_xyz_idx[0]) == 0):
                intersect_area = 0
                iou_list.append(intersect_area)
                continue
            key_points_xyz_proj = key_points_xyz_proj[key_points_xyz_idx[0], :].T
            key_points_xyz_norm = (key_points_xyz_proj[0:3, :]/key_points_xyz_proj[2, :])

            key_points_uv_proj = np.matmul(self.K, key_points_xyz_norm).T
            key_points_uv_proj = key_points_uv_proj.astype(np.int)
            key_points_uv_in, key_points_uv_selection = self.boundary_filter(key_points_uv_proj)
            key_points_uv_in = self.z_buffer_filter(depth_ref, key_points_uv_in, key_points_xyz_proj, key_points_uv_selection)
            #if key_points_uv_in.shape[0] != 0:
            #    key_points_uv_in = np.unique(key_points_uv_in, axis=0)
            intersect_area = key_points_uv_in.shape[0]
            union_area = 2 * self.img_width * self.img_height - intersect_area
            iou = intersect_area / union_area
            iou_list.append(iou)
        #return iou_list
            #merge_img = self.key_points_show(key_points_uv_, key_points_uv_in, color_cur, color_ref, mode = "iou")
            #print("saving " + str(i) + " image !")
            #cv2.imwrite(file_dir + str(i) +".png", merge_img)
        return iou_list

    def image_coords_key_points_ops(self, num_iter):
        img_data = self.depth_img_list[num_iter]
        key_points_array = self.key_points_array_list[num_iter]
        depth_vec = img_data[key_points_array[:, 1], key_points_array[:, 0]] / 1000.0
        # filter out the zero depth region to avoid numerical error
        depth_vec[depth_vec == 0] = 1e-10
        coords_xy = key_points_array
        ext_c = np.ones((coords_xy.shape[0], 1))
        coords_xy = np.hstack((coords_xy, ext_c)) * depth_vec.reshape(-1, 1)
        coords_xyz = np.matmul(np.linalg.inv(self.K), coords_xy.T).T
        return coords_xyz

    '''
    def reproject_pixel_iou(self, num_iter):
        depth_img_cur = cp.asarray(self.depth_img_list[num_iter])
        T_w_cur = cp.asarray(self.T_w_c_[num_iter])
        iou_list = []
        point_xyz, point_uv = self.image_coords_ops(depth_img_cur)
        ext_r = cp.ones((point_xyz.shape[0], 1))
        point_xyz = cp.hstack((point_xyz, ext_r))
        test = [0]
        for i in range(len(self.depth_img_list)):
        #for i in range(len(test)):
            T_w_ref = cp.asarray(self.T_w_c_[i])
            T_ref_cur = cp.matmul(cp.linalg.inv(T_w_ref), T_w_cur)
            point_proj = cp.matmul(T_ref_cur, point_xyz.T)
            point_proj_norm = point_proj[0:3, :] / point_proj[2, :]
            point_proj_uv = cp.matmul(cp.asarray(self.K), point_proj_norm)
            point_proj_uv = point_proj_uv.astype(np.int).T
            point_proj_u = (point_proj_uv[:, 0] >= 0) & (point_proj_uv[:, 0] <= depth_img_cur.shape[1])
            point_proj_v = (point_proj_uv[:, 1] >= 0) & (point_proj_uv[:, 1] <= depth_img_cur.shape[0])
            intersect_idx = cp.where((point_proj_v == True) & (point_proj_u) == True)
            #print("The ietnrsect idx is: ")
            #print(intersect_idx)
            #print("\n")

            intersect_coords = point_proj_uv[intersect_idx[0], :]
            #print("The intersect_coords is: ")
            #print(intersect_coords)
            intersect_coords = cp.asnumpy(intersect_coords)
            #print("The intersect_coords after unique operator")
            #print(intersect_coords)
            #print("\n")
            # print(intersect_coor)
            if(intersect_coords.shape[0] != 0):
                point_proj_uv_num = np.unique(intersect_coords, axis = 0)
                intersect_area = point_proj_uv_num.shape[0]
            else:
                intersect_area = 0

            union_area = depth_img_cur.shape[0] * depth_img_cur.shape[1] * 2 - intersect_area
            iou = float(intersect_area / union_area)
            iou_list.append(iou)
            #print("THe iou is: ")
            #print(iou)
        return iou_list
        '''
    def reproject_and_show(self, num_iter):
        test = [0, 1]
        for i in range(len(self.depth_img_list)-1):
        #for i in range(len(test)- 1):
            #print("The pose list is")
            #print(self.pose_list[i+1])
            T_w_cur = self.T_w_c_[i]
            T_w_ref = self.T_w_c_[i+1]

            color_cur = self.color_img_list[i].copy()
            color_ref = self.color_img_list[i+1].copy()
            depth_cur = self.depth_img_list[i]
            T_ref_cur = np.matmul(np.linalg.inv(T_w_ref), T_w_cur)

            key_points_array_cur = self.key_points_array_list[i]
            depth_value = depth_cur[key_points_array_cur[:, 1], key_points_array_cur[:, 0]].reshape(-1, 1) / 1000.0

            key_points_array_cur = key_points_array_cur * depth_value
            ext_c = np.ones((key_points_array_cur.shape[0], 1))
            key_points_array_cur = np.hstack((key_points_array_cur, depth_value))

            key_points_xy_cur = np.matmul(np.linalg.inv(self.K), key_points_array_cur.T).T

            key_points_xyz_cur = np.hstack((key_points_xy_cur, ext_c))

            key_points_xyz_proj = np.matmul(T_ref_cur, key_points_xyz_cur.T)

            key_points_xyz_proj = key_points_xyz_proj[0:3, :] / key_points_xyz_proj[2, :]
            key_points_uv_proj = np.matmul(self.K, key_points_xyz_proj).T
            key_points_u_proj = (key_points_uv_proj[:, 0] >= 0) & (key_points_uv_proj[:, 0] <= self.img_width)
            key_points_v_proj = (key_points_uv_proj[:, 1] >= 0) & (key_points_uv_proj[:, 1] <= self.img_height)
            key_points_uv_idx = np.where((key_points_u_proj == True) & (key_points_v_proj == True))
            key_points_uv_proj = key_points_uv_proj[key_points_uv_idx[0], :]
            self.key_points_show(self.key_points_array_list[i], key_points_uv_proj, color_cur, color_ref, mode = "reproject")

    def key_points_show(self, key_points_uv, key_points_uv_proj, img_cur, img_ref, mode = "iou"):
        if mode != "iou":
            for i in range(key_points_uv.shape[0]):
                u_cur = int(key_points_uv[i][0])
                v_cur = int(key_points_uv[i][1])

                cv2.circle(img_cur, (u_cur, v_cur), 1, (255, 0, 0), 2)

        if key_points_uv_proj.shape[0] == 0:
            return np.hstack([img_cur, img_ref])

        for j in range(key_points_uv_proj.shape[0]):
            u_ref = int(key_points_uv_proj[j][0])
            v_ref = int(key_points_uv_proj[j][1])
            cv2.circle(img_ref, (u_ref, v_ref), 1, (0, 255, 0), 2)

        imgs = np.hstack([img_cur, img_ref])
        return imgs

    def rotate_trans_merge(self):
        for pose in self.pose_list:
            T_matrix = np.identity(4)
            Rq = [pose["q_x"], pose["q_y"], pose["q_z"], pose["q_w"]]
            trans = [pose["t_x"], pose["t_y"], pose["t_z"], 1]
            Rm = R.from_quat(Rq)

            R_matrix = Rm.as_matrix()
            T_matrix[0:3, 0:3] = R_matrix
            T_matrix[::, 3] = trans

            T_w_c = np.matmul(T_matrix, self.Tic)
            self.T_w_c_.append(T_w_c)

    def image_key_points_filter(self, pose_list):
        test_num = [0]
        for i in range(len(os.listdir(self.img_dir))):
            #print("This is " + str(i) + " th image filter!")
            single_image_status = {}
        #for i in range(len(test_num)):
            #i = 771
            key_point_file = self.key_points_dir + str(i) + "_keypoints.txt"
            with open(key_point_file) as f:
                key_points_uv = []
                key_points_xy = []
                contents = f.readlines()

                if (len(contents) < 10):
                    #print("The " + str(i) + " th image points number is too low, drop it!")
                    continue

                for line in contents:
                    key_point_uv = np.ones((3, 1))
                    key_point_xy = np.ones((4, 1))
                    temp_line = line.split()
                    key_point_uv[0] = float(temp_line[0])
                    key_point_uv[1] = float(temp_line[1])
                    key_point_xy[0] = float(temp_line[2])
                    key_point_xy[1] = float(temp_line[3])
                    if(int(key_point_uv[0]) == int(self.img_width/2) or int(key_point_uv[1]) == int(self.img_height/2)):
                        continue
                    u_ori = int(key_point_uv[0])
                    v_ori = int(key_point_uv[1])
                    patch_u = int(u_ori - self.patch_size / 2.)
                    patch_v = int(v_ori - self.patch_size / 2.)
                    if (patch_u < 0 or patch_u + self.patch_size > self.img_width or patch_v < 0 or patch_v + self.patch_size >
                            self.img_height):
                        continue
                    key_points_uv.append(key_point_uv)
                    key_points_xy.append(key_point_xy)

            key_points_array = np.array(key_points_uv)
            key_points_array = key_points_array[:, 0:2]
            key_points_array = key_points_array.reshape(len(key_points_uv), 2)
            key_points_array = key_points_array.astype(np.int)

            if (key_points_array.shape[0] > 128):
                #print("The " + str(i) + " th key points number is larger than 128!")
                idx = np.random.randint(key_points_array.shape[0], size = 128)
                key_points_choice = key_points_array[idx, :]
                #print("The key_points_choice is: ")
                #print(key_points_choice)
                #print("\n")

                single_image_status["idx"] = i
                single_image_status["key_points_status"] = "enough"
                single_image_status["key_points_choice"] = key_points_choice
                single_image_status["key_points_surplus"] = None

                self.image_key_status_list.append(single_image_status)
                self.key_points_array_list.append(key_points_choice)

            else:
                #print("The " + str(i) + " th key points number is smaller than 128!")
                random_select_num = 128 - key_points_array.shape[0]
                points_fit = self.line_consecutive_extraction(key_points_array)
                if points_fit.shape[0] < random_select_num:
                    print("Can not find enough patch image! Drop it!")
                    continue
                idx = np.random.randint(points_fit.shape[0], size = random_select_num)
                key_points_choice_ = points_fit[idx, :]
                #print("The key points choice is: ")
                #print(key_points_choice.shape)
                #print("\n")
                #print("The key_points_array is: ")
                #print(key_points_array.shape)
                key_points_choice = np.r_[key_points_array, key_points_choice_]

                single_image_status["idx"] = i
                single_image_status["key_points_status"] = "not_enough"
                single_image_status["key_points_choice"] = key_points_array
                single_image_status["key_points_surplus"] = key_points_choice_
                self.image_key_status_list.append(single_image_status)
                self.key_points_array_list.append(key_points_choice)

            color_img = cv2.imread(self.img_dir + str(i) + "_image.png")
            depth_img = cv2.imread(self.depth_dir + str(i) + "_depth.png", -1)
            self.color_img_list.append(color_img)
            self.depth_img_list.append(depth_img)
            self.pose_list.append(pose_list[i])
            self.key_points_extraction["uv"].append(key_points_uv)
            self.key_points_extraction["xy"].append(key_points_xy)

    def extract_patch_and_draw(self, num_iter):

        color_img = self.color_img_list[num_iter]
        key_points_status = self.image_key_status_list[num_iter]
        key_points_ = key_points_status["key_points_choice"]

        for i in range(key_points_.shape[0]):
            u_ori = key_points_[i][0]
            v_ori = key_points_[i][1]
            patch_u = int(u_ori - self.patch_size / 2.)
            patch_v = int(v_ori - self.patch_size / 2.)
            cv2.rectangle(color_img, (patch_u, patch_v),(patch_u + self.patch_size, patch_v + self.patch_size), (0, 255, 0), 1)

        if key_points_status["key_points_status"] == "not_enough":
            key_points_surplus = key_points_status["key_points_surplus"]
            for i in range(key_points_surplus.shape[0]):
                u_ori = key_points_surplus[i][0]
                v_ori = key_points_surplus[i][1]
                patch_u = int(u_ori - self.patch_size / 2.)
                patch_v = int(v_ori - self.patch_size / 2.)
                cv2.rectangle(color_img, (patch_u, patch_v),(patch_u + self.patch_size, patch_v + self.patch_size), (0, 0, 255), 1)

        #return color_img
        file_name = str(num_iter).zfill(4)
        image_data_path = self.save_data_dir + "patch_image_show/" + str(file_name) + "_patch_image.png"
        image_data_dir = self.save_data_dir + "patch_image_show/"
        if not os.path.exists(image_data_dir):
            os.mkdir(image_data_dir)
        cv2.imwrite(image_data_path, color_img)

    def extract_patch(self, num_iter, patch_scales):
        color_img = self.color_img_list[num_iter]
        depth_img = self.depth_img_list[num_iter]

        key_points_array = self.key_points_array_list[num_iter]

        max_padding = self.patch_size * (2 ** max(patch_scales))
        color_img_copy = color_img.copy()
        depth_img_copy = depth_img.copy()
        color_img_copy = cv2.copyMakeBorder(color_img_copy, max_padding, max_padding, max_padding, max_padding,
                                            cv2.BORDER_CONSTANT, value=0)
        depth_img_copy = cv2.copyMakeBorder(depth_img_copy, max_padding, max_padding, max_padding, max_padding,
                                            cv2.BORDER_CONSTANT, value=0)

        torch_dict = {}
        test_len = [0]
        for patch_scale in patch_scales:
            patch_size = self.patch_size * (2 ** patch_scale)
            #print("The patch_size is: ")
            #print(patch_size)
            torch_tensor = torch.empty(1)
            for i in range(key_points_array.shape[0]):
                u_ori = key_points_array[i][0] + max_padding
                v_ori = key_points_array[i][1] + max_padding
                patch_u = int(u_ori - patch_size / 2.)
                patch_v = int(v_ori - patch_size / 2.)

                patch_color_image = torch.from_numpy(color_img_copy[patch_v: patch_v + patch_size, patch_u: patch_u + patch_size])
                patch_depth_image = torch.from_numpy((depth_img_copy[patch_v: patch_v + patch_size, patch_u: patch_u + patch_size]) / 1000.0)
                patch_depth_image = patch_depth_image.unsqueeze(2)
                patch_image_tensor = torch.cat((patch_color_image, patch_depth_image), 2).unsqueeze(0)
                if i == 0:
                    torch_tensor =  patch_image_tensor
                else:
                    torch_tensor = torch.cat((torch_tensor, patch_image_tensor), 0)

            torch_dict[str(patch_size)] = torch_tensor

        return torch_dict
            #patch_color_image_list.append(patch_color_image)
            #patch_depth_image_list.append(patch_depth_image)
            #draw_1 = cv2.rectangle(color_img, (patch_u, patch_v), (patch_u + self.patch_size, patch_v + self.patch_size), (0, 255, 0), 2)

        #cv2.imshow("draw_1", draw_1)
        #cv2.waitKey(0)
        #return patch_color_image_list, patch_depth_image_list

    #Reference from stackoverlow:https://stackoverflow.com/questions/37901186/faster-way-to-extract-patches-from-images
    def extract_patch_vec(self, num_iter):
        patch_color_image_list = []
        patch_depth_image_list = []

        color_img = self.color_img_list[num_iter]
        depth_img = self.depth_img_list[num_iter]
        key_points_array = self.key_points_array_list[num_iter]
        upper_left = key_points_array - self.patch_size // 2

        torch_tensor = torch.empty(1)
        for i in range(color_img.shape[2]):
            all_patches = view_as_windows(color_img[:, :, i], self.patch_size)
            patches = torch.from_numpy(all_patches[upper_left[:, 1], upper_left[:, 0]])
            patches = patches.unsqueeze(3)
            #print(patches.shape)
            if i == 0:
                torch_tensor = patches
            else:
                torch_tensor = torch.cat((torch_tensor, patches), 3)

        return torch_tensor

    def rgb_points_trans(self):
        for i in range(len(self.color_img_list) - 1):
            img_ref = self.color_img_list[i].copy()
            img_cur = self.color_img_list[i + 1].copy()
            T_w_ref = self.T_w_c_[i]
            T_w_cur = self.T_w_c_[i + 1]
            T_cur_ref = np.matmul(np.linalg.inv(T_w_cur), T_w_ref)
            T_test = np.matmul(np.linalg.inv(T_w_cur), T_w_cur)
            # print("The T_test is: ")
            # print(T_test)
            depth_img_ref = self.depth_img_list[i]
            depth_img_cur = self.depth_img_list[i + 1]
            points_ref_uv = self.key_points_array_list[i]
            points_ref_xy = self.key_points_extraction["xy"][i]
            for j in range(points_ref_uv.shape[0]):
                point_uv = points_ref_uv[j]
                print("The point uv is: ")
                print(point_uv)
                point_depth = depth_img_ref[int(point_uv[1]), int(point_uv[0])] / 1000.000
                print("The point depth is: ")
                print(point_depth)
                point_xy = np.ones((4, 1))
                point_xy[0] = ((point_uv[0] - self.cx) * point_depth) / self.fx
                point_xy[1] = ((point_uv[1] - self.cy) * point_depth) / self.fy
                point_xy[2] = point_depth
                point_proj = np.dot(T_cur_ref, point_xy)
                u_proj = int(fx * (point_proj[0] / point_proj[2]) + self.cx)
                v_proj = int(fy * (point_proj[1] / point_proj[2]) + self.cy)

                if u_proj > img_cur.shape[1] or u_proj < 0 or v_proj > img_cur.shape[0] or v_proj < 0:
                    continue
                cv2.circle(img_ref, (int(point_uv[0]), int(point_uv[1])), 1, (0, 255, 0), 4)
                cv2.circle(img_cur, (u_proj, v_proj), 1, (0, 0, 255), 4)

            imgs = np.hstack([img_ref, img_cur])
            cv2.imshow("multi_img", imgs)
            cv2.waitKey(0)

    def __getitem__(self, i):

        #patch_color_image_list, patch_depth_image_list = self.extract_patch(color_image, depth_image, key_points_list)
        #if len(self.depth_img_list) != len(self.color_img_list) or len(self.T_w_c_) != len(self.)
        pose = self.T_w_c_[i]
        patch_scales = [0, 1, 2]
        torch_tensor_dict = self.extract_patch(i, patch_scales)
        #torch_tensor = self.extract_patch_vec(i)

        iou_list = self.image_reproject_and_show(i)
        key_points_choice = self.key_points_array_list[i]
        key_points_xyz = self.image_coords_key_points_ops(i)
        self.extract_patch_and_draw(i)

        #print("torch_tensor_dict keys: ")
        #print(torch_tensor_dict.keys())
        print("The " + str(i) +" th size of the each patch_size is: ")
        print(torch_tensor_dict["16"].size())
        print(torch_tensor_dict["32"].size())
        print(torch_tensor_dict["64"].size())
        return {"patch_16": torch_tensor_dict["16"],
                "patch_32": torch_tensor_dict["32"],
                "patch_64": torch_tensor_dict["64"],
                "ious": iou_list, "key_points_uv": key_points_choice, "key_points_xyz": key_points_xyz,"pose":pose}
        #return 1



def write_hdf5(out_file_dir, arr_dict):
    with h5py.File(out_file_dir, 'w') as f:
        for key in arr_dict.keys():
            f.create_dataset(key, data=arr_dict[key])

Tic_ = np.identity(4)
#print("The Tic_ is: ")
#print(Tic_)

dataset_path_ = "../ECCV_DATASET/7scenes/ICRA_Dataset/stairs/train/"
#save_data_dir = "/mnt/Data/data_generate_3/"
save_data_dir = "../ECCV_DATASET/7scenes/ICRA_Dataset/stairs/train/"
#torch_data_dir = "/mnt/Data/data_generate_3/torch_data/"
torch_data_dir = "../ECCV_DATASET/7scenes/ICRA_Dataset/stairs/train/torch_data/"
if not os.path.exists(torch_data_dir):
    os.mkdir(torch_data_dir)
train_data = LoadDataset(dataset_path_, save_data_dir, transform_, 16, 480, 640, Tic_, camcalib_)
batch_size = 1
train_loader = DataLoader(dataset=train_data, batch_size = batch_size, shuffle=False)

i = 0

for torch_dict in train_loader:
    print("Saving " + str(i) + " th file!")
    file_name = str(i).zfill(4)
    out_file_name = torch_data_dir + file_name + "_torch_data.h5"
    write_hdf5(out_file_name, torch_dict)
    print("Finish !!!")
    print("\n")
    i = i+1
    #break