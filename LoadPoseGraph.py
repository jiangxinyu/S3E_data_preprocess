import numpy as np
import os
import cv2
from collections import defaultdict

from scipy.spatial.transform import Rotation as R

#camera projection parameters:
fx = 616.591125488
fy = 616.796264648
cx = 324.219360351
cy = 239.427017211

def load_pose_graph(pos_graph_path):
    pose_data_list = []

    with open(pos_graph_path) as f:
        for line in f.readlines():
            single_pose_data = {}
            temp_line = line.split()
            single_pose_data['timestamp'] = float(temp_line[0])
            single_pose_data['t_x'] = float(temp_line[1])
            single_pose_data['t_y'] = float(temp_line[2])
            single_pose_data['t_z'] = float(temp_line[3])
            single_pose_data['q_x'] = float(temp_line[4])
            single_pose_data['q_y'] = float(temp_line[5])
            single_pose_data['q_z'] = float(temp_line[6])
            single_pose_data['q_w'] = float(temp_line[7])
            pose_data_list.append(single_pose_data)
        print("Read pose graph txt finished! ")
        print("The size of of the pose_data_list is: " + str(len(pose_data_list)))

        return pose_data_list

def load_vins_pose_graph(vins_pose_path):
    vins_pose_list = []
    with open(vins_pose_path) as f:
        for line in f.readlines():
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
            vins_pose_list.append(single_pose_data)
    return vins_pose_list

def load_img_data(rgb_image_path, mode = "image"):
    img_list = []
    for i in range(len(os.listdir(rgb_image_path))):
        if mode == "image":
            image = cv2.imread(rgb_image_path+ str(i) + "_" + mode + ".png")
        else:
            image = cv2.imread(rgb_image_path + str(i) + "_" + mode + ".png", -1)
            #print("The shape of the image is: ")
            #print(image.shape)
            #image = cv2.split(image_depth)[0]
        img_list.append(image)
    print("Read img file finished !")
    print("The size of the img file is: " + str(len(img_list)) + " pices")
    #cv2.imshow("read_image", image)
    #cv2.waitKey(0)

    return img_list


def load_d2pts_data(d2_points_path, img_width, img_height):
    key_points_dict = defaultdict(list)
    brief_list = []
    for i in range(len(os.listdir(d2_points_path)) // 2):
        keypoints_file_name = d2_points_path + str(i) + "_keypoints.txt"
        with open(keypoints_file_name) as f:
            key_points_uv = []
            key_points_xy = []
            #print("The number of key points: ")
            #print(len(f.readlines()))
            for line in f.readlines():
                key_point_uv = np.ones((3, 1))
                key_point_xy = np.ones((4, 1))
                temp_line = line.split()
                key_point_uv[0] = float(temp_line[0])
                key_point_uv[1] = float(temp_line[1])
                key_point_xy[0] = float(temp_line[2])
                key_point_xy[1] = float(temp_line[3])
                if int(key_point_uv[0]) == int(img_width/2) or int(key_point_uv[1]) == int(img_height/2):
                    continue
                key_points_uv.append(key_point_uv)
                key_points_xy.append(key_point_xy)

            key_points_dict["uv"].append(key_points_uv)
            key_points_dict["xy"].append(key_points_xy)
    return key_points_dict




def load_d3pts_data(d3_points_path):
    d3_all_list = []
    for i in range(len(os.listdir(d3_points_path))):
        pts_file_name = d3_points_path + str(i) + "_3dpoints" + ".txt"
        line_num = 0
        with open(pts_file_name) as f:
            d3_single_list = []
            for line in f.readlines():
                points = np.ones((4, 1))
                temp_line = line.split()
                points[0] = float(temp_line[0])
                points[1] = float(temp_line[1])
                points[2] = float(temp_line[2])
                d3_single_list.append(points)
                line_num += 1
            #print("The number of line is: " + str(line_num))
            #print("The number of poitns is: " + str(len(d3_single_list)))
        d3_all_list.append(d3_single_list)

    print("Read 3d pts file finished!")
    print("The size of the key points is: " + str(len(d3_all_list)) + " images points")
    return d3_all_list

def project_and_imgshow(rgb_images_list, key_points_list, fx, fy, cx, cy):
    img_count = 0
    for image in rgb_images_list:
        img_key_points = key_points_list[img_count]
        print(len(img_key_points))
        for key_point in img_key_points:
            u_coor = int(fx * key_point[0]/key_point[2] + cx)
            v_coor = int(fy * key_point[1]/key_point[2] + cy)
            cv2.circle(image, (u_coor, v_coor), 1, (0, 0, 255), 4)
        print("Draw points has been finished!")
        img_count += 1
        cv2.imshow("image", image)
        cv2.waitKey(0)

#pose_data_list = load_pose_graph()
#img_data_list = load_img_data()
#d3pts_data_list = load_d3pts_data()
#print(len(d3pts_data_list[0]))
#project_and_imgshow(img_data_list, d3pts_data_list, fx, fy, cx, cy)
#vins_pose_list = load_vins_pose_graph(vins_pose_path)
#print(vins_pose_list)

#key_points_dict = load_d2pts_data(d2_points_path_)
#print("The length of uv: ")
#print(len(key_points_dict["uv"]))
#print("The length of xy: ")
#print(len(key_points_dict["xy"]))
#depth_images = load_img_data(depth_image_path_, "depth")
#depth_image = depth_images[0]
#print("The shape of depth image is: ")
#print(depth_image.shape)
#print(depth_image[])
#print("The length of the depth image: ")
#print(len(depth_images))