import numpy as np
from LoadPoseGraph import load_pose_graph, load_img_data, load_d3pts_data, load_vins_pose_graph, load_d2pts_data
from scipy.spatial.transform import Rotation as R
import cv2

#pos_graph_path = "../output/pose_graph/pose/stamped_traj_estimate_mono_pg.txt"
#rgb_image_path = "../output/pose_graph/color_img/"
#depth_image_path = "../output/pose_graph/depth/"
#d3_points_path = "../output/pose_graph/3d_points/"
#vins_pose_path = "../output/pose_graph/vins_result_loop.txt"
#d2_points_path = "../output/pose_graph/key_points/"

#fx = 370.9210933467814
#fy = 371.5067460407453
#cx = 315.6742570531217
#cy = 242.54830641769718

def test_numpy():
    points_ = [1,  1, 1,1]
    T_matrix = np.ones((4, 4)) * 0.1
    R_matrix = np.ones((3, 3)) * 0.5
    #print("The T_matrix before give value: ")
    #print(T_matrix)
    T_matrix[0:3, 0:3]= R_matrix
    T_matrix[::, 3] = points_
    print("The T_matrix before after value: ")
    print(T_matrix)

def rotate_trans_merge(pose_data_list):
    pose_list = []

    Ric = np.array([0.99986953, 0.0136529,
       0.00863274,  -0.0136466, 0.99990657,
       -0.00078815, -0.00864269, 0.00067024,
       0.99996243]).reshape(3,3)
    tic = [0.02914287, 0.00274512, -0.00492899, 1]
    Tic = np.identity(4)
    Tic[0:3, 0:3] = Ric
    Tic[::, 3] = tic
    for pose in pose_data_list:
        T_matrix = np.identity(4)
        Rq = [pose["q_x"], pose["q_y"], pose["q_z"], pose["q_w"]]
        Rm = R.from_quat(Rq)
        trans = [pose["t_x"], pose["t_y"], pose["t_z"], 1]
        R_matrix = Rm.as_matrix()
        T_matrix[0:3, 0:3] = R_matrix
        T_matrix[::, 3] = trans
        T_wc = np.matmul(T_matrix, Tic)
        #print("The R_matrix is: ")
        #print(R_matrix)
        #print("\n")
        #print("The T_matrix is: ")
        #print(T_matrix)
        #print("\n")
        pose_list.append(T_wc)
    return pose_list

def rgb_points_trans(pose_list, img_list, depth_img_list, d2_pts_list, fx, fy, cx, cy):
    for i in range(len(img_list) - 1):
        img_ref = img_list[i].copy()
        img_cur = img_list[i+1].copy()
        T_w_ref = pose_list[i]
        T_w_cur = pose_list[i+1]
        T_cur_ref = np.matmul(np.linalg.inv(T_w_cur), T_w_ref)
        T_test = np.matmul(np.linalg.inv(T_w_cur), T_w_cur)
        #print("The T_test is: ")
        #print(T_test)
        depth_img_ref = depth_img_list[i]
        depth_img_cur = depth_img_list[i+1]
        points_ref_uv = d2_pts_list["uv"][i]
        points_ref_xy = d2_pts_list["xy"][i]
        for j in range(len(points_ref_uv)):
            point_uv = points_ref_uv[j]
            #print("The point uv is: ")
            #print(points_ref_uv)
            point_depth = depth_img_ref[int(point_uv[1]), int(point_uv[0])]/1000.000
            #print("The point depth is: ")
            #print(point_depth)
            #print("The point depth is: ")
            #print(point_depth)
            point_xy = np.ones((4, 1))
            point_xy[0] = ((point_uv[0] - cx) * point_depth) / fx
            point_xy[1] = ((point_uv[1] - cy) * point_depth) / fy
            point_xy[2] = point_depth
            #print("The point xy is: ")
            #print(point_xy)
            point_proj = np.dot(T_cur_ref, point_xy)
            u_proj = int(fx * (point_proj[0] / point_proj[2]) + cx)
            v_proj = int(fy * (point_proj[1] / point_proj[2]) + cy)

            if u_proj > img_cur.shape[1] or u_proj < 0 or v_proj > img_cur.shape[0] or v_proj < 0:
                continue
            cv2.circle(img_ref, (int(point_uv[0]), int(point_uv[1])), 1, (0, 255, 0), 4)
            cv2.circle(img_cur, (u_proj, v_proj), 1, (0, 0, 255), 4)

        imgs = np.hstack([img_ref, img_cur])
        #cv2.imshow("multi_img", imgs)
        #cv2.waitKey(0)

def image_points_trans(pose_list, points_list, img_list, fx, fy, cx, cy):
    for i in range(len(img_list)-1):
        img_ref = img_list[i].copy()
        img_cur = img_list[i + 1].copy()
        T_w_ref = pose_list[i] # T_w_c1
        T_w_cur = pose_list[i+1] # T_w_c2
        T_cur_ref = np.matmul(np.linalg.inv(T_w_cur), T_w_ref) # T_w_c2 ^-1 * T_w_c1 = T_c2_c1

        points_ref = points_list[i]
        for point in points_ref:
            point_project = np.dot(T_cur_ref, point)
            u_coor_proj = int(fx * point_project[0]/point_project[2] + cx)
            v_coor_proj = int(fy * point_project[1]/point_project[2] + cy)
            if u_coor_proj > img_cur.shape[1] or u_coor_proj < 0 or v_coor_proj > img_cur.shape[0] or v_coor_proj < 0:
                continue
            u_coor = int(fx * point[0]/point[2] + cx)
            v_coor = int(fx * point[1]/point[2] + cy)

            cv2.circle(img_ref, (u_coor, v_coor), 1, (0, 255, 0), 4)
            cv2.circle(img_cur, (u_coor_proj, v_coor_proj), 1, (0, 0, 255), 4)

        imgs = np.hstack([img_ref, img_cur])
        cv2.imshow("multi_img", imgs)
        cv2.waitKey(0)

def image_points_trans_map_point(pose_list, points_list, img_list, fx, fy, cx, cy):
    for i in range(len(img_list)-1):
        img_ref = img_list[i].copy()
        img_cur = img_list[i + 1].copy()
        T_w_ref = pose_list[i] # T_w_c1
        T_w_cur = pose_list[i+1] # T_w_c2
        #T1_points = np.linalg.inv(T_w_ref)

        points_ref = points_list[i]
        point_count = 0
        for point in points_ref:
            point_project = np.dot(T_w_ref, point)
            if point_project[2] == 0 or point[2] == 0:
                continue
            point_count = point_count + 1

            u_coor_proj = int(fx * point_project[0]/point_project[2] + cx)
            v_coor_proj = int(fy * point_project[1]/point_project[2] + cy)
            print("u_coor_proj is:")
            print(u_coor_proj)
            print("v_coor_proj is:")
            print(v_coor_proj)
            print("\n")
            #if u_coor_proj > img_cur.shape[1] or u_coor_proj < 0 or v_coor_proj > img_cur.shape[0] or v_coor_proj < 0:
            #    continue
            #u_coor = int(fx * point[0]/point[2] + cx)
            #v_coor = int(fx * point[1]/point[2] + cy)

            cv2.circle(img_ref, (u_coor_proj, v_coor_proj), 1, (0, 255, 0), 4)
            #cv2.circle(img_cur, (u_coor_proj, v_coor_proj), 1, (0, 0, 255), 4)

        imgs = np.hstack([img_ref, img_cur])
        print("The point count is: ")
        print(point_count)
        cv2.imshow("multi_img", imgs)
        cv2.waitKey(0)

#pose_data_list = load_vins_pose_graph(vins_pose_path)
#pose_list = rotate_trans_merge(pose_data_list)
#print("The pose_list is: ")
#print(pose_list)
#img_data_list = load_img_data(rgb_image_path, mode= "image")
#depth_data_list = load_img_data(depth_image_path, mode = "depth")
#print(depth_data_list)
#d3pts_data_list = load_d3pts_data(d3_points_path)
#key_points_dict = load_d2pts_data(d2_points_path, img_width=640, img_height=480)
#print(len(key_points_dict["uv"][0]))
#rgb_points_trans(pose_list, img_data_list, depth_data_list, key_points_dict, fx, fy, cx, cy)
#image_points_trans(pose_list, d3pts_data_list, img_data_list, fx, fy, cx, cy)