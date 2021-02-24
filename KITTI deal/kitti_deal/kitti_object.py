''' Helper class and functions for loading KITTI objects

Author: Charles R. Qi

Date: September 2017

'''
from __future__ import print_function

import os

import sys
import time
import numpy as np
import kdtree as KDT
import cv2
import math
from PIL import Image
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ROOT_DIR = os.path.dirname(BASE_DIR)

sys.path.append(os.path.join(ROOT_DIR, 'mayavi'))

import kitti_utils as utils
raw_input = input  # Python 3
max_depth=0
class kitti_object(object):
    '''Load and parse object data into a usable format.'''

    def __init__(self, root_dir, split='training'):

        '''root_dir contains training and testing folders'''

        self.root_dir = root_dir

        self.split = split

        self.split_dir = os.path.join(root_dir, split)

        if split == 'training':

            self.num_samples = 7481

        elif split == 'testing':

            self.num_samples = 7518

        else:

            print('Unknown split: %s' % (split))

            exit(-1)

        self.image_dir = "E:/KITTI/Dataset/training/image_2"

        self.calib_dir = "E:/KITTI/Dataset/training/calib"

        self.lidar_dir ="E:/KITTI/Dataset/training/velodyne"

        self.label_dir = "E:/KITTI/Dataset/training/label_2"

    def __len__(self):

        return self.num_samples

    def get_image(self, idx):

        assert (idx < self.num_samples)

        img_filename = os.path.join(self.image_dir, '%06d.png' % (idx))

        return utils.load_image(img_filename)

    def get_lidar(self, idx):

        assert (idx < self.num_samples)

        lidar_filename = os.path.join(self.lidar_dir, '%06d.bin' % (idx))

        return utils.load_velo_scan(lidar_filename)

    def get_calibration(self, idx):

        assert (idx < self.num_samples)

        calib_filename = os.path.join(self.calib_dir, '%06d.txt' % (idx))

        return utils.Calibration(calib_filename)

    def get_label_objects(self, idx):

        assert (idx < self.num_samples and self.split == 'training')

        label_filename = os.path.join(self.label_dir, '%06d.txt' % (idx))

        return utils.read_label(label_filename)

    def get_depth_map(self, idx):

        pass

    def get_top_down(self, idx):

        pass
def get_lidar_in_image_fov(pc_velo, calib, xmin, ymin, xmax, ymax,
                           return_more=False, clip_distance=2.0):
    ''' Filter lidar points, keep those in image FOV '''

    pts_2d = calib.project_velo_to_image(pc_velo)

    fov_inds = (pts_2d[:, 0] < xmax) & (pts_2d[:, 0] >= xmin) & \
 \
               (pts_2d[:, 1] < ymax) & (pts_2d[:, 1] >= ymin)

    fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance)

    imgfov_pc_velo = pc_velo[fov_inds, :]

    if return_more:

      return imgfov_pc_velo, pts_2d, fov_inds

    else:

      return imgfov_pc_velo
def looknearest(num_list,i,j):
    sum=0
    size=0
    if i>0:
        if num_list[i-1][j]!=0:
            sum+=num_list[i-1][j]
            size+=1
    if i<num_list.shape[0]-1:
        if num_list[i+1][j]!=0:
            sum+=num_list[i+1][j]
            size+=1
    if j>0:
        if num_list[i][j-1]!=0:
          sum+=num_list[i][j-1]
          size+=1
    if j<num_list.shape[1]-1:
        if num_list[i][j+1]!=0:
          sum+=num_list[i][j+1]
          size+=1
    if i>0 and j>0:
        if num_list[i-1][j-1]!=0:
            sum+=num_list[i-1][j-1]
            size+=1
    if i>0 and j<num_list.shape[1]-1:
        if num_list[i-1][j+1]!=0:
            sum+=num_list[i-1][j+1]
            size+=1
    if i<num_list.shape[0]-1 and j>0:
        if num_list[i + 1][j - 1] != 0:
            sum += num_list[i + 1][j - 1]
            size += 1
    if i < num_list.shape[0] - 1 and j <num_list.shape[1]-1:
        if num_list[i + 1][j +1] != 0:
            sum += num_list[i + 1][j +1]
            size += 1
    if size!=0:
        return sum/size
    return 0
def generate_coor(x_min, x_max, y_min, y_max):
    # generate 2D coor
    x = range(x_min, x_max)
    y = range(y_min, y_max)
    X, Y = np.meshgrid(x, y) # 2D grid
    X, Y = X.flatten(), Y.flatten()
    coordinate = [[x,y] for x, y in zip(X, Y)]
    return (coordinate, x_min, x_max, y_min, y_max)
def KNN1(img_gray):
    img_gray1=img_gray
    kd_block = []
    start1=time.clock()
    print("开始生成块"+str(start1))
    for x_1 in range(0, 5):
        for y_1 in range(0, 6):
            y_min = y_1 * 206 - 15
            x_min = x_1 * 54 + 100 - 15
            y_max = y_min + 206 + 15
            x_max = x_min + 54 + 15
            # 有值点的坐标
            kd_block.append(generate_coor(x_min, x_max, y_min, y_max))
    start2=time.clock()
    print("30个块生成完了"+str(start2))
    for block in kd_block:
      coordinate, x_min, x_max, y_min, y_max = block
      x = range(x_min, x_max)
      y = range(y_min, y_max)
      X, Y = np.meshgrid(x, y)  # 2D grid
      X, Y = X.flatten(), Y.flatten()
      value_coordinate = [[x, y] for x, y in zip(X, Y) if
                                x > 0 and x < 375 and y > 0 and y < 1242 and img_gray[x, y] > 0]
      if not value_coordinate:
        continue
            # 构造KDTree
      KNN = KDT.create(value_coordinate)
            # 存放最近三个点的坐标与权重的字典
      dictionary = {}
      for x, y in coordinate:
                # 找到最近的四个点，第一个最近的点是该点本身
        if [x, y] in value_coordinate:
            _, a1, b1, c1 = KNN.search_knn([x, y], 4)
        else:
             a1, b1, c1 = KNN.search_knn([x, y], 3)
                    # 获取最近的三个点的坐标
             a = a1[0].data
             b = b1[0].data
             c = c1[0].data
                    # 获取最近的三个点距离当前点的距离
             da = a1[1]
             db = b1[1]
             dc = c1[1]
                    # 计算权重
             d_s = da + db + dc
             wa, wb, wc = da / d_s, db / d_s, dc / d_s
                    # 将最近三个点的坐标与权重存到字典中
             dictionary[(x, y)] = (a, b, c, wa, wb, wc)
            # 补全点云强度图的空缺点
      for i, j in coordinate:
          if i > 0 and i < 375 and j > 0 and j < 1242:
            if img_gray[i, j] == 0:
              a, b, c, da_, db_, dc_ = dictionary[(i, j)]
              A = img_gray[tuple(a)]
              B = img_gray[tuple(b)]
              C = img_gray[tuple(c)]
              img_gray1[i, j] = da_ * A + db_ * B + dc_ * C  # ABC是三个坐标上点云的强度值
      start3=time.clock()
      print("块生成完了"+str(start3))
    return img_gray1
def show_lidar_on_image(pc_velo, img, calib, img_width, img_height):

    ''' Project LiDAR points to image '''

    imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(pc_velo,

        calib, 0, 0, img_width, img_height, True)

    imgfov_pts_2d = pts_2d[fov_inds,:]

    imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)
    import matplotlib.pyplot as plt
    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
    num_list = np.zeros((img_height, img_width))   #img_height为行数，img_width为列数
    num_list_temp = np.zeros((img_height, img_width))
    for i in range(imgfov_pts_2d.shape[0]):
        depth = imgfov_pc_rect[i,2]
        if int(np.round(imgfov_pts_2d[i,0]))<img_width and int(np.round(imgfov_pts_2d[i,1]))<img_height:
          #max_depth=79.71860667884495
          #num_list[int(np.round(imgfov_pts_2d[i,1]))][int(np.round(imgfov_pts_2d[i,0]))]=depth/max_depth*256
          num_list[int(np.round(imgfov_pts_2d[i, 1]))][int(np.round(imgfov_pts_2d[i, 0]))] =depth
          #num_list_temp[int(np.round(imgfov_pts_2d[i, 1]))][int(np.round(imgfov_pts_2d[i, 0]))] = depth
    num_list=KNN1(num_list)
    for i in range(img_height):
        for j in range(img_width):
            num_list[i,j]=num_list[i,j]*3
    #cv2.imshow("result",num_list)
    #cv2.waitKey(-1)                          #把展示的代码改成存储的代码，cv.imwrite
    #cv2.imwrite("F:/000000.png",num_list)
    '''
    for i in range(img_height):
        for j in range(img_width):
            if num_list[i][j] != 0:
                temp = int(640.0 / num_list[i][j])
                if temp >= 256:
                    temp = 255
                color = cmap[temp, :]
                cv2.circle(img, (j, i), 2, color=tuple(color), thickness=-1)
    '''
    '''
    for i in range(img_height):
        for j in range(img_width):
            if num_list_temp[i][j]==0:
               num_list[i][j]=looknearest(num_list_temp,i,j)
               #num_list[i][j]=KNN1(num_list_temp)
            if num_list[i][j]!=0:
               temp = int(640.0 /num_list[i][j])
               if temp >= 256:
                 temp = 255
               color = cmap[temp, :]
               cv2.circle(img,(j,i),2, color=tuple(color), thickness=-1)
    #im = Image.fromarray(num_list)
    #im = im.convert('L')  # 这样才能转为灰度图，如果是彩色图则改L为‘RGB’
    #img.show("outsave")
    #im.save('outfile.png')
        #cv2.circle(img, (int(np.round(imgfov_pts_2d[i,0])),
            #int(np.round(imgfov_pts_2d[i,1]))),
            #2, color=tuple(color), thickness=-1)
    '''
    return num_list
def dataset_viz():
    dataset = kitti_object("E:/KITTI/Dataset/")   #改成你的KITTI的dataset路径
    for data_idx in range(len(dataset)):
    # Load data from dataset
        if data_idx<3000:
            continue
        objects = dataset.get_label_objects(data_idx)
        objects[0].print_object()
        img = dataset.get_image(data_idx)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_height, img_width, img_channel = img.shape
        print(('Image shape: ', img.shape))
        pc_velo = dataset.get_lidar(data_idx)[:, 0:3]
        calib = dataset.get_calibration(data_idx)
        gray0 = np.zeros((img_height,img_width), dtype=np.uint8)
        # Show all LiDAR points. Draw 3d box in LiDAR point cloud
        gray0=cv2.cvtColor(gray0, cv2.COLOR_BGR2RGB)
        img=show_lidar_on_image(pc_velo,gray0,calib, img_width, img_height)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #cv2.imwrite("F:/result11.jpg",img)
        #cv2.imshow("result",img)
        #cv2.waitKey(-1)
        cv2.imwrite("F:/instance/"+str(data_idx)+".png",img)
        print(str(data_idx)+"finished")
if __name__ == '__main__':
    import mayavi.mlab as mlab
    dataset_viz()