"""
2024年03月11日
"""
import os, os.path
import cv2.cv2 as cv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def show(img):
    if img.ndim == 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()


def IMfill(img, th, imH, imW):
    img_ori = img
    img_ori_g = cv.cvtColor(img_ori[0: imH, 0: imW, :], cv.COLOR_BGR2GRAY)
    th, img_th = cv.threshold(img_ori_g, th, 255, cv.THRESH_BINARY_INV)
    # Mask
    img_floodfill = img_th.copy()
    h, w = img_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    # Floodfill from point (0,0)
    cv.floodFill(img_floodfill, mask, (0, 0), 255)
    img_floodfill_inv = cv.bitwise_not(img_floodfill)
    img_dst = img_th | img_floodfill_inv
    return img_dst


def cnt_area(cnt):
    area = cv.contourArea(cnt)
    return area

def CntProcess(img, H, W, n):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))  # 开运算卷积核
    img_g = cv.cvtColor(img[0: H, 0: W, :], cv.COLOR_BGR2GRAY)     # 450 裁掉部分干扰区域
    bini = cv.adaptiveThreshold(img_g, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 23, 2)
    th, img_th = cv.threshold(bini, 90, 255, cv.THRESH_BINARY_INV)  #80
    #show(img_th)
    img_th = cv.morphologyEx(img_th, cv.MORPH_OPEN, kernel)  # 开运算 去除噪点
    #show(img_th)

    # Mask
    img_floodfill = img_th.copy()
    h, w = img_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0,0)
    cv.floodFill(img_floodfill, mask, (0, H // 2), 255)
    img_floodfill_inv = cv.bitwise_not(img_floodfill)
    img_dst = img_th | img_floodfill_inv

    #show(img_dst)

    # 获取轮廓并找出面积最大的轮廓
    cnts, hiers = cv.findContours(img_dst, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts.sort(key=cnt_area, reverse=True)             # False 降序排列； True 升序排列
    (x, y, wi, he) = cv.boundingRect(cnts[n])         # 包络矩形顶点位置; n表示找的轮廓面积大小，从0开始是最大
    return img_dst, x, y, wi, he


# 查找文件夹中指定文件的数目
def filenum(img_path):
    num = 0
    count_path = img_path
    files = os.listdir(count_path)
    for file in files:
        if os.path.splitext(file)[1] == '.jpg':
            #  os.path.splitext()是一个元组,类似于('188739', '.jpg')，索引1可以获得文件的扩展名
            num = num + 1
    return num


# 图片合成视频函数
def Img2Video(img_path, video_index, dest_path, play_fps, blankWid):
    i = 0  # initialize frame number for loop
    vfps = 3648
    file_path = img_path
    vid_index = video_index
    target_path = dest_path + vid_index +'video.avi'
    # fourc : Four character code
    fourcc = cv.VideoWriter_fourcc('D', 'I', 'V', 'X')
    fps = play_fps
    frame_num = filenum(file_path) + 1
    t = np.linspace(0, 1/10000*(frame_num-2), (frame_num-2))

    frame0 = cv.imread(frame_path + 'frame_0.jpg')
    size = frame0.shape[0:2]   # 只取长宽，不取色彩通道数
    print(size[0])
    print(size[1])
    video = cv.VideoWriter(target_path, fourcc, fps, (size[1], size[0]))
    # Loop
    xf = 0
    yf = 0
    trajX = []
    trajY= []
    Ds = []           # Spreading Diameter
    We_arr = []           # Weber Number
    for i in range(0, frame_num - 2):
        img_name = file_path + 'frame_' + str(i) + '.jpg'
        if os.path.isfile(img_name):
            #print(i)
            img = cv.imread(img_name)

            # 液滴分析部分

            if i >= 0:
                img_g, xp, yp, width, height = CntProcess(img, size[0]-blankWid, size[1], 0)  # 获取液滴轮廓及包络矩形坐标高宽, blankWid是下方白边的像素数; 最后数字是色块大小排序
                #show(img_g)
                x_cen = int(xp + width / 2)
                y_cen = int(yp + height / 2)
                trajX.append(x_cen)
                trajY.append(y_cen)
                Ds.append(width)
                print([xp, yp, width, height])
                print([xf, yf])
                print((len(t), len(Ds)))
                if i >= 0 & i <= 309:
                    vx = (xp - xf) * pixelLEN * (10 ** (-6)) * vfps  # 速度分量
                    vy = (yp - yf) * pixelLEN * (10 ** (-6)) * vfps  # 速度分量

                    v = (vx ** 2 + vy ** 2) ** 0.5  # 速度幅值
                    print([vx, vy, v])
                    We = rho * (width * pixelLEN * 1e-6) * v ** 2 / sigma
                    We_arr.append(We)

                    cv.rectangle(img, (xp, yp), (xp + width, yp + height), (0, 0, 255),1, cv.LINE_AA)

                    # 文本跟随液滴
                    cv.putText(img, "D:%d" % (width * pixelLEN) + "um", (xp, yp - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    cv.putText(img, "V:%.2f" % v + "m/s", (xp, yp + height + 13), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    cv.putText(img, "We:%.2f" % We, (xp, yp + height + 26), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                    # 文本固定在某个位置
                    #cv.putText(img, "D:%d" % (width * pixelLEN) + "um", (size[1]-180, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    #cv.putText(img, "V:%.2f" % v + "m/s", (size[1]-180, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    #cv.putText(img, "We:%.2f" % We, (size[1]-180, 90), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # 绘制标尺 200 um
                    cv.line(img, (30,60), (30+int(200/pixelLEN), 60), (255, 255, 255), 2, cv.LINE_AA)
                    cv.putText(img, " 200 um", (20,55), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    # 绘制时间戳
                    cv.putText(img, str(round(t[i]*1000, 2))+"ms", (120, 55), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                xf = xp
                yf = yp

            # i = i + 1
            print(i)
            video.write(img)
    video.release()
    plt.plot(t*1000,Ds,marker = 'o')
    print((len(t),len(Ds)))
    plt.show()
    dataframeD = pd.DataFrame({'Time/ms':t,'Diameter/um':Ds})
    dataframeD.to_csv(file_path+"1_D.csv", index = False, sep=',')
    dataframeWe = pd.DataFrame({'Time/ms': t, 'We': We_arr})
    dataframeWe.to_csv(file_path + "1_We.csv", index=False, sep=',')
    #cv.destroyAllWindows()

# 讲输入的视频拆成图片
def video2frames(img_path):
    # 读取视频文件
    video_path = img_path
    cap = cv.VideoCapture(video_path+'1.avi')  # Export_20211022_201215.avi
    # 定义图片保存路径和帧数计数器
    frame_count = 0
    image_folder = video_path+'1/'

    # 检查是否成功打开视频文件
    if not cap.isOpened():
        print("Error opening video file.")
    else:
        # 循环读取每一帧直到没有帧可读
        while cap.isOpened():
            print(frame_count)
            ret, frame = cap.read()
            if not ret:  # 如果读取不成功（视频结束），则退出循环
                break
            # 将当前帧保存为图片
            filename = f"frame_{frame_count}.jpg"
            filepath = os.path.join(image_folder, filename)
            cv.imwrite(filepath, frame)

            # 帧数计数器加1
            frame_count += 1

        # 关闭视频文件
        cap.release()

    print(f"Successfully extracted {frame_count} frames from the video.")

    # 确保目录存在
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)



# ------ main function ---------

# Parameters for Droplet detective
pixelLEN = 24  # 7μm/pixel
rho = 1e3   # kg/m^3
sigma = 0.072   # N/m

# Parameter for video synthesis
frame_path = 'D:/yourfile/'

video_index = '1_show_'  # 欲生成的视频的命名前缀
dest_path = frame_path
play_fps = 15  # 播放帧率
blankWid = 158  # 帧下方参数说明部分厚度。若没有则设为0。
# 这部分处理原始输入是视频而不是帧序列的情形
#video2frames(frame_path)
frame_path = frame_path + '1/'
Img2Video(frame_path, video_index, dest_path, play_fps, blankWid)
