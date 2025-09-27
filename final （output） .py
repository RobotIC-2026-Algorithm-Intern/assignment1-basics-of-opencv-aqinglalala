import cv2
import numpy as np
import matplotlib.pyplot as plt


class ColorRange:
    """管理不同颜色的HSV阈值范围"""
    def __init__(self):
        # 红色有两个区间
        self.ranges = {
            'red': [
                (np.array([0, 0, 214]), np.array([45, 255, 255])),
                (np.array([160, 100, 100]), np.array([179, 255, 255]))
            ],
            'purple': [
                (np.array([130, 0, 139]), np.array([180, 133, 255]))
            ]
        }

    def get_masks(self, hsv_img, color_name):
        if color_name not in self.ranges:
            raise ValueError(f"未定义颜色: {color_name}")
        masks = [cv2.inRange(hsv_img, lower, upper) for lower, upper in self.ranges[color_name]]
        mask = masks[0]
        for m in masks[1:]:
            mask = cv2.bitwise_or(mask, m)
        return mask
    


class BallDetector:
    def preprocess_frame(self, frame):
        # 高斯模糊
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        # 转换为HSV色彩空间
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        return hsv
    
    def __init__(self, video_path):
        self.video_path = video_path
        self.color_range = ColorRange()

    def read_video_frames(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error: 无法打开视频文件")
            return

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("已到达视频末尾或读取失败")
                break
            frame_count += 1
            print(f"正在处理第 {frame_count} 帧")
            # 只取感兴趣区域
            roi = frame[150:400, 200:400, :]
            hsv = self.preprocess_frame(roi)
            # 获取红色和紫色掩码
            red_mask = self.color_range.get_masks(hsv, 'red')
            purple_mask = self.color_range.get_masks(hsv, 'purple')
            # 形态学处理
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
            purple_mask = cv2.morphologyEx(purple_mask, cv2.MORPH_OPEN, kernel)
            
            # 判断是否有红球
            red_pixels = cv2.countNonZero(red_mask)
            # 判断是否有紫球
            purple_pixels = cv2.countNonZero(purple_mask)

            if red_pixels > 9000:
                text = "red ball detected"
                color = (0, 0, 255)  # 红色
            elif purple_pixels > 10000:
                text = "purple ball detected"
                color = (255, 0, 255)  # 紫色
            else:
                text = "no ball"
                color = (255, 255, 255)  # 白色

            # 在原始frame左上角标注文字
            cv2.putText(
                frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, color, 2, cv2.LINE_AA
            )

            # 可选：画出ROI区域
            cv2.rectangle(frame, (200, 150), (400, 400), (0, 255, 0), 2)

            # 显示原始视频帧
            cv2.imshow('Video Frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = 'res/output.avi'  # 替换为你的AVI视频路径
    detector = BallDetector(video_path)
    detector.read_video_frames()
