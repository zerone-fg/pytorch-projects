import cv2
import matplotlib.pyplot as plt


def trans(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def show_image_with_box(img, boxes, point_color=(0, 0, 255), thickness=2, line_type=8):
    img = trans(img)
    for box in boxes:
        v1 = (box[0], box[1])
        v2 = (box[2], box[3])
        cv2.rectangle(img, v1, v2, color=point_color, thickness=thickness, lineType=line_type)
    plt.imshow(img)
    plt.show()


def show_image(path):
    img = cv2.imread(path)
    img = trans(img)
    cv2.imshow("read", img)