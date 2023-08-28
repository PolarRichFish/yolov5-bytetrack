import cv2


img = cv2.imread('images/0.jpg')
print(img)
print(img.shape)  # (1080, 1920, 3)
"""[::-1]：这是一个切片操作，它会将图像的各个维度都反转。在这个上下文中
，它会导致颜色通道的顺序从原来的 (channels, height, width) 变为 (width, height, channels)。"""
img = img.transpose((2, 0, 1))[::-1]

print(img)
print(img.shape) # (3, 1080, 1920)



