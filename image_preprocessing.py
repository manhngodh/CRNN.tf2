import cv2


def preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return image


if __name__ == '__main__':
    img = cv2.imread("data/manh.png")
    cv2.imwrite('result2.jpg', img)
