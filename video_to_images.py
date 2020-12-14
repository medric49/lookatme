import cv2

if __name__ == '__main__':

    # Opens the Video file
    cap = cv2.VideoCapture('videos/furniture.mp4')
    i = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        frame = cv2.resize(frame, (231, 231))
        cv2.imwrite('images/furniture/image' + str(i) + '.png', frame)
        i += 1

    cap.release()
    cv2.destroyAllWindows()
