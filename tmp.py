import cv2
for i in range(10000):
    n = f'{i}'.zfill(5)
    image = 'images/faces_128/'+n+'.png'
    image = cv2.imread(image)
    image = cv2.resize(image, (231, 231))
    cv2.imwrite(f'images/faces/{i}.png', image)
