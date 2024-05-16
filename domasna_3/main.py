import cv2
import numpy as np
import os
import zipfile

with zipfile.ZipFile('database.zip', 'r') as zip_ref:
    zip_ref.extractall('images')

def segment_and_find_contours(image_path):
    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

    cv2.imshow('Konturi', contour_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


image_files = os.listdir('images')

for image_file in image_files:
    image_path = os.path.join('images', image_file)
    segment_and_find_contours(image_path)

#За точна сегментација, можеби ќе треба да се изведе истовремено користење на повеќе алгоритми или комбинации од нив
#како и да се изведат додатни техники за процесирање, како што се филтрирање на шум или дополнително процесирање на сликата пред сегментацијата