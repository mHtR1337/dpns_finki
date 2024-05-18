import cv2
import numpy as np
import os
import zipfile
def extract_zip(zip_path, extract_to='.'):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)


def get_contour(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Ne moze da se procita {image_path}")
        return None

    _, thresholded = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        return contours[0]
    else:
        return None


def compare_shapes(query_image_path, database_path):
    query_contour = get_contour(query_image_path)
    if query_contour is None:
        raise ValueError("Nema pronajdeni konturi")

    results = []
    for image_name in os.listdir(database_path):
        image_path = os.path.join(database_path, image_name)
        if image_path == query_image_path:
            continue

        contour = get_contour(image_path)
        if contour is not None:
            similarity = cv2.matchShapes(query_contour, contour, cv2.CONTOURS_MATCH_I1, 0.0)
            results.append((image_name, similarity))
        else:
            print(f"Warning: Nema pronajdeni konturi {image_name}")

    results.sort(key=lambda x: x[1])
    return results


def main():
    zip_path = 'database.zip'
    extract_path = 'database'

    extract_zip(zip_path, extract_path)

    print("sodrzina na datotekata:")
    for root, dirs, files in os.walk(extract_path):
        for file in files:
            print(os.path.join(root, file))

    query_image = os.path.join(extract_path, 'query_image.png')
    if not os.path.exists(query_image):
        raise FileNotFoundError(f"Nema query slika: {query_image}")

    results = compare_shapes(query_image, extract_path)

    print("slicnosti:")
    for result in results:
        print(f"Slika: {result[0]}, Sklicnost: {result[1]}")


if __name__ == "__main__":
    main()
