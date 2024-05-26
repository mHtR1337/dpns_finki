import cv2
import os
import zipfile


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), 0)
        if img is not None:
            images.append(img)
    return images


with zipfile.ZipFile('database.zip', 'r') as zip_ref:
    zip_ref.extractall('database')

with zipfile.ZipFile('query_images.zip', 'r') as zip_ref:
    zip_ref.extractall('query_images')

query_images = load_images_from_folder('query_images')
database_images = load_images_from_folder('database')

results = []

for query_img in query_images:
    query_contour, _ = cv2.findContours(query_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    query_contour = query_contour[0]

    similarities = []
    for db_img in database_images:
        db_contour, _ = cv2.findContours(db_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        db_contour = db_contour[0]

        similarity = cv2.matchShapes(query_contour, db_contour, cv2.CONTOURS_MATCH_I1, 0.0)
        similarities.append(similarity)

    results.append(list(enumerate(similarities)))

sorted_results = [sorted(res, key=lambda x: x[1]) for res in results]

for i, query_img in enumerate(query_images):
    print(f"Резултати за слика {i + 1}:")
    for j, (index, similarity) in enumerate(sorted_results[i]):
        print(f"Слика {index + 1}: Сличност - {similarity}")
