import cv2
import numpy as np
import os

def load_images_from_directory(directory):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(os.path.join(directory, filename))
            if img is not None:
                images.append((filename, img))
    return images


def compute_sift_descriptors(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors


def match_descriptors(desc1, desc2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return good_matches


def find_homography_and_inliers(kp1, kp2, matches):
    if len(matches) >= 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matches_inliers = [m for m, inlier in zip(matches, mask.ravel()) if inlier]
        return M, matches_inliers
    return None, []


def draw_matches(img1, kp1, img2, kp2, matches, title):
    result = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow(title, result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    database_images = load_images_from_directory('Database')
    query_images = [('hw7_poster_1.jpg', cv2.imread('hw7_poster_1.jpg')),
                    ('hw7_poster_2.jpg', cv2.imread('hw7_poster_2.jpg')),
                    ('hw7_poster_3.jpg', cv2.imread('hw7_poster_3.jpg'))]

    for query_name, query_img in query_images:
        query_kp, query_desc = compute_sift_descriptors(query_img)

        best_match = None
        best_inliers = 0
        best_img = None
        best_kp = None
        best_matches = None

        for db_name, db_img in database_images:
            db_kp, db_desc = compute_sift_descriptors(db_img)
            matches = match_descriptors(query_desc, db_desc)
            M, inliers = find_homography_and_inliers(query_kp, db_kp, matches)
            M, inliers = find_homography_and_inliers(query_kp, db_kp, matches)

            if len(inliers) > best_inliers:
                best_inliers = len(inliers)
                best_match = db_name
                best_img = db_img
                best_kp = db_kp
                best_matches = inliers

        if best_img is not None:
            print(f"Best match for {query_name}: {best_match} with {best_inliers} inliers")

            combined_img = np.hstack((query_img, best_img))
            cv2.imshow('Query and Best Match Image', combined_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            draw_matches(query_img, query_kp, best_img, best_kp, best_matches, 'SIFT Descriptors Matched')


if __name__ == "__main__":
    main()