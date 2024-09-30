import cv2
import sys
import numpy as np
from helpers import *

def wrapper(chess_side, chess_dimensions):

    #load images
    calib_images = LoadImagesFromFolder("calib_images")
    h,w = calib_images[0].shape[:2]

    #set obj points(chess coordinates in 2d board coordinate frame)
    obj_points = fill_objpoints(chess_side, chesslength=chess_dimensions[0], chessheight=chess_dimensions[1]) 
    obj_points = [np.array(obj_pts, dtype=np.float32) for obj_pts in obj_points]

    
    imgpoints_images = []
    objpoints_images = []

    #fill obj points and image points(chess corners in image frame)
    for image in calib_images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        img_points = find_chesscorners(gray, (9,6))
        img_points = img_points
        img_points = [np.array(img_pts, dtype=np.float32) for img_pts in img_points]
        imgpoints_images.append(img_points)
        objpoints_images.append(obj_points)

    imgpoints_images = np.array(imgpoints_images)
    objpoints_images = np.array(objpoints_images)

    #get transformation between chacker board , camera.
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints_images, imgpoints_images, gray.shape[::-1], None, None)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    checker_transformations = []
    for rvec, tvec in zip(rvecs, tvecs):
        
        T = np.eye(4)
        R, _ = cv2.Rodrigues(rvec)
        T[:3, :3] = R
        T[:3, 3] = tvec.reshape(-1)
        checker_transformations.append(T)

    #read Robot's flange transformations 
    csv_data = read_csv_file('robot_poses/robot_data_log.csv')
    transformations_EB = getTFs(csv_data)

    #initialize camera calibration matrix
    T_CE = None

    if T_CE is None:
        T_CE = camera_calibration(checker_transformations, transformations_EB)
    T_CE = T_CE/T_CE[3,3] #homogenizing the matrix

    #camera w.r.t to robot's base in each frame.
    transformations_CB = []
    for transformation in transformations_EB:
        T_CB = transformation@T_CE
        transformations_CB.append(T_CB)
        
    print(f"camera with respect to end effector: {T_CE}")
    print(f"End effector with respect to camera: {T_inv(T_CE)}")
        
    final_residual = residual_calculation(checker_transformations, transformations_EB, T_CE)
    print(f"final Residual: {final_residual}")

    visualize_camera_poses(transformations_CB)


def main():
    # Default values
    default_chess_side = 22.5  # Default chess side value in mm
    default_dimension_x = 10    # Default dimension x value
    default_dimension_y = 7    # Default dimension y value

    # Check command-line arguments
    if len(sys.argv) > 4:
        print("Usage: python wrapper.py [<chess_side>] [<dimension_x>] [<dimension_y>]")
        return

    # Set default values or use command-line arguments
    chess_side = default_chess_side if len(sys.argv) < 2 else float(sys.argv[1])
    dimension_x = default_dimension_x if len(sys.argv) < 3 else int(sys.argv[2])
    dimension_y = default_dimension_y if len(sys.argv) < 4 else int(sys.argv[3])

    print(f"Chess Side: {chess_side} mm")
    print(f"Dimension X: {dimension_x}")
    print(f"Dimension Y: {dimension_y}")

    chess_side_mts = chess_side / 1000  # Convert mm to meters
    dimensions = [dimension_x, dimension_y]

    wrapper(chess_side=chess_side_mts, chess_dimensions=dimensions)

if __name__ == "__main__":
    main()