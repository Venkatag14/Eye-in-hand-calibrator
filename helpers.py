import cv2
import numpy as np
import os
import csv
import random
import open3d as o3d
from scipy.optimize import minimize
from tqdm import tqdm



def read_csv_file(file_path):
    data = []
    with open(file_path, 'r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    return data

def LoadImagesFromFolder(folder):
    images = []
    # Get a sorted list of files in the folder
    file_names = sorted(os.listdir(folder))
    # Load each image in alphabetical order
    for file in file_names:
        # Construct the full file path
        file_path = os.path.join(folder, file)    
        # Load the image
        tmp = cv2.imread(file_path)   
        # If the image was successfully loaded, add it to the list
        if tmp is not None:
            images.append(tmp)
    return images

def fill_objpoints(side, chesslength, chessheight):
    obj_points = []
    for i in range(1,chessheight):
        for j in range(1,chesslength):
            obj_points.append([side*i, side*j, 0])
            
    obj_points = np.array(obj_points)
    return obj_points

def find_chesscorners(image, dimensions):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    imgpoints = []
    ret, corners = cv2.findChessboardCorners(image, (9,6), None) #find chess corners  
    if ret:
        corners2 = cv2.cornerSubPix(image,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
    
    imgpoints = np.array(imgpoints).reshape(-1,2)
    return imgpoints

def calculate_reprojection_error(self, params, R, T, image_points, world_points):
    
    alpha, gamma, beta, u0, v0, k1, k2 = params
    K = np.array([[alpha, gamma, u0], [0, beta, v0], [0, 0, 1]])
    error = []
    reprojected_corners = []
    for i in range(len(image_points)):
        img_corners = image_points[i]
        RT = np.vstack((R[i], T[i]))
        RT = RT.T
        H = np.dot(K, RT)
        temp_error = 0
        temp_reprojected_corners = []
        for j in range(image_points[i].shape[0]):
            world_point = world_points[j]
            world_point = np.append(world_point, 1)
            world_point = world_point.reshape(-1, 1)
            world_point = world_point.T
            reprojected_point = np.matmul(RT, world_point.T)
            reprojected_point = reprojected_point / reprojected_point[2]
            corner_point_orig = img_corners[j]
            corner_point_orig = np.array(
                [corner_point_orig[0, 0], corner_point_orig[0, 1], 1]
            )
            corner_point = np.matmul(H, world_point.T)
            corner_point = corner_point / corner_point[2]

            x = reprojected_point[0]
            y = reprojected_point[1]
            u = corner_point[0]
            v = corner_point[1]

            r = np.square(x) + np.square(y)
            u_hat = u + (u - u0) * (k1 * r + k2 * np.square(r))
            v_hat = v + (v - v0) * (k1 * r + k2 * np.square(r))

            corner_hat = np.array([u_hat, v_hat, 1], dtype=np.float32)
            temp_reprojected_corners.append(
                np.array((corner_hat[0], corner_hat[1]))
            )

            temp_error += np.linalg.norm((corner_point_orig - corner_hat), ord=2)
            # corner_hat.astype(np.float32)
        error.append(temp_error / image_points[i].shape[0])
        reprojected_corners.append(temp_reprojected_corners)

    return np.array(error), np.array(reprojected_corners)

def optimization_function(self, params, R, T, image_points, world_points):
    
    error, _ = self.calculate_reprojection_error(
        params, R, T, image_points, world_points
    )
    return error.flatten()

def getTFs(csv):
    transformations = []
    for tf in csv:
        transformation = np.eye(4)
        for i in range(4):
            transformation[i,:] = tf[i*4:i*4+4]
        transformations.append(transformation)
    return transformations

def T_inv(transformation):
    t_inverse = np.linalg.inv(transformation)
    return t_inverse

def objective_function(X_vec, A_list, B_list):
    X = np.reshape(X_vec, (4, 4))
    total_residual = 0
    for A, B in zip(A_list, B_list):
        residual = A @ X - X @ B
        total_residual += np.linalg.norm(residual, 'fro')**2
    return total_residual

def last_row_constraint(X_vec):
    X = np.reshape(X_vec, (4, 4))
    return np.concatenate([X[-1, :-1], [X[-1, -1] - 1]])

def orthogonality_constraint(X_vec):

    X = np.reshape(X_vec, (4, 4))
    X = X[:3,:3]
    I = np.eye(3)
    return (np.linalg.norm(X.T @ X - I, 'fro'))

def camera_calibration(T_HC, T_EB, iterations = 200):
    
    X0 = np.ones((4, 4))
    X0_vec = X0.flatten()
    
    best_X = None
    best_value = float('inf')
    
    for i in tqdm(range(iterations)):
        # Randomly select 4 distinct pairs of indices
        random_indices = random.sample(range(0, len(T_EB)), 5)
        
        A_list = []
        B_list = []
        
        for idx in random_indices:
            idx2 = random.choice([j for j in range(len(T_HC)) if j != idx])
            T_HC1 = T_HC[idx]
            T_HC2 = T_HC[idx2]
            T_EB1 = T_EB[idx]
            T_EB2 = T_EB[idx2]
            
            A = T_inv(T_EB2) @ T_EB1
            B = T_HC2 @ T_inv(T_HC1)
            
            A_list.append(A)
            B_list.append(B)
        
        constraints = [
            {'type': 'eq', 'fun': last_row_constraint},  # Last row constraint
            {'type': 'eq', 'fun': orthogonality_constraint}  # Orthogonality constraint
        ]
    
        result = minimize(objective_function, X0_vec, args=(A_list, B_list), constraints=constraints, method='SLSQP')
        
        X_opt = np.reshape(result.x, (4, 4))
        # Update the best solution if needed
        current_value = residual_calculation(T_HC, T_EB, X_opt)
        if current_value < best_value:
            best_value = current_value
            best_X = X_opt
            X0 = best_X
            X0_vec = X0.flatten()
    
    return best_X

def residual_calculation(T_HC, T_EB, t_CE):
    X0 = t_CE
    total_residual = 0
    for idx in range(1,len(T_HC)):
        T_HC1 = T_HC[0]
        T_HC2 = T_HC[idx]
        T_EB1 = T_EB[0]
        T_EB2 = T_EB[idx]
        
        A = T_inv(T_EB2) @ T_EB1
        B = T_HC2 @ T_inv(T_HC1)
        
        residual = A@X0 - X0@B
        total_residual += np.linalg.norm(residual, 'fro')**2
    
    total_residual = total_residual/(len(T_HC)-1)
    return total_residual
        

def transform_points(points, pose):

    num_points = points.shape[0]
    points_homogeneous = np.hstack((points, np.ones((num_points, 1))))
    transformed_points_homogeneous = points_homogeneous @ pose.T
    transformed_points = transformed_points_homogeneous[:, :3]
    
    return transformed_points
        
def visualize_point_lists(points1, points2, color1=[1, 0, 0], color2=[0, 1, 0]):

    pc1 = o3d.geometry.PointCloud()
    pc1.points = o3d.utility.Vector3dVector(points1)
    pc1.paint_uniform_color(color1) 
    
    pc2 = o3d.geometry.PointCloud()
    pc2.points = o3d.utility.Vector3dVector(points2)
    pc2.paint_uniform_color(color2) 

    o3d.visualization.draw_geometries([pc1, pc2])

def calculate_reprojection_error(points1, points2):
    assert points1.shape == points2.shape, "Point lists must have the same shape."
    
    # Calculate the reprojection error for each point pair
    errors = np.linalg.norm(points1 - points2, axis=1)
    
    # Find indices of points where the error is greater than 5 centimeters (0.05)
    outlier_indices = np.where(errors > 0.05)[0]
    
    # Print the indices of the outliers
    if len(outlier_indices) > 0:
        print("Indices of outlier points:", outlier_indices)
    
    # Exclude outliers from the error array
    errors_cleaned = np.delete(errors, outlier_indices)
    
    # Optionally, remove the largest error if needed (after removing outliers)
    if len(errors_cleaned) > 0:
        errors_sorted = np.argsort(errors_cleaned)
        errors_cleaned = np.delete(errors_cleaned, errors_sorted[-1])
        print("Maximum error after outlier removal:", np.max(errors_cleaned))
    
    # Return the mean of the cleaned errors
    return np.mean(errors_cleaned) if len(errors_cleaned) > 0 else 0.0

def visualize_camera_poses(poses):

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    vis.add_geometry(base_frame)


    for pose in poses:
        cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
        cam_frame.transform(pose)
        vis.add_geometry(cam_frame)

    vis.run()
    vis.destroy_window()

def create_camera_frustum(scale=1.0):
    # Define the vertices of the pyramid (camera frustum)
    vertices = np.array([[0, 0, 0],  # Camera origin
                         [1, 1, 2],  # Top right
                         [-1, 1, 2],  # Top left
                         [-1, -1, 2],  # Bottom left
                         [1, -1, 2]], dtype=np.float64)  # Ensure vertices are float64
    vertices *= scale

    # Define the lines that make up the edges of the frustum
    lines = [[0, 1], [0, 2], [0, 3], [0, 4],  # From camera origin to the four corners
             [1, 2], [2, 3], [3, 4], [4, 1]]  # The square at the far end of the frustum

    # Define the colors for the lines (all white)
    colors = [[1, 1, 1] for _ in range(len(lines))]

    # Create the LineSet object
    frustum = o3d.geometry.LineSet()
    frustum.points = o3d.utility.Vector3dVector(vertices)
    frustum.lines = o3d.utility.Vector2iVector(lines)
    frustum.colors = o3d.utility.Vector3dVector(colors)

    return frustum


def visualize_camera_frustums(poses, scale=0.01):
    # Create an Open3D visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Create and add the base coordinate frame
    base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    vis.add_geometry(base_frame)

    # Create and add the frustums for each camera pose
    for pose in poses:
        frustum = create_camera_frustum(scale)
        frustum.transform(pose)
        vis.add_geometry(frustum)

    # Start the visualizer
    vis.run()
    vis.destroy_window()