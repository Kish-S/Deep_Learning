import numpy as np

# Camera intrinsics matrices
K_101 = np.array([[2.063200e+03, -5.000000e-01, 9.783000e+02],
                  [0.0, 2.062400e+03, 5.847000e+02],
                  [0.0, 0.0, 1.0]])

K_103 = np.array([[2.063400e+03, -1.000000e-01, 9.734000e+02],
                  [0.0, 2.062600e+03, 5.999000e+02],
                  [0.0, 0.0, 1.0]])

# Average focal length
focal_length_101 = (K_101[0, 0] + K_101[1, 1]) / 2
focal_length_103 = (K_103[0, 0] + K_103[1, 1]) / 2

# Translation vectors
T_101 = np.array([0.0, 0.0, 0.0])  # Assuming no translation for camera 101
T_103 = np.array([-5.446076e-01, -7.500610e-04, -2.395167e-03])

# Baseline calculation
baseline = np.linalg.norm(T_103 - T_101)

print("Average Focal Length 101:", focal_length_101)
print("Average Focal Length 103:", focal_length_103)
print("Baseline:", baseline)