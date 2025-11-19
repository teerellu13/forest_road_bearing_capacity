import os
from typing import List, Tuple, Dict

import statistics
import pandas as pd
import plotly.express as px
from scipy.spatial import cKDTree

from typing import List, Tuple, Optional
import pyvista as pv
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD

import rasterio
from rasterio.plot import show

from sklearn.neighbors import NearestNeighbors
from numpy.linalg import svd

flag_col = 4

def mean_center_xy(original_points): 
    max_x, max_y, max_z = np.max(original_points[:, :3], axis=0)
    min_x, min_y, min_z = np.min(original_points[:, :3], axis=0)

    center_x = (max_x + min_x) / 2
    center_y = (max_y + min_y) / 2
    center_z = (max_z + min_z) / 2

    transformed_points = original_points.copy()

    transformed_points[:, 0] = transformed_points[:, 0] - center_x
    transformed_points[:, 1] = transformed_points[:, 1] - center_y
    transformed_points[:, 2] = transformed_points[:, 2] - center_z

    # TODO use only ground points for SVD, now data lacks classification attribute
    svd = TruncatedSVD(n_components=2)
    transformed_points[:,:2] = svd.fit_transform(transformed_points[:,:2])  # no centering here

    return transformed_points, original_points


def compute_cross_sectional_profile(triplets):
    values = [middle[2] - (left[2] + right[2]) / 2 for left, right, middle in triplets]
    return statistics.median(values), statistics.mean(values), statistics.stdev(values)

def compute_road_skeweness(triplets):
    values = [left[2] + right[2] for left, right, _ in triplets]
    return statistics.median(values), statistics.mean(values), statistics.stdev(values)


def compute_ditch_depths(ditch_profiles, triplets):
    left_edge_z = statistics.median([left[2] for left, _, _ in triplets])
    right_edge_z = statistics.median([right[2] for _, right, _ in triplets])

    ditch_profile_depths_left = []
    ditch_profile_depths_right = []

    for i in range(len(ditch_profiles[0]["left"])-1):
        if not (np.isnan(ditch_profiles[i]["left"][i]).any() or np.isnan(ditch_profiles[i]["right"][i]).any()):
            left_min_z = np.min(ditch_profiles[i]["left"][i])
            right_min_z = np.min(ditch_profiles[i]["right"][i])
            # print(left_min_z, right_min_z)

            ditch_profile_depths_left.append(triplets[i][0][2] - left_min_z)
            ditch_profile_depths_right.append(triplets[i][1][2] - right_min_z)

            # print(triplets[i][0][2], left_min_z)

    return np.median(ditch_profile_depths_right), np.median(ditch_profile_depths_left)

def compute_ditch_slopes(ditch_profiles):
    ditch_angles_left = []
    ditch_angles_right = []

    for i in range(len(ditch_profiles[0]["left"])-1):
        left_z = ditch_profiles[i]["left"][i]
        right_z = ditch_profiles[i]["right"][i]

        if not (np.isnan(left_z).any() or np.isnan(right_z).any()):
            left_z = left_z.astype(float)
            right_z = right_z.astype(float) 

            n_left = len(left_z)
            y = np.arange(0, n_left*0.3, 0.3)

            n_right = len(right_z)
            y = np.arange(0, n_right*0.3, 0.3)

            # Fit z = m*y + b (least squares)
            m_left, _b = np.polyfit(y, left_z, 1)
            m_right, _b = np.polyfit(y, right_z, 1)

            # Absolute angle of the slope (degrees)
            angle_deg_left = float(np.degrees(np.arctan(m_left)))
            angle_deg_right = float(np.degrees(np.arctan(m_right)))

            ditch_angles_left.append(angle_deg_left)
            ditch_angles_right.append(angle_deg_right)

    return np.median(ditch_angles_left), np.median(ditch_angles_right)

def fetch_road_triplets(
    pts: np.ndarray, flag_col: int,
    y_start: float = -1.6, y_stop: float = 1.6,
    y_step: float = 0.20, k: int = 1
) -> Tuple[float, list]:
    mask = pts[:, flag_col] == 0
    road_pts = pts[mask]
    if road_pts.size == 0:
        return np.nan, []

    xs, ys = road_pts[:, 0], road_pts[:, 1]
    widths = []
    triplets = []
    bin_edges = np.arange(y_start, y_stop, y_step)
    mid_points_idxs = []

    tree = cKDTree(road_pts[:, :2])

    for low in bin_edges:
        high = low + y_step
        in_bin = (ys >= low) & (ys < high)
        bin_pts = road_pts[in_bin]
        bin_xs = bin_pts[:, 0]

        if bin_pts.shape[0] < 2 * k:
            continue

        left_indices = np.argpartition(bin_xs, k - 1)[:k]
        right_indices = np.argpartition(-bin_xs, k - 1)[:k]

        left_pts = bin_pts[left_indices]
        right_pts = bin_pts[right_indices]

        # Sort by y-coordinate for consistent pairing
        left_pts = left_pts[np.argsort(left_pts[:, 1])]
        right_pts = right_pts[np.argsort(right_pts[:, 1])]

        left_med = np.median(left_pts[:, 0])
        right_med = np.median(right_pts[:, 0])

        # Choose a representative left and right point (median x-value closeness)
        left_med_idx = np.argmin(np.abs(left_pts[:, 0] - left_med))
        right_med_idx = np.argmin(np.abs(right_pts[:, 0] - right_med))

        left_pt = left_pts[left_med_idx]
        right_pt = right_pts[right_med_idx]

        # Find the midpoint and its closest road point
        midpoint = (left_pt + right_pt) / 2
        midpoint_xy = midpoint[:2]
        dist, idx = tree.query(midpoint_xy, k=1)
        mid_points_idxs.append(idx)
        middle_pt = road_pts[idx]

        # Store as a triplet (left, right, middle)
        triplets.append((left_pt, right_pt, middle_pt))

    return triplets, mid_points_idxs


def mls_to_als(mid_point, big_round=True):
    if big_round:
        translation = np.array([4.217286e+06, -5.329880e+06, -1.310565e+05])
        rotation = np.array([
            [ 7.465840e-01, -6.652710e-01,  5.179679e-03],
            [ 6.652199e-01,  7.463672e-01, -2.045683e-02],
            [ 9.743394e-03,  1.871837e-02,  9.997773e-01]
        ])
        rot_inv = rotation.T
        trans_inv = -rot_inv @ translation
        # print(f"mid point shape: {mid_point.shape}")
        # print(f"rot_inv shape: {rot_inv.shape}")
        # print(f"trans_inv shape: {trans_inv.shape}")
        points_als = (rot_inv @ mid_point).T + trans_inv
        return points_als

def fetch_middle_point_of_road(triplets):  
    mid_points = np.array([middle for _, _, middle in triplets])
    median_mid_point = np.median(mid_points, axis=0)
    median_mid_point = mls_to_als(median_mid_point[:3], big_round=True).flatten()
    
    return median_mid_point


def ditch_profiles_from_triplets(
    pts: np.ndarray,
    triplets: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    flag_col: Optional[int] = None,          # if provided, rows with flag==0 are treated as "road" and excluded
    y_start: float = -1.6, y_stop: float = 1.6, y_step: float = 0.20,  # SAME y-bins as your road binning
    x_width: float = 2.2,                    # ditch width from road edge
    x_step: float = 0.30                     # x_bin size
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, np.ndarray]]]:

    if flag_col is not None:
        ditch_pts = pts[pts[:, flag_col] != 0]
    else:
        ditch_pts = pts

    if ditch_pts.size == 0 or len(triplets) == 0:
        y_bin_edges = np.arange(y_start, y_stop, y_step)
        x_bin_edges = np.arange(0.0, x_width, x_step)
        return y_bin_edges, x_bin_edges, []

    X = ditch_pts[:, 0]
    Y = ditch_pts[:, 1]
    Z = ditch_pts[:, 2]

    # Bins
    y_bin_edges = np.arange(y_start, y_stop, y_step)
    n_y_bins = len(y_bin_edges)
    x_bin_edges = np.arange(0.0, x_width, x_step)
    n_x_bins = len(x_bin_edges)

    profiles: List[Dict[str, np.ndarray]] = []

    def fill_row_for_side(edge_x: float, yi: int, side: str) -> np.ndarray:
        grid = np.full((n_y_bins, n_x_bins), np.nan, dtype=float)

        # Select points in this y-bin row
        y_low = y_start + yi * y_step
        y_high = y_low + y_step
        in_row = (Y >= y_low) & (Y < y_high)
        if not np.any(in_row):
            return grid

        Xr = X[in_row]
        Zr = Z[in_row]

        if side == "left":
            in_side = (Xr <= edge_x) & (Xr >= edge_x - x_width)
            x_off = edge_x - Xr[in_side] 
        else:  # "right"
            in_side = (Xr >= edge_x) & (Xr <= edge_x + x_width)
            x_off = Xr[in_side] - edge_x  

        if not np.any(in_side):
            return grid

        Zs = Zr[in_side]
        x_idx = np.floor(x_off / x_step).astype(int)
        valid = (x_idx >= 0) & (x_idx < n_x_bins)
        if not np.any(valid):
            return grid

        x_idx = x_idx[valid]
        Zs = Zs[valid]

        order = np.argsort(x_idx)
        x_idx = x_idx[order]
        Zs = Zs[order]

        start = 0
        while start < x_idx.size:
            end = start + 1
            while end < x_idx.size and x_idx[end] == x_idx[start]:
                end += 1
            xi = int(x_idx[start])
            grid[yi, xi] = np.nanmin(Zs[start:end])
            start = end

        return grid

    for (left_pt, right_pt, middle_pt) in triplets:
        y_val = float(middle_pt[1])
        yi = int(np.floor((y_val - y_start) / y_step))
        if yi < 0 or yi >= n_y_bins:
            profiles.append({"left": np.full((n_y_bins, n_x_bins), np.nan),
                             "right": np.full((n_y_bins, n_x_bins), np.nan)})
            continue

        lx = float(left_pt[0])
        rx = float(right_pt[0])

        left_grid  = fill_row_for_side(lx, yi, "left")
        right_grid = fill_row_for_side(rx, yi, "right")

        profiles.append({"left": left_grid, "right": right_grid})

    return y_bin_edges, x_bin_edges, profiles

def compute_slope(pts: np.ndarray) -> float:
    y = pts[:, 1].astype(float)
    z = pts[:, 2].astype(float)

    # Fit z = m*y + b (least squares)
    m, _b = np.polyfit(y, z, 1)

    # Absolute angle of the slope (degrees)
    angle_deg = float(np.degrees(np.arctan(m)))
    return abs(angle_deg)

def compute_width(triplets: np.array):
    widths = [right[0] - left[0] for left, right, _ in triplets]
    return statistics.median(widths)

def compute_average_intensity_value_for_road(points):
    road_points = points[points[:,flag_col] == 0]
    return np.mean(road_points[:,3]), np.std(road_points[:,3])

def get_high_veg_points_above_road(triplets: np.array, all_points: np.array):
    median_right_edge_x = np.median([r[0] for _, r, _ in triplets])
    median_left_edge_x = np.median([l[0] for l, _, _ in triplets])

    hv = all_points[all_points[:, 5] == 1]
    if median_left_edge_x < median_right_edge_x:
        hv = hv[(hv[:, 0] > median_left_edge_x) & (hv[:, 0] < median_right_edge_x)]
  
    return hv

def compute_stand_height(triplets: np.array, all_points: np.array):
    median_right_edge_x = np.median([r[0] for _, r, _ in triplets])
    median_left_edge_x = np.median([l[0] for l, _, _ in triplets])
    
    median_right_edge_z = np.median([r[2] for _, r, _ in triplets])
    median_left_edge_z = np.median([l[2] for l, _, _ in triplets])
    
    right_side_points = all_points[all_points[:, 0] > median_right_edge_x]
    left_side_points = all_points[all_points[:, 0] < median_left_edge_x]
    
    right_highest = np.sort(right_side_points[:, 2])[-50:] 
    left_highest = np.sort(left_side_points[:, 2])[-50:] 
    
    median_right_highest = np.median(right_highest) if len(right_highest) > 0 else np.nan
    median_left_highest = np.median(left_highest) if len(left_highest) > 0 else np.nan
    
    stand_height_right = median_right_highest - median_right_edge_z if not np.isnan(median_right_highest) else np.nan
    stand_height_left = median_left_highest - median_left_edge_z if not np.isnan(median_left_highest) else np.nan
    
    return stand_height_left, stand_height_right

def cov_3x3(neigh_pts):
    c = neigh_pts.mean(axis=0, keepdims=True)
    X = neigh_pts - c
    M = X.shape[0]
    if M <= 1:
        return None
    return (X.T @ X) / (M - 1)

def compute_road_surface_roughness(points: np.ndarray):
    road_points = points[points[:,flag_col] == 0][:,:3] 

    nbrs = NearestNeighbors(n_neighbors=16, algorithm='auto').fit(road_points)
    dist, indices = nbrs.kneighbors(road_points)

    surface_variation = np.zeros(len(indices)) 

    for k, ind in enumerate(indices):
        neigh_pts = road_points[ind,:] 
        cov = cov_3x3(neigh_pts)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        surface_variation[k] = eigenvalues[0] / eigenvalues.sum()

    return np.mean(surface_variation), np.std(surface_variation)
