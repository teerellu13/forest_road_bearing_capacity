import numpy as np
import pyvista as pv

def visualize_with_pyvista(
    pts: np.ndarray,
    triplets,
    ditch,                       # either (y_edges, x_edges, profiles) OR just profiles (list)
    on_road_veg_height,
    number_of_on_road_veg_points,
    high_veg_points_above_road,
    stand_height_left,
    stand_height_right,
    mean_road_surface_variation,
    std_road_surface_variation,
    width,
    slope,
    cross_sectional_profile,
    right_ditch_depth,
    left_ditch_depth,
    ditch_slope_right,
    ditch_slope_left,
    average_intensity_value_for_road,
    std_intensity_value_for_road,
    y_edges: np.ndarray = None,  # required if 'ditch' is just a profiles list
    x_edges: np.ndarray = None,  # required if 'ditch' is just a profiles list
    cloud_point_size: float = 4.0,
    triplet_point_size: float = 5.0,
    ditch_point_size: float = 5.0,
    road_point_size: float = 4.0,
    view_xy: bool = False,
):
    if isinstance(ditch, (tuple, list)) and len(ditch) == 3 and isinstance(ditch[0], np.ndarray):
        y_edges, x_edges, profiles = ditch
    else:
        profiles = ditch
        if y_edges is None or x_edges is None:
            raise ValueError()

    cloud_xyz = pts[:, :3]
    cloud = pv.PolyData(cloud_xyz)

    road_mask = pts[:, 4] == 0
    on_road_veg_mask = pts[:, 4] == 2
    road_pd = pv.PolyData(pts[road_mask, :3])
    on_road_veg_pd = pv.PolyData(pts[on_road_veg_mask, :3])

    triplet_pd = None
    if len(triplets) > 0:
        triplet_xyz = np.vstack([np.vstack([l[:3], r[:3], m[:3]]) for (l, r, m) in triplets])
        triplet_pd = pv.PolyData(triplet_xyz)

    y_step = (y_edges[1] - y_edges[0]) if len(y_edges) > 1 else 0.10
    x_step = (x_edges[1] - x_edges[0]) if len(x_edges) > 1 else 0.10
    y_centers = y_edges + 0.5 * y_step
    x_centers = x_edges + 0.5 * x_step

    ditch_points_all = []
    line_cells = []  
    current_idx = 0

    def add_profile_line(x_vals, y_vals, z_vals):
        nonlocal current_idx
        order = np.argsort(x_vals)
        pts_profile = np.column_stack([x_vals[order], y_vals[order], z_vals[order]])
        n = len(pts_profile)
        if n < 2:
            return
        ditch_points_all.append(pts_profile)
        line = np.hstack(([n], np.arange(current_idx, current_idx+n)))
        line_cells.append(line)
        current_idx += n

    for i, (lpt, rpt, _mpt) in enumerate(triplets):
        if i >= len(profiles):
            break
        grids = profiles[i]
        left_grid  = grids.get("left")
        right_grid = grids.get("right")

        # LEFT side
        if left_grid is not None and left_grid.size > 0:
            mask = ~np.isnan(left_grid)
            if np.any(mask):
                yi, xi = np.where(mask)
                y_vals = y_centers[yi]
                x_off  = x_centers[xi]
                x_vals = float(lpt[0]) - x_off
                z_vals = left_grid[mask]
                add_profile_line(x_vals, y_vals, z_vals)

        # RIGHT side
        if right_grid is not None and right_grid.size > 0:
            mask = ~np.isnan(right_grid)
            if np.any(mask):
                yi, xi = np.where(mask)
                y_vals = y_centers[yi]
                x_off  = x_centers[xi]
                x_vals = float(rpt[0]) + x_off
                z_vals = right_grid[mask]
                add_profile_line(x_vals, y_vals, z_vals)

    ditch_pd = None
    if ditch_points_all:
        all_points = np.vstack(ditch_points_all)
        cells = np.hstack(line_cells).astype(int)
        ditch_pd = pv.PolyData(all_points, lines=cells)

    p = pv.Plotter()
    p.set_background("black")

    p.add_mesh(cloud, color="gray", render_points_as_spheres=True, point_size=cloud_point_size)

    if triplet_pd is not None:
        p.add_mesh(triplet_pd, color="white", render_points_as_spheres=True, point_size=triplet_point_size)

    if ditch_pd is not None:
        p.add_mesh(ditch_pd, color="orange", line_width=2)
        p.add_mesh(ditch_pd, color="orange", render_points_as_spheres=True, point_size=ditch_point_size)

    if high_veg_points_above_road is not None:
        high_veg_pd = pv.PolyData(high_veg_points_above_road[:,:3])
        p.add_mesh(high_veg_pd, color="green", render_points_as_spheres=True, point_size=cloud_point_size)

    p.add_mesh(road_pd, color="brown", render_points_as_spheres=True, point_size=road_point_size)

    if on_road_veg_mask.sum() > 0:
        p.add_mesh(on_road_veg_pd, color="lime", render_points_as_spheres=True, point_size=road_point_size)

    legend_entries = []

    legend_entries.append((f"Longitudinal slope: {slope:.2f}°", "white"))
    legend_entries.append((f"Estimated road width: {width:.2f} m", "white"))
    legend_entries.append((f"Cross-sectional profile: {cross_sectional_profile:.2f} m", "white"))

    legend_entries.append((f"Ditch depth (right): {right_ditch_depth:.2f} m", "white"))
    legend_entries.append((f"Ditch depth (left): {left_ditch_depth:.2f} m", "white"))
    legend_entries.append((f"Ditch slope (left): {ditch_slope_left:.2f}°", "white"))
    legend_entries.append((f"Ditch slope (right): {ditch_slope_right:.2f}°", "white"))
    legend_entries.append((f"Stand height of on-road veg.: {on_road_veg_height:.2f} m", "white"))
    legend_entries.append((f"Number of on-road veg. points: {number_of_on_road_veg_points:.2f}", "white"))
    # legend_entries.append((f"Number of high veg. points above road: {high_veg_points_above_road.shape[0]}", "white"))
    legend_entries.append((f"Stand height (left): {stand_height_left:.2f} m", "white"))
    legend_entries.append((f"Stand height (right): {stand_height_right:.2f} m", "white"))
    legend_entries.append((f"Road surface variation: mean {mean_road_surface_variation:.4f}, std: {std_road_surface_variation:.4f}", "white"))
    legend_entries.append((f"Road intensity: mean {average_intensity_value_for_road:.4f}, std: {std_intensity_value_for_road:.4f}", "white"))

    # legend_entries.append((f"Topographic wetness index: ", "white"))
    # legend_entries.append((f"Depth to water (2x2 m resolution, LUKE 2023): {depth_to_water:.2f} cm", "white"))

    legend = p.add_legend(legend_entries, size=(0.5, 0.5))

    p.show()