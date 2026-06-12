#!/usr/bin/env python3
# Copyright (C) 2025 Dexmate Inc.

"""Example demonstrating how to access 3D LiDAR point cloud data.

This example shows:
1. How to initialize and access the 3D LiDAR sensor
2. Different ways to retrieve point cloud data
3. How to access metadata (timestamps, point count, cloud shape)
4. Fast 3D visualization of point cloud data with Open3D

Run this example:
    python examples/basic_examples/sensors/get_3d_lidar_data.py

Note: Requires Open3D for fast visualization. Install with:
    pip install open3d

For headless systems or if Open3D is not available, the example will
still run and display point cloud statistics.
"""

import time

import numpy as np
from dexcontrol.config.vega import get_vega_config

from dexcontrol import Robot

# Try to import Open3D
try:
    import open3d as o3d

    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("⚠️  Open3D not available. Install with: pip install open3d")
    print("   Visualization will be skipped, but data access will still work.\n")


def visualize_with_open3d_live(lidar_sensor, duration_sec=300, update_rate_hz=10):
    """Live visualization of point cloud with Open3D.

    Args:
        lidar_sensor: Lidar3DSensor instance
        duration_sec: How long to visualize (seconds)
        update_rate_hz: Update frequency (Hz)
    """
    if not HAS_OPEN3D:
        print("Open3D not available, skipping visualization")
        return

    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D LiDAR Live Point Cloud", width=1280, height=720)

    # Set viewing options
    opt = vis.get_render_option()
    opt.background_color = np.array([0.95, 0.95, 0.95])  # Light gray background
    opt.point_size = 3.0  # Larger point size for better visibility
    opt.show_coordinate_frame = True

    # Add coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    vis.add_geometry(coord_frame)

    # Initialize with empty point cloud
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)

    # Set initial camera view
    ctr = vis.get_view_control()
    ctr.set_zoom(0.6)
    ctr.set_front([0.5, -0.3, -0.4])
    ctr.set_up([0, 0, 1])

    print("\n🎨 Open3D Live Viewer Controls:")
    print("  - Left mouse: Rotate")
    print("  - Middle mouse: Pan")
    print("  - Scroll: Zoom")
    print("  - Press 'q' or close window to exit")
    print(f"  - Live updating at {update_rate_hz} Hz for {duration_sec}s\n")

    update_interval = 1.0 / update_rate_hz
    start_time = time.time()
    frame_count = 0

    try:
        while time.time() - start_time < duration_sec:
            # Get latest point cloud
            points = lidar_sensor.get_points()
            obs = lidar_sensor.get_obs()

            if points is not None and obs is not None and len(points) > 0:
                # Update point cloud data
                pcd.points = o3d.utility.Vector3dVector(points)

                # Use gray color for all points
                gray_color = np.array([0.5, 0.5, 0.5])  # Medium gray
                colors = np.tile(gray_color, (len(points), 1))

                pcd.colors = o3d.utility.Vector3dVector(colors)

                # Update geometry
                vis.update_geometry(pcd)

                frame_count += 1

            # Poll events and update renderer
            vis.poll_events()
            vis.update_renderer()

            # Control update rate
            time.sleep(update_interval)

            # Check if window was closed
            if not vis.poll_events():
                break

    except KeyboardInterrupt:
        print("\nVisualization interrupted by user")
    finally:
        vis.destroy_window()
        print(f"\n✓ Displayed {frame_count} frames")


def visualize_with_open3d_snapshot(points, intensity=None):
    """Visualize a single point cloud snapshot with Open3D.

    Args:
        points: Nx3 array of xyz coordinates
        intensity: Optional intensity values for color mapping
    """
    if not HAS_OPEN3D:
        print("Open3D not available, skipping visualization")
        return

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Use gray color for all points
    gray_color = np.array([0.5, 0.5, 0.5])  # Medium gray
    colors = np.tile(gray_color, (len(points), 1))

    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="3D LiDAR Point Cloud Snapshot", width=1280, height=720
    )
    vis.add_geometry(pcd)

    # Set viewing options
    opt = vis.get_render_option()
    opt.background_color = np.array([0.95, 0.95, 0.95])  # Light gray background
    opt.point_size = 3.0  # Larger point size
    opt.show_coordinate_frame = True

    # Set camera view
    ctr = vis.get_view_control()
    ctr.set_zoom(0.6)
    ctr.set_front([0.5, -0.3, -0.4])
    ctr.set_up([0, 0, 1])

    # Add coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    vis.add_geometry(coord_frame)

    print("\n🎨 Open3D Viewer Controls:")
    print("  - Left mouse: Rotate")
    print("  - Middle mouse: Pan")
    print("  - Scroll: Zoom")
    print("  - Press 'q' or close window to exit\n")

    vis.run()
    vis.destroy_window()


def main():
    """Main example demonstrating 3D LiDAR data access."""
    print("=" * 60)
    print("3D LiDAR Data Access Example")
    print("=" * 60)
    configs = get_vega_config()
    configs.enable_sensor("front_lidar_3d")
    with Robot(configs=configs) as robot:
        # Check if 3D lidar is available
        if not robot.has_sensor("front_lidar_3d"):
            print("❌ Front 3D LiDAR sensor not available on this robot")
            return

        front_lidar_3d = robot.sensors.front_lidar_3d

        print(f"\n3D LiDAR Name: {front_lidar_3d.name}")
        print("Waiting for 3D LiDAR to become active...")

        # Wait for sensor to start publishing data
        if not front_lidar_3d.wait_for_active(timeout=10.0):
            print("❌ 3D LiDAR did not become active within timeout")
            return

        print("✓ 3D LiDAR is active\n")

        # Demonstrate different ways to access point cloud data
        for i in range(5):
            print(f"\n--- Scan {i + 1} ---")

            # Method 1: Get full observation dictionary
            obs = front_lidar_3d.get_obs()
            if obs:
                print(f"Point count: {obs['point_count']}")
                print(f"Timestamp (ns): {obs['timestamp_ns']}")
                print(f"Sequence: {obs['sequence']}")
                print(f"Cloud shape (H×W): {obs['height']}×{obs['width']}")
                print(f"Is dense: {obs['is_dense']}")
                print(f"X range: [{obs['x'].min():.2f}, {obs['x'].max():.2f}] m")
                print(f"Y range: [{obs['y'].min():.2f}, {obs['y'].max():.2f}] m")
                print(f"Z range: [{obs['z'].min():.2f}, {obs['z'].max():.2f}] m")
                print(
                    f"Intensity range: [{obs['intensity'].min()}, {obs['intensity'].max()}]"
                )

            # Method 2: Get points as Nx3 array
            points = front_lidar_3d.get_points()
            if points is not None:
                print(f"\nPoints shape (Nx3): {points.shape}")
                print(f"Sample points:\n{points[:3]}")

            # Method 3: Get points with intensity as Nx4 array
            points_xyzi = front_lidar_3d.get_points_with_intensity()
            if points_xyzi is not None:
                print(f"\nPoints+Intensity shape (Nx4): {points_xyzi.shape}")

            # Method 4: Get x, y, z separately
            xyz = front_lidar_3d.get_xyz()
            if xyz:
                x, y, z = xyz
                print(f"\nSeparate arrays - X: {x.shape}, Y: {y.shape}, Z: {z.shape}")

            # Additional metadata
            point_count = front_lidar_3d.get_point_count()
            cloud_shape = front_lidar_3d.get_cloud_shape()
            is_dense = front_lidar_3d.is_dense()

            print("\nMetadata:")
            print(f"  Total points: {point_count}")
            print(f"  Cloud shape: {cloud_shape}")
            print(f"  Dense cloud: {is_dense}")

            time.sleep(0.5)

        # Demonstrate filtering and processing
        print("\n" + "=" * 60)
        print("Point Cloud Processing Examples")
        print("=" * 60)

        points = front_lidar_3d.get_points()
        if points is not None:
            # Example 1: Filter points by distance
            distances = np.linalg.norm(points, axis=1)
            near_points = points[distances < 5.0]  # Points within 5 meters
            print(f"\nPoints within 5m: {len(near_points)} / {len(points)}")

            # Example 2: Filter points by height
            ground_points = points[points[:, 2] < 0.2]  # Points close to ground
            print(f"Ground points (z < 0.2m): {len(ground_points)}")

            # Example 3: Get points in front of robot
            front_points = points[points[:, 0] > 0]  # Positive X is forward
            print(f"Points in front: {len(front_points)}")

            # Example 4: Downsample point cloud
            step = 10
            downsampled = points[::step]
            print(f"\nDownsampled cloud: {len(downsampled)} points (1/{step})")

        # Demonstrate accessing per-point timestamps and ring info
        point_timestamps = front_lidar_3d.get_point_timestamps()
        ring_info = front_lidar_3d.get_ring()

        if point_timestamps is not None:
            print(f"\nPer-point timestamps available: {len(point_timestamps)} points")

        if ring_info is not None:
            unique_rings = np.unique(ring_info)
            print(f"Unique rings/channels: {unique_rings}")

        # Live visualization with Open3D (if available)
        if HAS_OPEN3D:
            print("\n" + "=" * 60)
            print("Live 3D Visualization with Open3D")
            print("=" * 60)

            print("\nStarting live point cloud visualization...")
            print("The viewer will update in real-time for 30 seconds.")
            print("Close the window or press 'q' to exit early.\n")

            # Live visualization for 300 seconds at 10 Hz
            visualize_with_open3d_live(
                front_lidar_3d, duration_sec=300, update_rate_hz=10
            )

        print("\n" + "=" * 60)
        print("Example completed successfully!")
        print("=" * 60)


if __name__ == "__main__":
    main()
