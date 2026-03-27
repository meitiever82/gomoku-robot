"""Create the Gomoku table scene USD programmatically using Isaac Sim.

Generates a scene with:
- Table surface (kinematic)
- 9x9 Gomoku board (kinematic)
- One stone (dynamic rigid body, for pick-and-place)
- Tray (kinematic)

Usage:
    source /path/to/.venv-sim/bin/activate
    python scripts/create_gomoku_scene.py [--output PATH]

The output USD is saved to the leisaac assets directory by default.
"""

import argparse
import math
import os
import sys

# --- Isaac Sim bootstrap (must come before any omni/pxr imports) ----------
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import omni.usd
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade

# ---------------------------------------------------------------------------
# Scene parameters (SI units: meters, kg)
# ---------------------------------------------------------------------------
# Board (enlarged for better visualization in Isaac Sim)
BOARD_GRID_SIZE = 9
GRID_SPACING = 0.030  # 30 mm (larger than real 22mm for visibility)
BOARD_SIZE = (BOARD_GRID_SIZE - 1) * GRID_SPACING  # 0.240 m
BOARD_MARGIN = 0.020  # extra margin around the grid
BOARD_EXTENT = BOARD_SIZE + 2 * BOARD_MARGIN  # ~0.280 m
BOARD_THICKNESS = 0.008

# Stone (scaled up proportionally)
STONE_RADIUS = 0.012  # 12 mm
STONE_HEIGHT = 0.010  # 10 mm
STONE_MASS = 0.005  # 5 g

# Tray
TRAY_SIZE_X = 0.06
TRAY_SIZE_Y = 0.06
TRAY_THICKNESS = 0.005

# Table
TABLE_SIZE_X = 0.70
TABLE_SIZE_Y = 0.60
TABLE_THICKNESS = 0.02

# Positions (world frame)
TABLE_POS = Gf.Vec3f(0.30, 0.0, 0.0)
BOARD_POS = Gf.Vec3f(0.30, 0.0, TABLE_THICKNESS / 2 + BOARD_THICKNESS / 2)
TRAY_POS = Gf.Vec3f(0.08, -0.20, TABLE_THICKNESS / 2 + TRAY_THICKNESS / 2)
STONE_POS = Gf.Vec3f(TRAY_POS[0], TRAY_POS[1], TRAY_POS[2] + TRAY_THICKNESS / 2 + STONE_HEIGHT / 2)


def _set_color(prim, color: tuple[float, float, float]):
    """Apply a simple preview surface material with the given diffuse color."""
    stage = prim.GetStage()
    mat_path = prim.GetPath().AppendPath("Material")
    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, mat_path.AppendPath("Shader"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(prim).Bind(mat)


def _make_kinematic_rigid_body(prim):
    """Apply kinematic rigid body + collision to a prim."""
    UsdPhysics.RigidBodyAPI.Apply(prim)
    prim.GetAttribute("physics:kinematicEnabled").Set(True)
    UsdPhysics.CollisionAPI.Apply(prim)
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
    mesh_collision.CreateApproximationAttr("convexHull")


def _make_dynamic_rigid_body(prim, mass: float):
    """Apply dynamic rigid body + collision + mass to a prim."""
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.CollisionAPI.Apply(prim)
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
    mesh_collision.CreateApproximationAttr("convexHull")
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(mass)
    # High friction for better grasping
    material_path = prim.GetPath().AppendPath("PhysicsMaterial")
    stage = prim.GetStage()
    mat = UsdPhysics.MaterialAPI.Apply(stage.DefinePrim(material_path))
    mat.CreateStaticFrictionAttr(1.0)
    mat.CreateDynamicFrictionAttr(1.0)
    mat.CreateRestitutionAttr(0.0)
    physx_mat = PhysxSchema.PhysxMaterialAPI.Apply(stage.GetPrimAtPath(material_path))
    physx_mat.CreateFrictionCombineModeAttr("max")
    # Bind material
    binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
    binding_api.Bind(
        UsdShade.Material(stage.GetPrimAtPath(material_path)),
        UsdShade.Tokens.weakerThanDescendants,
        "physics",
    )


def create_cube(stage, path: str, size: tuple, pos: Gf.Vec3f, color: tuple, kinematic: bool = True):
    """Create a cube prim with given size and position."""
    cube_prim = UsdGeom.Cube.Define(stage, path)
    # USD Cube has default extent [-1,1] (side length 2), so scale = desired_size / 2
    cube_prim.AddScaleOp().Set(Gf.Vec3f(size[0] / 2, size[1] / 2, size[2] / 2))
    cube_prim.AddTranslateOp().Set(Gf.Vec3d(pos[0], pos[1], pos[2]))
    prim = stage.GetPrimAtPath(path)
    _set_color(prim, color)
    if kinematic:
        _make_kinematic_rigid_body(prim)
    return prim


def create_cylinder(stage, path: str, radius: float, height: float, pos: Gf.Vec3f, color: tuple, mass: float):
    """Create a dynamic cylinder prim."""
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(radius)
    cyl.CreateHeightAttr(height)
    cyl.CreateAxisAttr("Z")
    cyl.AddTranslateOp().Set(Gf.Vec3d(pos[0], pos[1], pos[2]))
    prim = stage.GetPrimAtPath(path)
    _set_color(prim, color)
    _make_dynamic_rigid_body(prim, mass)
    return prim


def create_ground_plane(stage, path: str):
    """Create an invisible ground plane for physics."""
    ground = UsdGeom.Plane.Define(stage, path) if hasattr(UsdGeom, "Plane") else None
    if ground is None:
        # Fallback: use a large thin cube as ground
        ground_prim = UsdGeom.Cube.Define(stage, path)
        ground_prim.AddScaleOp().Set(Gf.Vec3f(5.0, 5.0, 0.001))
        ground_prim.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.0005))
        prim = stage.GetPrimAtPath(path)
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdGeom.Imageable(prim).CreateVisibilityAttr("invisible")


def create_board_grid_lines(stage, parent_path: str, board_pos: Gf.Vec3f):
    """Create thin cylinder lines on the board surface for visual reference."""
    lines_path = f"{parent_path}/GridLines"
    stage.DefinePrim(lines_path)

    line_radius = 0.0005  # 0.5 mm
    half_size = BOARD_SIZE / 2
    board_top_z = board_pos[2] + BOARD_THICKNESS / 2 + 0.0001

    line_idx = 0
    # Horizontal lines (along X)
    for i in range(BOARD_GRID_SIZE):
        y_offset = -half_size + i * GRID_SPACING
        path = f"{lines_path}/H{i}"
        line = UsdGeom.Cylinder.Define(stage, path)
        line.CreateRadiusAttr(line_radius)
        line.CreateHeightAttr(BOARD_SIZE)
        line.CreateAxisAttr("X")
        line.AddTranslateOp().Set(Gf.Vec3d(board_pos[0], board_pos[1] + y_offset, board_top_z))
        _set_color(stage.GetPrimAtPath(path), (0.1, 0.1, 0.1))

    # Vertical lines (along Y)
    for i in range(BOARD_GRID_SIZE):
        x_offset = -half_size + i * GRID_SPACING
        path = f"{lines_path}/V{i}"
        line = UsdGeom.Cylinder.Define(stage, path)
        line.CreateRadiusAttr(line_radius)
        line.CreateHeightAttr(BOARD_SIZE)
        line.CreateAxisAttr("Y")
        line.AddTranslateOp().Set(Gf.Vec3d(board_pos[0] + x_offset, board_pos[1], board_top_z))
        _set_color(stage.GetPrimAtPath(path), (0.1, 0.1, 0.1))


def create_gomoku_scene(output_path: str):
    """Create the complete Gomoku table scene and save as USD."""
    # Create a new stage
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()

    # Set up-axis to Z
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # Physics scene
    physics_scene_path = "/PhysicsScene"
    physics_scene = UsdPhysics.Scene.Define(stage, physics_scene_path)
    physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    physics_scene.CreateGravityMagnitudeAttr(9.81)

    # Root Xform — this is the defaultPrim that IsaacLab references
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

    # Ground plane
    create_ground_plane(stage, "/World/GroundPlane")

    # Table (kinematic) - tan/wood color
    create_cube(
        stage,
        "/World/Table",
        (TABLE_SIZE_X, TABLE_SIZE_Y, TABLE_THICKNESS),
        TABLE_POS,
        (0.55, 0.40, 0.25),
        kinematic=True,
    )

    # Board (kinematic) - light wood color
    create_cube(
        stage,
        "/World/Board",
        (BOARD_EXTENT, BOARD_EXTENT, BOARD_THICKNESS),
        BOARD_POS,
        (0.85, 0.70, 0.45),
        kinematic=True,
    )

    # Grid lines on the board
    create_board_grid_lines(stage, "/World", BOARD_POS)

    # Tray (kinematic) - gray
    create_cube(
        stage,
        "/World/Tray",
        (TRAY_SIZE_X, TRAY_SIZE_Y, TRAY_THICKNESS),
        TRAY_POS,
        (0.5, 0.5, 0.5),
        kinematic=True,
    )

    # Stone (dynamic rigid body) - black
    create_cylinder(
        stage,
        "/World/Stone",
        STONE_RADIUS,
        STONE_HEIGHT,
        STONE_POS,
        (0.08, 0.08, 0.08),
        STONE_MASS,
    )

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    stage.Export(output_path)
    print(f"Scene saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Create Gomoku table scene USD")
    default_output = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "leisaac", "assets", "scenes", "gomoku_table", "scene.usd",
    )
    parser.add_argument("--output", default=os.path.normpath(default_output), help="Output USD file path")
    args = parser.parse_args()

    create_gomoku_scene(args.output)
    simulation_app.close()


if __name__ == "__main__":
    main()
