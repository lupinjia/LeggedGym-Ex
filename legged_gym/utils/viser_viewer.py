"""Web-based robot visualization using viser with MuJoCo XML support."""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh
import trimesh.visual

from legged_gym import LEGGED_GYM_ROOT_DIR

try:
    import viser
    import viser.transforms as vtf
    HAS_VISER = True
except ImportError:
    HAS_VISER = False


def _xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.array([q[3], q[0], q[1], q[2]])


@dataclass
class Body:
    name: str
    pos: np.ndarray
    quat: np.ndarray
    parent: Optional[str]
    joint_name: Optional[str]
    joint_axis: Optional[np.ndarray]
    joint_range: Optional[Tuple[float, float]]


@dataclass
class Geom:
    body_name: str
    mesh_path: Optional[str]
    geom_pos: np.ndarray
    geom_quat: np.ndarray
    size: Optional[np.ndarray]
    rgba: Optional[np.ndarray]
    material: Optional[str]


class MjcfKinematicModel:
    def __init__(self, xml_path: str, dof_names: Optional[List[str]] = None):
        self.xml_path = xml_path
        self.xml_dir = os.path.dirname(os.path.abspath(xml_path))
        self._dof_names = dof_names
        
        self.bodies: Dict[str, Body] = {}
        self.geoms: List[Geom] = []
        self.meshes: Dict[str, str] = {}
        self.materials: Dict[str, np.ndarray] = {}
        self._dof_indices: Dict[str, int] = {}
        self._mesh_cache: Dict[str, trimesh.Trimesh] = {}
        self._default_joint_axis: Dict[str, np.ndarray] = {}
        
        self._parse_xml()
        self._assign_dof_indices()
        
    def _parse_xyz(self, s: Optional[str]) -> np.ndarray:
        if s is None:
            return np.zeros(3)
        return np.array([float(x) for x in s.split()])

    def _parse_rpy(self, s: Optional[str]) -> np.ndarray:
        if s is None:
            return np.zeros(3)
        return np.array([float(x) for x in s.split()])

    def _rpy_to_quat(self, rpy: np.ndarray) -> np.ndarray:
        roll, pitch, yaw = rpy
        cr, sr = np.cos(roll/2), np.sin(roll/2)
        cp, sp = np.cos(pitch/2), np.sin(pitch/2)
        cy, sy = np.cos(yaw/2), np.sin(yaw/2)
        return np.array([
            cr*cp*cy + sr*sp*sy,
            sr*cp*cy - cr*sp*sy,
            cr*sp*cy + sr*cp*sy,
            cr*cp*sy - sr*sp*cy,
        ])

    def _parse_xml(self) -> None:
        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        
        compiler = root.find('compiler')
        self.meshdir = compiler.get('meshdir', 'meshes') if compiler is not None else 'meshes'

        self._parse_defaults(root)
        self._parse_assets(root)
        self._parse_worldbody(root)
        
    def _parse_defaults(self, root: ET.Element) -> None:
        default = root.find('default')
        if default is None:
            return
        
        parent_axis = np.array([0.0, 1.0, 0.0])
        
        def parse_default_class(elem: ET.Element, parent_joint_axis: np.ndarray, class_prefix: str = "") -> None:
            class_name = elem.get('class', class_prefix)
            
            joint = elem.find('joint')
            if joint is not None:
                axis_str = joint.get('axis')
                if axis_str:
                    current_axis = self._parse_xyz(axis_str)
                else:
                    current_axis = parent_joint_axis.copy()
                self._default_joint_axis[class_name] = current_axis
            elif class_name:
                self._default_joint_axis[class_name] = parent_joint_axis.copy()
            
            for child in elem.findall('default'):
                parse_default_class(child, 
                    self._default_joint_axis.get(class_name, parent_joint_axis), 
                    child.get('class', ''))
        
        for child in default.findall('default'):
            parse_default_class(child, parent_axis)
        
    def _parse_assets(self, root: ET.Element) -> None:
        asset = root.find('asset')
        if asset is None:
            return
        
        for material in asset.findall('material'):
            name = material.get('name')
            rgba_str = material.get('rgba')
            if name and rgba_str:
                self.materials[name] = self._parse_xyz(rgba_str)

        for mesh in asset.findall('mesh'):
            name = mesh.get('name')
            filename = mesh.get('file')
            if name and filename:
                self.meshes[name] = filename

    def _parse_worldbody(self, root: ET.Element) -> None:
        worldbody = root.find('worldbody')
        if worldbody is None:
            return
        
        for body_elem in worldbody.findall('body'):
            self._parse_body(body_elem, parent_name=None)

    def _parse_body(self, body_elem: ET.Element, parent_name: Optional[str]) -> None:
        name = body_elem.get('name')
        if name is None:
            return
        
        pos = self._parse_xyz(body_elem.get('pos', '0 0 0'))
        
        euler_str = body_elem.get('euler')
        quat_str = body_elem.get('quat')
        
        if quat_str is not None:
            quat_parts = np.array([float(x) for x in quat_str.split()])
            if len(quat_parts) == 3:
                quat = np.array([1, quat_parts[0], quat_parts[1], quat_parts[2]])
            else:
                quat = quat_parts
        elif euler_str is not None:
            quat = self._rpy_to_quat(self._parse_rpy(euler_str))
        else:
            quat = np.array([1, 0, 0, 0])

        joint_elem = body_elem.find('joint')
        joint_name = joint_elem.get('name') if joint_elem is not None else None
        joint_axis = None
        joint_range = None
        
        if joint_elem is not None:
            axis_str = joint_elem.get('axis')
            if axis_str:
                joint_axis = self._parse_xyz(axis_str)
            else:
                class_name = joint_elem.get('class', '')
                joint_axis = self._default_joint_axis.get(class_name)
                if joint_axis is None:
                    for key in self._default_joint_axis:
                        if key in class_name or class_name in key:
                            joint_axis = self._default_joint_axis[key]
                            break
            
            range_str = joint_elem.get('range')
            if range_str:
                parts = range_str.split()
                joint_range = (float(parts[0]), float(parts[1]))

        self.bodies[name] = Body(
            name=name,
            pos=pos,
            quat=quat,
            parent=parent_name,
            joint_name=joint_name,
            joint_axis=joint_axis,
            joint_range=joint_range,
        )

        for geom in body_elem.findall('geom'):
            self._parse_geom(geom, name)

        for child_body in body_elem.findall('body'):
            self._parse_body(child_body, parent_name=name)

    def _parse_geom(self, geom_elem: ET.Element, body_name: str) -> None:
        mesh_name = geom_elem.get('mesh')
        
        mesh_path = None
        if mesh_name and mesh_name in self.meshes:
            mesh_path = os.path.join(self.meshdir, self.meshes[mesh_name])

        pos = self._parse_xyz(geom_elem.get('pos', '0 0 0'))
        
        euler_str = geom_elem.get('euler')
        quat_str = geom_elem.get('quat')
        
        if quat_str is not None:
            quat_parts = np.array([float(x) for x in quat_str.split()])
            if len(quat_parts) == 3:
                quat = np.array([1, quat_parts[0], quat_parts[1], quat_parts[2]])
            else:
                quat = quat_parts
        elif euler_str is not None:
            quat = self._rpy_to_quat(self._parse_rpy(euler_str))
        else:
            quat = np.array([1, 0, 0, 0])

        size_str = geom_elem.get('size')
        size = self._parse_xyz(size_str) if size_str else None

        rgba = None
        rgba_str = geom_elem.get('rgba')
        if rgba_str:
            rgba = self._parse_xyz(rgba_str)
        else:
            material_name = geom_elem.get('material')
            if material_name and material_name in self.materials:
                rgba = self.materials[material_name]

        self.geoms.append(Geom(
            body_name=body_name,
            mesh_path=mesh_path,
            geom_pos=pos,
            geom_quat=quat,
            size=size,
            rgba=rgba,
            material=geom_elem.get('material'),
        ))

    def _assign_dof_indices(self) -> None:
        if self._dof_names is None:
            return
        for i, name in enumerate(self._dof_names):
            self._dof_indices[name] = i

    def _quat_to_matrix(self, quat_wxyz: np.ndarray) -> np.ndarray:
        w, x, y, z = quat_wxyz
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
        ])

    def _quat_mul(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])

    def load_body_meshes(self) -> Dict[str, trimesh.Trimesh]:
        body_meshes: Dict[str, List[trimesh.Trimesh]] = {}
        
        for geom in self.geoms:
            if geom.mesh_path is None:
                continue
                
            full_path = os.path.join(self.xml_dir, geom.mesh_path)
            if not os.path.exists(full_path):
                continue
            
            if full_path in self._mesh_cache:
                mesh = self._mesh_cache[full_path].copy()
            else:
                try:
                    mesh = trimesh.load(full_path, force='mesh')
                    self._mesh_cache[full_path] = mesh.copy()
                except Exception:
                    continue
            
            T_geom = np.eye(4)
            T_geom[:3, :3] = self._quat_to_matrix(geom.geom_quat)
            T_geom[:3, 3] = geom.geom_pos
            mesh.apply_transform(T_geom)
            
            if geom.rgba is not None:
                color = (np.clip(geom.rgba, 0, 1) * 255).astype(np.uint8)
                mesh.visual = trimesh.visual.ColorVisuals(
                    vertex_colors=np.tile(color, (len(mesh.vertices), 1))
                )
            
            if geom.body_name not in body_meshes:
                body_meshes[geom.body_name] = []
            body_meshes[geom.body_name].append(mesh)
        
        result = {}
        for body_name, meshes in body_meshes.items():
            if meshes:
                result[body_name] = trimesh.util.concatenate(meshes)
        return result

    def _axis_angle_to_quat(self, axis: np.ndarray, angle: float) -> np.ndarray:
        if abs(angle) < 1e-10:
            return np.array([1.0, 0.0, 0.0, 0.0])
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        c = np.cos(angle / 2)
        s = np.sin(angle / 2)
        return np.array([c, axis[0] * s, axis[1] * s, axis[2] * s])

    def forward_kinematics(
        self,
        base_pos: np.ndarray,
        base_quat_wxyz: np.ndarray,
        dof_pos: np.ndarray,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        world_pos = {}
        world_quat = {}
        
        root_bodies = [n for n, b in self.bodies.items() if b.parent is None]
        if not root_bodies:
            root_bodies = [list(self.bodies.keys())[0]] if self.bodies else []
        
        for root in root_bodies:
            world_pos[root] = base_pos.copy()
            world_quat[root] = base_quat_wxyz.copy()
        
        changed = True
        while changed:
            changed = False
            for name, body in self.bodies.items():
                if name in world_pos or body.parent is None:
                    continue
                if body.parent not in world_pos:
                    continue
                
                parent_pos = world_pos[body.parent]
                parent_quat = world_quat[body.parent]
                parent_rot = self._quat_to_matrix(parent_quat)
                
                if body.joint_name and body.joint_axis is not None:
                    dof_idx = self._dof_indices.get(body.joint_name, -1)
                    if dof_idx >= 0 and dof_idx < len(dof_pos):
                        joint_angle = float(dof_pos[dof_idx])
                        joint_quat = self._axis_angle_to_quat(body.joint_axis, joint_angle)
                        effective_quat = self._quat_mul(joint_quat, body.quat)
                    else:
                        effective_quat = body.quat
                else:
                    effective_quat = body.quat
                
                world_quat[name] = self._quat_mul(parent_quat, effective_quat)
                
                # body.pos is in PARENT frame, only apply parent rotation
                world_pos[name] = parent_pos + parent_rot @ body.pos
                
                changed = True
        
        return {name: (world_pos[name], world_quat[name]) for name in world_pos}


class ViserViewer:
    def __init__(
        self,
        xml_path: str,
        dof_names: Optional[List[str]] = None,
        num_envs: int = 1,
        server: Optional[object] = None,
        host: str = "0.0.0.0",
        port: int = 8080,
    ):
        if not HAS_VISER:
            raise ImportError("viser is required. Install: pip install viser")

        self.num_envs = num_envs
        self.xml_path = xml_path

        self.kin_model = MjcfKinematicModel(xml_path, dof_names)
        self.body_meshes = self.kin_model.load_body_meshes()

        if server is not None:
            self.server = server
        else:
            self.server = viser.ViserServer(host=host, port=port)

        self._body_handles: Dict[str, object] = {}
        self._env_frames: List[object] = []

        self._build_scene()

    def _build_scene(self) -> None:
        for env_idx in range(self.num_envs):
            prefix = f"/env_{env_idx}" if self.num_envs > 1 else ""
            frame = self.server.scene.add_frame(f"{prefix}/robot", show_axes=False)
            self._env_frames.append(frame)

            for body_name, mesh in self.body_meshes.items():
                path = f"{prefix}/robot/{body_name}"
                handle = self.server.scene.add_mesh_trimesh(
                    path,
                    mesh,
                    cast_shadow=True,
                    receive_shadow=True,
                )
                self._body_handles[(env_idx, body_name)] = handle

        self.server.scene.add_grid(
            "/ground",
            infinite_grid=True,
            fade_distance=50.0,
            shadow_opacity=0.2,
            plane_opacity=0.4,
        )

        self._setup_camera()
        self._setup_camera_gui()
        self._setup_command_sliders()

    def _setup_camera(self) -> None:
        self._camera_offset = np.array([2.0, 2.0, 1.5])
        self._camera_look_at_offset = np.array([0.0, 0.0, 0.3])

        @self.server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            client.camera.position = self._camera_offset.copy()
            client.camera.look_at = self._camera_look_at_offset.copy()
            client.camera.fov = np.radians(60.0)

    def _setup_camera_gui(self) -> None:
        """Add camera tracking and FOV controls."""
        self._camera_tracking_enabled = True
        self._camera_fov_degrees = 60.0

        with self.server.gui.add_folder("Camera"):
            cb_tracking = self.server.gui.add_checkbox(
                "Track robot",
                initial_value=self._camera_tracking_enabled,
            )

            @cb_tracking.on_update
            def _(_) -> None:
                self._camera_tracking_enabled = cb_tracking.value

            slider_fov = self.server.gui.add_slider(
                "FOV (\u00b0)",
                min=30.0,
                max=120.0,
                step=1.0,
                initial_value=self._camera_fov_degrees,
            )

            @slider_fov.on_update
            def _(_) -> None:
                self._camera_fov_degrees = slider_fov.value
                for client in self.server.get_clients().values():
                    client.camera.fov = np.radians(slider_fov.value)

    def _setup_command_sliders(self) -> None:
        """Add sliders for velocity commands (lin_vel_x, lin_vel_y, ang_vel_z)."""
        self._command_sliders = {}

        with self.server.gui.add_folder("Velocity Commands", expand_by_default=True):
            self._command_sliders['lin_vel_x'] = self.server.gui.add_slider(
                "Linear X (m/s)",
                min=-2.0,
                max=2.0,
                step=0.1,
                initial_value=0.0,
            )
            self._command_sliders['lin_vel_y'] = self.server.gui.add_slider(
                "Linear Y (m/s)",
                min=-1.0,
                max=1.0,
                step=0.1,
                initial_value=0.0,
            )
            self._command_sliders['ang_vel_yaw'] = self.server.gui.add_slider(
                "Angular Yaw (rad/s)",
                min=-1.0,
                max=1.0,
                step=0.1,
                initial_value=0.0,
            )

    def get_command(self) -> np.ndarray:
        """Get current velocity command from sliders.

        Returns:
            np.ndarray: [lin_vel_x, lin_vel_y, ang_vel_yaw]
        """
        if not hasattr(self, '_command_sliders'):
            return np.array([0.0, 0.0, 0.0])

        return np.array([
            self._command_sliders['lin_vel_x'].value,
            self._command_sliders['lin_vel_y'].value,
            self._command_sliders['ang_vel_yaw'].value,
        ])

    def set_terrain_mesh(self, terrain_mesh: Optional[trimesh.Trimesh]) -> None:
        if terrain_mesh is None:
            return

        if hasattr(self, '_terrain_handle') and self._terrain_handle is not None:
            self._terrain_handle.remove()

        self._terrain_handle = self.server.scene.add_mesh_trimesh(
            "/terrain",
            terrain_mesh,
            cast_shadow=True,
            receive_shadow=True,
        )

    def set_terrain_from_heightfield(
        self,
        height_samples: np.ndarray,
        horizontal_scale: float = 0.1,
        vertical_scale: float = 0.005,
        return_mesh: bool = False,
    ):
        hfield = height_samples.astype(np.float64) * vertical_scale
        nrow, ncol = hfield.shape

        x = np.arange(ncol) * horizontal_scale
        y = np.arange(nrow) * horizontal_scale
        xx, yy = np.meshgrid(x, y)
        zz = hfield

        vertices = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))

        ri, ci = np.mgrid[:nrow-1, :ncol-1]
        i0 = (ri * ncol + ci).ravel()
        faces = np.column_stack([
            i0, i0 + 1, i0 + ncol + 1,
            i0, i0 + ncol + 1, i0 + ncol,
        ]).reshape(-1, 3)

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        heights = vertices[:, 2]
        h_min, h_max = heights.min(), heights.max()
        if h_max - h_min > 1e-6:
            normalized = (heights - h_min) / (h_max - h_min)
        else:
            normalized = np.zeros_like(heights)
        colors = np.zeros((len(heights), 4), dtype=np.uint8)
        colors[:, 0] = (50 + 100 * normalized).astype(np.uint8)
        colors[:, 1] = (150 + 100 * (1 - normalized)).astype(np.uint8)
        colors[:, 2] = (100 + 50 * normalized).astype(np.uint8)
        colors[:, 3] = 255
        mesh.visual = trimesh.visual.ColorVisuals(vertex_colors=colors)

        if return_mesh:
            return mesh
        self.set_terrain_mesh(mesh)

    def update(
        self,
        base_pos: np.ndarray,
        base_quat_wxyz: np.ndarray,
        dof_pos: np.ndarray,
        env_idx: int = 0,
    ) -> None:
        fk_results = self.kin_model.forward_kinematics(
            base_pos, base_quat_wxyz, dof_pos
        )

        with self.server.atomic():
            for body_name, (pos, quat) in fk_results.items():
                handle = self._body_handles.get((env_idx, body_name))
                if handle is not None:
                    handle.position = pos
                    handle.wxyz = quat

            if self._camera_tracking_enabled:
                for client in self.server.get_clients().values():
                    client.camera.position = base_pos + self._camera_offset
                    client.camera.look_at = base_pos + self._camera_look_at_offset

        self.server.flush()

    def update_from_simulator(self, env, robot_index: int = 0) -> None:
        base_pos = env.simulator.base_pos[robot_index].cpu().numpy()
        base_quat_xyzw = env.simulator.base_quat[robot_index].cpu().numpy()
        base_quat_wxyz = _xyzw_to_wxyz(base_quat_xyzw)
        dof_pos = env.simulator.dof_pos[robot_index].cpu().numpy()

        self.update(base_pos, base_quat_wxyz, dof_pos, env_idx=0)

    def stop(self) -> None:
        if hasattr(self, 'server'):
            self.server.stop()


def create_viser_viewer(env, host: str = "0.0.0.0", port: int = 8080, robot_index: int = 0) -> ViserViewer:
    xml_path = env.cfg.asset.xml_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)

    dof_names = env.cfg.asset.dof_names

    viewer = ViserViewer(
        xml_path=xml_path,
        dof_names=dof_names,
        num_envs=1,
        host=host,
        port=port,
    )

    try:
        terrain = getattr(env.simulator, '_terrain', None)
        if terrain is None:
            return viewer

        mesh_type = env.cfg.terrain.mesh_type

        if mesh_type == 'plane':
            pass
        elif mesh_type == 'trimesh' and hasattr(terrain, 'terrain_mesh'):
            import copy
            terrain_mesh = copy.deepcopy(terrain.terrain_mesh)
            transform = np.zeros((3,))
            transform[0] = -env.cfg.terrain.border_size
            transform[1] = -env.cfg.terrain.border_size
            transform[2] = 0.0
            translation = trimesh.transformations.translation_matrix(transform)
            terrain_mesh.apply_transform(translation)
            viewer.set_terrain_mesh(terrain_mesh)
        elif mesh_type == 'heightfield' and hasattr(terrain, 'heightsamples'):
            transform = np.zeros((3,))
            transform[0] = -env.cfg.terrain.border_size
            transform[1] = -env.cfg.terrain.border_size
            transform[2] = 0.0
            mesh = viewer.set_terrain_from_heightfield(
                terrain.heightsamples,
                horizontal_scale=env.cfg.terrain.horizontal_scale,
                vertical_scale=env.cfg.terrain.vertical_scale,
                return_mesh=True,
            )
            if mesh is not None:
                translation = trimesh.transformations.translation_matrix(transform)
                mesh.apply_transform(translation)
                viewer.set_terrain_mesh(mesh)
    except Exception as e:
        print(f"[viser_viewer] Warning: Could not load terrain: {e}")

    return viewer
