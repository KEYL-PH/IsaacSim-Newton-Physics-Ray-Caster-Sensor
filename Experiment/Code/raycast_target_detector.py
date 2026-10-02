import numpy as np
import omni.usd
import warp as wp

import newton
import isaacsim.physics.newton as isaac_newton

from newton._src.core import MAXVAL
from newton._src.geometry.raycast import (
    map_ray_to_local,
    ray_intersect_mesh,
    ray_intersect_shape,
)
from newton._src.geometry.types import GeoType

from pxr import UsdPhysics

from isaacsim.sensors.experimental.physics import RaycastSensor


# =============================================================
# WARP
# =============================================================

wp.init()


# =============================================================
# NEWTON COLLISION FLAG
# =============================================================

COLLIDE_SHAPES_FLAG = 2


# =============================================================
# GPU RAYCAST
# =============================================================

@wp.kernel
def raycast_kernel(
    bvh_id: wp.uint64,
    bvh_shapes_group_roots: wp.array(dtype=wp.int32),

    bvh_shape_enabled: wp.array(dtype=wp.uint32),
    shape_transform_world: wp.array(dtype=wp.transform),
    shape_type: wp.array(dtype=wp.int32),
    shape_scale: wp.array(dtype=wp.vec3),
    shape_source_ptr: wp.array(dtype=wp.uint64),

    ray_origins: wp.array(dtype=wp.vec3),
    ray_endpoints: wp.array(dtype=wp.vec3),

    ignored_shape_ids: wp.array(dtype=wp.int32),

    out_distances: wp.array(dtype=wp.float32),
    out_endpoints: wp.array(dtype=wp.vec3),
):
    ray_index = wp.tid()

    origin = ray_origins[
        ray_index
    ]

    endpoint = ray_endpoints[
        ray_index
    ]

    delta = endpoint - origin

    length_squared = (
        delta[0] * delta[0]
        +
        delta[1] * delta[1]
        +
        delta[2] * delta[2]
    )

    if length_squared < 1.0e-16:

        out_distances[
            ray_index
        ] = 0.0

        out_endpoints[
            ray_index
        ] = origin

        return

    ray_length = wp.sqrt(
        length_squared
    )

    direction = (
        delta / ray_length
    )

    min_dist = ray_length

    min_shape_id = wp.int32(
        -1
    )

    # ---------------------------------------------------------
    # Always read the CURRENT Newton BVH roots.
    #
    # The model BVH may be rebuilt by another detector instance
    # when another robot is duplicated.
    # ---------------------------------------------------------

    world_bvh_root = (
        bvh_shapes_group_roots[
            0
        ]
    )

    global_bvh_root = (
        bvh_shapes_group_roots[
            bvh_shapes_group_roots.shape[
                0
            ]
            -
            1
        ]
    )

    # ---------------------------------------------------------
    # World BVH.
    # ---------------------------------------------------------

    if world_bvh_root >= 0:

        query = wp.bvh_query_ray(
            bvh_id,
            origin,
            direction,
            world_bvh_root,
        )

        bvh_shape_id = wp.int32(
            0
        )

        while wp.bvh_query_next(
            query,
            bvh_shape_id,
            min_dist,
        ):

            shape_id = wp.int32(
                bvh_shape_enabled[
                    bvh_shape_id
                ]
            )

            if (
                ignored_shape_ids[
                    shape_id
                ] != 0
            ):

                continue

            geom_type = (
                shape_type[
                    shape_id
                ]
            )

            if (
                geom_type
                ==
                GeoType.MESH
                or
                geom_type
                ==
                GeoType.CONVEX_MESH
                or
                geom_type
                ==
                GeoType.HFIELD
            ):

                geom_to_world = (
                    shape_transform_world[
                        shape_id
                    ]
                )

                (
                    ray_origin_local,
                    ray_direction_local,
                ) = (
                    map_ray_to_local(
                        geom_to_world,
                        origin,
                        direction,
                        shape_scale[
                            shape_id
                        ],
                    )
                )

                (
                    hit_dist,
                    _hit_normal_local,
                    _u,
                    _v,
                    _face,
                ) = (
                    ray_intersect_mesh(
                        ray_origin_local,
                        ray_direction_local,
                        shape_scale[
                            shape_id
                        ],
                        shape_source_ptr[
                            shape_id
                        ],
                        False,
                        min_dist,
                    )
                )

                if (
                    hit_dist >= 0.0
                    and
                    hit_dist <= ray_length
                    and
                    hit_dist < min_dist
                ):

                    min_dist = (
                        hit_dist
                    )

                    min_shape_id = (
                        shape_id
                    )

            else:

                (
                    hit_dist,
                    _hit_normal,
                ) = (
                    ray_intersect_shape(
                        shape_transform_world[
                            shape_id
                        ],
                        shape_scale[
                            shape_id
                        ],
                        geom_type,
                        origin,
                        direction,
                        False,
                    )
                )

                if (
                    hit_dist >= 0.0
                    and
                    hit_dist <= ray_length
                    and
                    hit_dist < min_dist
                ):

                    min_dist = (
                        hit_dist
                    )

                    min_shape_id = (
                        shape_id
                    )

    # ---------------------------------------------------------
    # Global BVH.
    # ---------------------------------------------------------

    if (
        global_bvh_root >= 0
        and
        global_bvh_root != world_bvh_root
    ):

        query = wp.bvh_query_ray(
            bvh_id,
            origin,
            direction,
            global_bvh_root,
        )

        bvh_shape_id = wp.int32(
            0
        )

        while wp.bvh_query_next(
            query,
            bvh_shape_id,
            min_dist,
        ):

            shape_id = wp.int32(
                bvh_shape_enabled[
                    bvh_shape_id
                ]
            )

            if (
                ignored_shape_ids[
                    shape_id
                ] != 0
            ):

                continue

            geom_type = (
                shape_type[
                    shape_id
                ]
            )

            if (
                geom_type
                ==
                GeoType.MESH
                or
                geom_type
                ==
                GeoType.CONVEX_MESH
                or
                geom_type
                ==
                GeoType.HFIELD
            ):

                geom_to_world = (
                    shape_transform_world[
                        shape_id
                    ]
                )

                (
                    ray_origin_local,
                    ray_direction_local,
                ) = (
                    map_ray_to_local(
                        geom_to_world,
                        origin,
                        direction,
                        shape_scale[
                            shape_id
                        ],
                    )
                )

                (
                    hit_dist,
                    _hit_normal_local,
                    _u,
                    _v,
                    _face,
                ) = (
                    ray_intersect_mesh(
                        ray_origin_local,
                        ray_direction_local,
                        shape_scale[
                            shape_id
                        ],
                        shape_source_ptr[
                            shape_id
                        ],
                        False,
                        min_dist,
                    )
                )

                if (
                    hit_dist >= 0.0
                    and
                    hit_dist <= ray_length
                    and
                    hit_dist < min_dist
                ):

                    min_dist = (
                        hit_dist
                    )

                    min_shape_id = (
                        shape_id
                    )

            else:

                (
                    hit_dist,
                    _hit_normal,
                ) = (
                    ray_intersect_shape(
                        shape_transform_world[
                            shape_id
                        ],
                        shape_scale[
                            shape_id
                        ],
                        geom_type,
                        origin,
                        direction,
                        False,
                    )
                )

                if (
                    hit_dist >= 0.0
                    and
                    hit_dist <= ray_length
                    and
                    hit_dist < min_dist
                ):

                    min_dist = (
                        hit_dist
                    )

                    min_shape_id = (
                        shape_id
                    )

    # ---------------------------------------------------------
    # Final output.
    # ---------------------------------------------------------

    if min_shape_id < 0:

        out_distances[
            ray_index
        ] = ray_length

        out_endpoints[
            ray_index
        ] = endpoint

    else:

        out_distances[
            ray_index
        ] = min_dist

        out_endpoints[
            ray_index
        ] = (
            origin
            +
            direction * min_dist
        )


# =============================================================
# DETECTOR
# =============================================================

class RaycastTargetDetector:

    def __init__(
        self,
        sensor_paths,
        max_range=500.0,
    ):

        self.sensor_paths = list(
            sensor_paths
        )

        self.max_range = float(
            max_range
        )

        self.stage = None

        # -----------------------------------------------------
        # Raycast sensors
        # -----------------------------------------------------

        self.sensors = []

        # -----------------------------------------------------
        # Newton
        # -----------------------------------------------------

        self.newton_stage = None
        self.model = None
        self.state = None

        # -----------------------------------------------------
        # Warp
        # -----------------------------------------------------

        self.wp_device = None

        # -----------------------------------------------------
        # Sensor articulation filtering
        # -----------------------------------------------------

        self.ignored_articulation_roots = []

        self.gpu_ignored_shape_ids = None

        # -----------------------------------------------------
        # Persistent ray buffers
        # -----------------------------------------------------

        self.ray_capacity = 0

        self.gpu_ray_origins = None
        self.gpu_ray_endpoints = None

        # -----------------------------------------------------
        # Final outputs
        # -----------------------------------------------------

        self.gpu_distances = None
        self.gpu_endpoints = None

    # =========================================================
    # ARTICULATION DISCOVERY
    # =========================================================

    def _find_sensor_articulation_roots(
        self,
    ):

        roots = []

        for sensor_path in (
            self.sensor_paths
        ):

            prim = (
                self.stage.GetPrimAtPath(
                    sensor_path
                )
            )

            if (
                not prim
                or
                not prim.IsValid()
            ):

                continue

            current = prim

            while (
                current
                and
                current.IsValid()
            ):

                if current.HasAPI(
                    UsdPhysics.ArticulationRootAPI
                ):

                    root_path = str(
                        current.GetPath()
                    )

                    if root_path not in roots:

                        roots.append(
                            root_path
                        )

                    break

                current = (
                    current.GetParent()
                )

        return roots

    # =========================================================
    # ARTICULATION PATH TEST
    # =========================================================

    def _is_under_ignored_articulation(
        self,
        path,
    ):

        path = str(
            path
        )

        for root_path in (
            self.ignored_articulation_roots
        ):

            if (
                path == root_path
                or
                path.startswith(
                    root_path + "/"
                )
            ):

                return True

        return False

    # =========================================================
    # USD COLLISION ENABLED TEST
    # =========================================================

    def _is_usd_collision_enabled(
        self,
        path,
    ):

        prim = (
            self.stage.GetPrimAtPath(
                path
            )
        )

        if (
            not prim
            or
            not prim.IsValid()
        ):

            return True

        if not prim.HasAPI(
            UsdPhysics.CollisionAPI
        ):

            return True

        collision_api = (
            UsdPhysics.CollisionAPI(
                prim
            )
        )

        attr = (
            collision_api.GetCollisionEnabledAttr()
        )

        if not attr:

            return True

        value = attr.Get()

        if value is None:

            return True

        return bool(
            value
        )

    # =========================================================
    # SHAPE FILTER
    # =========================================================

    def _build_ignored_shape_mask(
        self,
    ):

        if self.model is None:

            raise RuntimeError(
                "Newton model is not available."
            )

        shape_count = int(
            getattr(
                self.model,
                "shape_count",
                0,
            )
        )

        if shape_count <= 0:

            self.gpu_ignored_shape_ids = (
                wp.zeros(
                    1,
                    dtype=wp.int32,
                    device=self.wp_device,
                )
            )

            print(
                "Newton shape filter: "
                "model contains no shapes."
            )

            return

        ignored = np.zeros(
            shape_count,
            dtype=np.int32,
        )

        labels = getattr(
            self.model,
            "shape_label",
            None,
        )

        collision_groups = getattr(
            self.model,
            "shape_collision_group",
            None,
        )

        ignored_count = 0
        usd_disabled_count = 0
        group_disabled_count = 0
        articulation_ignored_count = 0

        collision_group_values = None

        if collision_groups is not None:

            try:

                collision_group_values = (
                    collision_groups.numpy()
                )

            except Exception:

                try:

                    collision_group_values = (
                        np.asarray(
                            collision_groups
                        )
                    )

                except Exception:

                    collision_group_values = None

        for shape_index in range(
            shape_count
        ):

            should_ignore = False

            if (
                collision_group_values is not None
                and
                shape_index
                <
                len(collision_group_values)
            ):

                try:

                    collision_group = int(
                        collision_group_values[
                            shape_index
                        ]
                    )

                    if collision_group == 0:

                        should_ignore = True

                        group_disabled_count += 1

                except Exception:

                    pass

            shape_path = None

            if labels is not None:

                try:

                    if shape_index < len(labels):

                        shape_path = labels[
                            shape_index
                        ]

                except Exception:

                    shape_path = None

            if (
                shape_path is not None
            ):

                try:

                    if not (
                        self._is_usd_collision_enabled(
                            shape_path
                        )
                    ):

                        should_ignore = True

                        usd_disabled_count += 1

                except Exception:

                    pass

            if (
                shape_path is not None
            ):

                try:

                    if (
                        self._is_under_ignored_articulation(
                            shape_path
                        )
                    ):

                        should_ignore = True

                        articulation_ignored_count += 1

                except Exception:

                    pass

            if should_ignore:

                ignored[
                    shape_index
                ] = 1

                ignored_count += 1

        self.gpu_ignored_shape_ids = (
            wp.array(
                ignored,
                dtype=wp.int32,
                device=self.wp_device,
            )
        )

        print(
            "Newton shape filter: "
            f"{ignored_count}/{shape_count} "
            "shape(s) marked as ignored."
        )

        print(
            "  Collision group disabled: "
            f"{group_disabled_count}"
        )

        print(
            "  USD collision disabled: "
            f"{usd_disabled_count}"
        )

        print(
            "  Sensor articulation: "
            f"{articulation_ignored_count}"
        )

        if (
            collision_groups is None
        ):

            print(
                "WARNING: Newton model does not "
                "provide shape_collision_group."
            )

    # =========================================================
    # BUILD RAYCAST BVH
    # =========================================================

    def _build_raycast_bvh(
        self,
    ):

        if self.model is None:

            raise RuntimeError(
                "Newton model is not available."
            )

        if self.state is None:

            raise RuntimeError(
                "Newton state is not available."
            )

        if not hasattr(
            self.model,
            "bvh_build_shapes",
        ):

            raise RuntimeError(
                "Newton model does not provide "
                "bvh_build_shapes()."
            )

        self.model.bvh_build_shapes(
            self.state,
            bvh_constructor="sah",
            shape_flags=COLLIDE_SHAPES_FLAG,
        )

        bvh_shape_count = int(
            getattr(
                self.model,
                "bvh_shape_count_enabled",
                0,
            )
        )

        print(
            "Newton raycast BVH rebuilt: "
            f"{bvh_shape_count} collision shape(s)."
        )

    # =========================================================
    # REFRESH NEWTON REFERENCES
    # =========================================================

    def _refresh_newton_references(
        self,
    ):

        newton_stage = (
            isaac_newton.acquire_stage()
        )

        if newton_stage is None:

            raise RuntimeError(
                "Could not acquire Newton stage."
            )

        newton_model = (
            newton_stage.model
        )

        if newton_model is None:

            raise RuntimeError(
                "Newton stage does not provide "
                "a model."
            )

        model_changed = (
            newton_model is not self.model
        )

        self.newton_stage = (
            newton_stage
        )

        self.model = (
            newton_model
        )

        self.state = getattr(
            newton_stage,
            "state",
            None,
        )

        if self.state is None:

            self.state = getattr(
                newton_stage,
                "state_0",
                None,
            )

        if self.state is None:

            self.state = (
                self.model.state()
            )

        if self.state is None:

            raise RuntimeError(
                "Could not obtain current Newton state."
            )

        if model_changed:

            print(
                "Newton model changed. "
                "Refreshing collision filter and BVH."
            )

            self._build_ignored_shape_mask()

            self._build_raycast_bvh()

    # =========================================================
    # SENSOR RAY DATA
    # =========================================================

    def get_sensor_ray_data(
        self,
        sensor,
    ):

        reading = (
            sensor.get_sensor_reading()
        )

        if reading is None:

            return (
                np.empty(
                    (0, 3),
                    dtype=np.float32,
                ),
                np.empty(
                    (0, 3),
                    dtype=np.float32,
                ),
            )

        ray_origins = getattr(
            reading,
            "ray_origins_world",
            None,
        )

        ray_endpoints = getattr(
            reading,
            "ray_end_points_world",
            None,
        )

        if (
            ray_origins is None
            or
            ray_endpoints is None
        ):

            return (
                np.empty(
                    (0, 3),
                    dtype=np.float32,
                ),
                np.empty(
                    (0, 3),
                    dtype=np.float32,
                ),
            )

        ray_origins = np.asarray(
            ray_origins,
            dtype=np.float32,
        )

        ray_endpoints = np.asarray(
            ray_endpoints,
            dtype=np.float32,
        )

        if ray_origins.size == 0:

            return (
                np.empty(
                    (0, 3),
                    dtype=np.float32,
                ),
                np.empty(
                    (0, 3),
                    dtype=np.float32,
                ),
            )

        ray_origins = (
            ray_origins.reshape(
                -1,
                3,
            )
        )

        ray_endpoints = (
            ray_endpoints.reshape(
                -1,
                3,
            )
        )

        ray_count = min(
            len(ray_origins),
            len(ray_endpoints),
        )

        if ray_count <= 0:

            return (
                np.empty(
                    (0, 3),
                    dtype=np.float32,
                ),
                np.empty(
                    (0, 3),
                    dtype=np.float32,
                ),
            )

        return (
            np.ascontiguousarray(
                ray_origins[
                    :ray_count
                ],
                dtype=np.float32,
            ),
            np.ascontiguousarray(
                ray_endpoints[
                    :ray_count
                ],
                dtype=np.float32,
            ),
        )

    # =========================================================
    # COLLECT RAYS
    # =========================================================

    def _collect_all_rays(
        self,
    ):

        origin_arrays = []
        endpoint_arrays = []

        sensor_ray_counts = []

        for sensor in self.sensors:

            (
                origins,
                endpoints,
            ) = (
                self.get_sensor_ray_data(
                    sensor
                )
            )

            ray_count = len(
                origins
            )

            sensor_ray_counts.append(
                ray_count
            )

            if ray_count <= 0:

                continue

            origin_arrays.append(
                origins
            )

            endpoint_arrays.append(
                endpoints
            )

        if not origin_arrays:

            return (
                np.empty(
                    (0, 3),
                    dtype=np.float32,
                ),
                np.empty(
                    (0, 3),
                    dtype=np.float32,
                ),
                sensor_ray_counts,
            )

        all_origins = np.concatenate(
            origin_arrays,
            axis=0,
        )

        all_endpoints = np.concatenate(
            endpoint_arrays,
            axis=0,
        )

        return (
            np.ascontiguousarray(
                all_origins,
                dtype=np.float32,
            ),
            np.ascontiguousarray(
                all_endpoints,
                dtype=np.float32,
            ),
            sensor_ray_counts,
        )

    # =========================================================
    # RAY BUFFERS
    # =========================================================

    def _ensure_ray_capacity(
        self,
        ray_count,
    ):

        if ray_count <= self.ray_capacity:

            return

        new_capacity = max(
            ray_count,
            max(
                256,
                self.ray_capacity * 2,
            ),
        )

        self.ray_capacity = (
            new_capacity
        )

        self.gpu_ray_origins = (
            wp.zeros(
                new_capacity,
                dtype=wp.vec3,
                device=self.wp_device,
            )
        )

        self.gpu_ray_endpoints = (
            wp.zeros(
                new_capacity,
                dtype=wp.vec3,
                device=self.wp_device,
            )
        )

        self.gpu_distances = (
            wp.zeros(
                new_capacity,
                dtype=wp.float32,
                device=self.wp_device,
            )
        )

        self.gpu_endpoints = (
            wp.zeros(
                new_capacity,
                dtype=wp.vec3,
                device=self.wp_device,
            )
        )

    # =========================================================
    # GPU RAYCAST
    # =========================================================

    def _run_gpu_raycast(
        self,
        ray_origins,
        ray_endpoints,
    ):

        ray_count = len(
            ray_origins
        )

        if ray_count <= 0:

            return (
                np.empty(
                    0,
                    dtype=np.float32,
                ),
                np.empty(
                    (0, 3),
                    dtype=np.float32,
                ),
            )

        if self.model is None:

            raise RuntimeError(
                "Newton model is not available."
            )

        if getattr(
            self.model,
            "bvh_shapes",
            None,
        ) is None:

            raise RuntimeError(
                "Newton shape BVH is unavailable."
            )

        self._ensure_ray_capacity(
            ray_count
        )

        self.gpu_ray_origins[
            :ray_count
        ].assign(
            ray_origins
        )

        self.gpu_ray_endpoints[
            :ray_count
        ].assign(
            ray_endpoints
        )

        wp.launch(
            kernel=raycast_kernel,
            dim=ray_count,
            inputs=[
                self.model.bvh_shapes.id,
                self.model.bvh_shapes_group_roots,
                self.model.bvh_shape_enabled,
                self.model.bvh_shape_world_transforms,
                self.model.shape_type,
                self.model.shape_scale,
                self.model.shape_source_ptr,
                self.gpu_ray_origins[
                    :ray_count
                ],
                self.gpu_ray_endpoints[
                    :ray_count
                ],
                self.gpu_ignored_shape_ids,
                self.gpu_distances[
                    :ray_count
                ],
                self.gpu_endpoints[
                    :ray_count
                ],
            ],
            device=self.wp_device,
        )

        distances = (
            self.gpu_distances[
                :ray_count
            ].numpy()
        )

        endpoints = (
            self.gpu_endpoints[
                :ray_count
            ].numpy()
        )

        return (
            distances,
            endpoints,
        )

    # =========================================================
    # SETUP
    # =========================================================

    def setup(
        self,
    ):

        self.stage = (
            omni.usd
            .get_context()
            .get_stage()
        )

        if self.stage is None:

            raise RuntimeError(
                "Could not get active USD stage."
            )

        cuda_devices = (
            wp.get_cuda_devices()
        )

        if not cuda_devices:

            raise RuntimeError(
                "No CUDA device is available "
                "for Warp."
            )

        self.wp_device = (
            cuda_devices[0]
        )

        # -----------------------------------------------------
        # Sensor articulation discovery.
        # -----------------------------------------------------

        self.ignored_articulation_roots = (
            self._find_sensor_articulation_roots()
        )

        if (
            self.ignored_articulation_roots
        ):

            print(
                "Ignoring articulation root(s):",
                self.ignored_articulation_roots,
            )

        # -----------------------------------------------------
        # Raycast sensors.
        # -----------------------------------------------------

        self.sensors = []

        for sensor_path in (
            self.sensor_paths
        ):

            print(
                f"Attaching to existing RaycastSensor: "
                f"{sensor_path}"
            )

            self.sensors.append(
                RaycastSensor(
                    sensor_path
                )
            )

        print(
            f"Attached to "
            f"{len(self.sensors)} "
            "existing raycast sensor(s)."
        )

        # -----------------------------------------------------
        # Newton stage.
        # -----------------------------------------------------

        self.newton_stage = (
            isaac_newton.acquire_stage()
        )

        if self.newton_stage is None:

            raise RuntimeError(
                "Could not acquire Newton stage."
            )

        self.model = (
            self.newton_stage.model
        )

        if self.model is None:

            raise RuntimeError(
                "Newton stage does not provide "
                "a model."
            )

        shape_count = int(
            getattr(
                self.model,
                "shape_count",
                0,
            )
        )

        world_count = int(
            getattr(
                self.model,
                "world_count",
                1,
            )
        )

        print(
            "Newton model ready: "
            f"{shape_count} shape(s), "
            f"{world_count} world(s)."
        )

        # -----------------------------------------------------
        # Current Newton state.
        # -----------------------------------------------------

        self.state = getattr(
            self.newton_stage,
            "state",
            None,
        )

        if self.state is None:

            self.state = getattr(
                self.newton_stage,
                "state_0",
                None,
            )

        if self.state is None:

            self.state = (
                self.model.state()
            )

        if self.state is None:

            raise RuntimeError(
                "Could not obtain current Newton state."
            )

        # -----------------------------------------------------
        # Build collision filter.
        # -----------------------------------------------------

        self._build_ignored_shape_mask()

        # -----------------------------------------------------
        # Build collision-only raycast BVH.
        # -----------------------------------------------------

        self._build_raycast_bvh()

        return True

    # =========================================================
    # RESULTS
    # =========================================================

    def get_results(
        self,
    ):

        self._refresh_newton_references()

        (
            all_origins,
            all_raw_endpoints,
            sensor_ray_counts,
        ) = (
            self._collect_all_rays()
        )

        if len(all_origins) == 0:

            return (
                [],
                0,
                sensor_ray_counts,
                [],
                [],
            )

        # -----------------------------------------------------
        # Refit the existing BVH.
        # -----------------------------------------------------

        if self.state is not None:

            self.model.bvh_refit_shapes(
                self.state
            )

        # -----------------------------------------------------
        # Perform raycast.
        # -----------------------------------------------------

        (
            distances,
            endpoints,
        ) = (
            self._run_gpu_raycast(
                all_origins,
                all_raw_endpoints,
            )
        )

        return (
            distances.tolist(),
            len(all_origins),
            sensor_ray_counts,
            all_origins.tolist(),
            endpoints.tolist(),
        )

    # =========================================================
    # COMPATIBILITY HELPER
    # =========================================================

    def get_sensor_result(
        self,
        sensor_index,
    ):

        (
            distances,
            total_ray_count,
            sensor_ray_counts,
            origins,
            endpoints,
        ) = (
            self.get_results()
        )

        start = 0

        for index in range(
            sensor_index
        ):

            start += (
                sensor_ray_counts[
                    index
                ]
            )

        count = (
            sensor_ray_counts[
                sensor_index
            ]
        )

        end = (
            start + count
        )

        return (
            distances[start:end],
            origins[start:end],
            endpoints[start:end],
        )

    # =========================================================
    # CLEANUP
    # =========================================================

    def cleanup(
        self,
    ):

        self.sensors = []

        self.sensor_paths = []

        self.ignored_articulation_roots = []

        self.gpu_ignored_shape_ids = None

        self.gpu_ray_origins = None
        self.gpu_ray_endpoints = None

        self.gpu_distances = None
        self.gpu_endpoints = None

        self.ray_capacity = 0

        self.state = None

        self.model = None
        self.newton_stage = None

        self.stage = None
        self.wp_device = None