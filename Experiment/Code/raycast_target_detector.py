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
# GPU RAY PREPARATION
# =============================================================

@wp.kernel
def prepare_rays_kernel(
    origins: wp.array(dtype=wp.vec3),
    endpoints: wp.array(dtype=wp.vec3),
    directions: wp.array(dtype=wp.vec3),
    ray_max_distances: wp.array(dtype=wp.float32),
    valid: wp.array(dtype=wp.int32),
):
    ray_index = wp.tid()

    origin = origins[ray_index]

    endpoint = endpoints[ray_index]

    delta = endpoint - origin

    length_squared = (
        delta[0] * delta[0]
        +
        delta[1] * delta[1]
        +
        delta[2] * delta[2]
    )

    if length_squared < 1.0e-16:

        directions[ray_index] = wp.vec3(
            0.0,
            0.0,
            0.0,
        )

        ray_max_distances[ray_index] = 0.0
        valid[ray_index] = 0

        return

    ray_length = wp.sqrt(
        length_squared
    )

    inverse_length = (
        1.0 / ray_length
    )

    directions[ray_index] = (
        delta * inverse_length
    )

    ray_max_distances[ray_index] = (
        ray_length
    )

    valid[ray_index] = 1


# =============================================================
# GPU FILTERED RAYCAST
# =============================================================
#
# This follows Newton's BVH raycast traversal but adds one
# early shape-ID filter before performing the expensive geometry
# intersection.
#
# The original Newton BVH is used unchanged.
# =============================================================

@wp.kernel
def intersect_filtered_rays_kernel(
    bvh_id: wp.uint64,
    bvh_shapes_group_roots: wp.array(dtype=wp.int32),
    bvh_shape_enabled: wp.array(dtype=wp.uint32),
    shape_transform_world: wp.array(dtype=wp.transform),
    shape_type: wp.array(dtype=wp.int32),
    shape_scale: wp.array(dtype=wp.vec3),
    shape_source_ptr: wp.array(dtype=wp.uint64),

    ray_origins: wp.array(dtype=wp.vec3),
    ray_directions: wp.array(dtype=wp.vec3),
    ray_worlds: wp.array(dtype=wp.int32),

    ignored_shape_ids: wp.array(dtype=wp.int32),

    out_dist: wp.array(dtype=wp.float32),
    out_shape_id: wp.array(dtype=wp.int32),
):
    ray_index = wp.tid()

    origin = ray_origins[
        ray_index
    ]

    direction = ray_directions[
        ray_index
    ]

    min_dist = float(
        MAXVAL
    )

    min_shape_id = wp.int32(
        -1
    )

    # ---------------------------------------------------------
    # Query:
    #
    # 1. The ray's own world.
    # 2. The global world.
    # ---------------------------------------------------------

    for world_pass in range(
        wp.static(2)
    ):

        if world_pass == 0:

            group_id = (
                ray_worlds[
                    ray_index
                ]
            )

        else:

            group_id = (
                bvh_shapes_group_roots.shape[
                    0
                ]
                -
                1
            )

        bvh_root = (
            bvh_shapes_group_roots[
                group_id
            ]
        )

        if bvh_root < 0:
            continue

        query = wp.bvh_query_ray(
            bvh_id,
            origin,
            direction,
            bvh_root,
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

            # -------------------------------------------------
            # Skip ignored articulation shapes before doing
            # any geometry intersection work.
            # -------------------------------------------------

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

            # -------------------------------------------------
            # Mesh geometry.
            # -------------------------------------------------

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
                    hit_normal_local,
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

                if hit_dist >= 0.0:

                    if hit_dist < min_dist:

                        min_dist = (
                            hit_dist
                        )

                        min_shape_id = (
                            shape_id
                        )

            # -------------------------------------------------
            # Analytic geometry.
            # -----------------------------------------------------

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

                if hit_dist >= 0.0:

                    if hit_dist < min_dist:

                        min_dist = (
                            hit_dist
                        )

                        min_shape_id = (
                            shape_id
                        )

    out_dist[
        ray_index
    ] = wp.where(
        min_shape_id < 0,
        -1.0,
        min_dist,
    )

    out_shape_id[
        ray_index
    ] = (
        min_shape_id
    )


# =============================================================
# GPU HIT RESOLUTION
# =============================================================

@wp.kernel
def resolve_hits_kernel(
    origins: wp.array(dtype=wp.vec3),
    directions: wp.array(dtype=wp.vec3),
    ray_max_distances: wp.array(dtype=wp.float32),
    ray_valid: wp.array(dtype=wp.int32),

    hit_distances: wp.array(dtype=wp.float32),
    hit_shape_ids: wp.array(dtype=wp.int32),

    out_distances: wp.array(dtype=wp.float32),
    out_endpoints: wp.array(dtype=wp.vec3),
):
    ray_index = wp.tid()

    origin = origins[
        ray_index
    ]

    direction = directions[
        ray_index
    ]

    max_distance = (
        ray_max_distances[
            ray_index
        ]
    )

    if ray_valid[
        ray_index
    ] == 0:

        out_distances[
            ray_index
        ] = max_distance

        out_endpoints[
            ray_index
        ] = origin

        return

    hit_distance = (
        hit_distances[
            ray_index
        ]
    )

    shape_id = (
        hit_shape_ids[
            ray_index
        ]
    )

    hit_is_valid = 1

    if hit_distance < 0.0:

        hit_is_valid = 0

    if hit_distance > max_distance:

        hit_is_valid = 0

    if shape_id < 0:

        hit_is_valid = 0

    if hit_is_valid == 0:

        out_distances[
            ray_index
        ] = max_distance

        out_endpoints[
            ray_index
        ] = (
            origin
            +
            direction * max_distance
        )

        return

    out_distances[
        ray_index
    ] = hit_distance

    out_endpoints[
        ray_index
    ] = (
        origin
        +
        direction * hit_distance
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
        self.gpu_ray_directions = None
        self.gpu_ray_max_distances = None
        self.gpu_ray_valid = None
        self.gpu_ray_worlds = None

        # -----------------------------------------------------
        # Newton outputs
        # -----------------------------------------------------

        self.gpu_hit_distances = None
        self.gpu_hit_shape_ids = None

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

        ignored_count = 0

        if labels is not None:

            try:

                label_count = min(
                    len(labels),
                    shape_count,
                )

            except TypeError:

                label_count = 0

            for shape_index in range(
                label_count
            ):

                try:

                    label = labels[
                        shape_index
                    ]

                    if (
                        self._is_under_ignored_articulation(
                            label
                        )
                    ):

                        ignored[
                            shape_index
                        ] = 1

                        ignored_count += 1

                except Exception:
                    continue

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

        if (
            self.ignored_articulation_roots
            and
            ignored_count == 0
        ):

            print(
                "WARNING: No Newton shape labels "
                "matched the sensor articulation roots. "
                "Self-shape filtering may be inactive."
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
                "Refreshing shape filter."
            )

            self._build_ignored_shape_mask()

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

        self.gpu_ray_directions = (
            wp.zeros(
                new_capacity,
                dtype=wp.vec3,
                device=self.wp_device,
            )
        )

        self.gpu_ray_max_distances = (
            wp.zeros(
                new_capacity,
                dtype=wp.float32,
                device=self.wp_device,
            )
        )

        self.gpu_ray_valid = (
            wp.zeros(
                new_capacity,
                dtype=wp.int32,
                device=self.wp_device,
            )
        )

        self.gpu_ray_worlds = (
            wp.zeros(
                new_capacity,
                dtype=wp.int32,
                device=self.wp_device,
            )
        )

        self.gpu_hit_distances = (
            wp.zeros(
                new_capacity,
                dtype=wp.float32,
                device=self.wp_device,
            )
        )

        self.gpu_hit_shape_ids = (
            wp.zeros(
                new_capacity,
                dtype=wp.int32,
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

        # -----------------------------------------------------
        # Upload sensor rays.
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # Newton world.
        # -----------------------------------------------------

        self.gpu_ray_worlds[
            :ray_count
        ].fill_(
            0
        )

        # -----------------------------------------------------
        # Normalize rays.
        # -----------------------------------------------------

        wp.launch(
            kernel=prepare_rays_kernel,
            dim=ray_count,
            inputs=[
                self.gpu_ray_origins[
                    :ray_count
                ],
                self.gpu_ray_endpoints[
                    :ray_count
                ],
                self.gpu_ray_directions[
                    :ray_count
                ],
                self.gpu_ray_max_distances[
                    :ray_count
                ],
                self.gpu_ray_valid[
                    :ray_count
                ],
            ],
            device=self.wp_device,
        )

        # -----------------------------------------------------
        # Filtered Newton BVH ray query.
        #
        # Uses the EXISTING Newton BVH.
        # Robot shapes are skipped before geometry intersection.
        # -----------------------------------------------------

        wp.launch(
            kernel=intersect_filtered_rays_kernel,
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
                self.gpu_ray_directions[
                    :ray_count
                ],
                self.gpu_ray_worlds[
                    :ray_count
                ],
                self.gpu_ignored_shape_ids,
                self.gpu_hit_distances[
                    :ray_count
                ],
                self.gpu_hit_shape_ids[
                    :ray_count
                ],
            ],
            device=self.wp_device,
        )

        # -----------------------------------------------------
        # Resolve hit distances and endpoints.
        # -----------------------------------------------------

        wp.launch(
            kernel=resolve_hits_kernel,
            dim=ray_count,
            inputs=[
                self.gpu_ray_origins[
                    :ray_count
                ],
                self.gpu_ray_directions[
                    :ray_count
                ],
                self.gpu_ray_max_distances[
                    :ray_count
                ],
                self.gpu_ray_valid[
                    :ray_count
                ],
                self.gpu_hit_distances[
                    :ray_count
                ],
                self.gpu_hit_shape_ids[
                    :ray_count
                ],
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

        # -----------------------------------------------------
        # Access Newton model.
        # -----------------------------------------------------

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

        bvh_shapes = getattr(
            self.model,
            "bvh_shapes",
            None,
        )

        print(
            "Newton model ready: "
            f"{shape_count} shape(s), "
            f"{world_count} world(s)."
        )

        if bvh_shapes is None:

            raise RuntimeError(
                "Newton model does not contain "
                "a shape BVH."
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

        # -----------------------------------------------------
        # Build self-collision exclusion mask.
        # -----------------------------------------------------

        self._build_ignored_shape_mask()

        return True

    # =========================================================
    # RESULTS
    # =========================================================

    def get_results(
        self,
    ):

        # -----------------------------------------------------
        # Always acquire the current Newton simulation objects.
        #
        # Stop -> Play can replace the Newton model/state.
        # Refreshing these references here prevents this detector
        # from continuing to use the previous simulation model.
        # -----------------------------------------------------

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
        # Refit the EXISTING shape BVH using the current
        # Newton body transforms.
        # -----------------------------------------------------

        if self.state is not None:

            self.model.bvh_refit_shapes(
                self.state
            )

        # -----------------------------------------------------
        # Perform Newton raycast.
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
        self.gpu_ray_directions = None
        self.gpu_ray_max_distances = None
        self.gpu_ray_valid = None
        self.gpu_ray_worlds = None

        self.gpu_hit_distances = None
        self.gpu_hit_shape_ids = None

        self.gpu_distances = None
        self.gpu_endpoints = None

        self.ray_capacity = 0

        self.state = None

        self.model = None
        self.newton_stage = None

        self.stage = None
        self.wp_device = None