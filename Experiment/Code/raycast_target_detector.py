"""Newton raycast target detector for Isaac Sim 6.1 (Newton 1.5.x).

Changes vs. the first version:
  * Rays are traced against THEIR world's BVH group + the global group.
  * Per-sensor minimum distance is reduced on the GPU (atomic_min).
  * One BVH build / refit per model per app update, shared by all detectors.
  * Self-filtering uses joint connectivity (not only USD path prefixes).
  * Zero-length rays are invalid (excluded from the per-sensor minimum).
  * max_range / min_range are honoured.
  * Public Newton imports first, `_src` only as a fallback.
  * Optional hit shape ids and normals.
"""

import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import omni.usd
import warp as wp

import newton
import isaacsim.physics.newton as isaac_newton

from pxr import UsdPhysics
from isaacsim.sensors.experimental.physics import RaycastSensor

try:
    from newton.geometry.raycast import (
        map_ray_to_local,
        ray_intersect_mesh,
        ray_intersect_shape,
    )
except ImportError:
    from newton._src.geometry.raycast import (
        map_ray_to_local,
        ray_intersect_mesh,
        ray_intersect_shape,
    )

try:
    from newton import GeoType
except ImportError:
    from newton._src.geometry.types import GeoType

try:
    from newton import ShapeFlags

    COLLIDE_SHAPES_FLAG = int(ShapeFlags.COLLIDE_SHAPES)
except ImportError:
    COLLIDE_SHAPES_FLAG = 2

wp.init()

TESTED_NEWTON_PREFIX = "1.5"
_BIG = 1.0e30

# Survives importlib.reload() so reloading the module does not orphan state.
_BVH_ENTRIES = globals().get("_BVH_ENTRIES", {})
_logged_once = globals().get("_logged_once", set())


def _log_once(key, message):
    if key in _logged_once:
        return
    _logged_once.add(key)
    print(message)


# =============================================================
# GPU RAYCAST
# =============================================================

@wp.func
def _trace_group(
    bvh_id: wp.uint64,
    root: wp.int32,
    shape_enabled: wp.array(dtype=wp.uint32),
    shape_xf: wp.array(dtype=wp.transform),
    shape_type: wp.array(dtype=wp.int32),
    shape_scale: wp.array(dtype=wp.vec3),
    shape_src: wp.array(dtype=wp.uint64),
    ignored: wp.array(dtype=wp.int32),
    origin: wp.vec3,
    direction: wp.vec3,
    ray_len: float,
    min_range: float,
    best_in: float,
    shape_in: wp.int32,
    normal_in: wp.vec3,
):
    best = float(best_in)
    best_shape = wp.int32(shape_in)
    best_normal = wp.vec3(normal_in[0], normal_in[1], normal_in[2])

    query = wp.bvh_query_ray(bvh_id, origin, direction, root)
    bvh_idx = wp.int32(0)

    while wp.bvh_query_next(query, bvh_idx, best):
        sid = wp.int32(shape_enabled[bvh_idx])

        if ignored[sid] != 0:
            continue

        gt = shape_type[sid]
        hit = float(-1.0)
        hit_normal = wp.vec3(0.0, 0.0, 0.0)

        if gt == GeoType.MESH or gt == GeoType.CONVEX_MESH or gt == GeoType.HFIELD:
            xf = shape_xf[sid]
            o_l, d_l = map_ray_to_local(xf, origin, direction, shape_scale[sid])
            h, n_l, _u, _v, _f = ray_intersect_mesh(
                o_l, d_l, shape_scale[sid], shape_src[sid], False, best
            )
            hit = h
            hit_normal = wp.transform_vector(xf, n_l)
        else:
            h, n_w = ray_intersect_shape(
                shape_xf[sid], shape_scale[sid], gt, origin, direction, False
            )
            hit = h
            hit_normal = n_w

        if hit >= 0.0 and hit >= min_range and hit <= ray_len and hit < best:
            best = hit
            best_shape = sid
            best_normal = hit_normal

    return best, best_shape, best_normal


@wp.kernel
def raycast_kernel(
    bvh_id: wp.uint64,
    group_roots: wp.array(dtype=wp.int32),
    shape_enabled: wp.array(dtype=wp.uint32),
    shape_xf: wp.array(dtype=wp.transform),
    shape_type: wp.array(dtype=wp.int32),
    shape_scale: wp.array(dtype=wp.vec3),
    shape_src: wp.array(dtype=wp.uint64),
    ignored: wp.array(dtype=wp.int32),
    ray_origins: wp.array(dtype=wp.vec3),
    ray_endpoints: wp.array(dtype=wp.vec3),
    ray_world: wp.array(dtype=wp.int32),
    ray_sensor: wp.array(dtype=wp.int32),
    max_range: float,
    min_range: float,
    out_dist: wp.array(dtype=wp.float32),
    out_end: wp.array(dtype=wp.vec3),
    out_shape: wp.array(dtype=wp.int32),
    out_normal: wp.array(dtype=wp.vec3),
    sensor_min: wp.array(dtype=wp.float32),
):
    i = wp.tid()

    origin = ray_origins[i]
    delta = ray_endpoints[i] - origin
    len_sq = wp.dot(delta, delta)

    # Degenerate ray: invalid, excluded from the per-sensor minimum.
    if len_sq < 1.0e-16:
        out_dist[i] = -1.0
        out_end[i] = origin
        out_shape[i] = -1
        out_normal[i] = wp.vec3(0.0, 0.0, 0.0)
        return

    full_len = wp.sqrt(len_sq)
    direction = delta / full_len

    ray_len = float(full_len)
    if max_range > 0.0:
        ray_len = wp.min(full_len, max_range)

    best = float(ray_len)
    shape = wp.int32(-1)
    normal = wp.vec3(0.0, 0.0, 0.0)

    # Group layout: [world 0, world 1, ..., world N-1, global].
    n_groups = group_roots.shape[0]
    w = ray_world[i]

    if w >= 0 and w < n_groups - 1:
        w_root = group_roots[w]
        if w_root >= 0:
            d1, s1, n1 = _trace_group(
                bvh_id, w_root, shape_enabled, shape_xf, shape_type,
                shape_scale, shape_src, ignored, origin, direction,
                ray_len, min_range, best, shape, normal,
            )
            best = d1
            shape = s1
            normal = n1

    g_root = group_roots[n_groups - 1]
    if g_root >= 0:
        d2, s2, n2 = _trace_group(
            bvh_id, g_root, shape_enabled, shape_xf, shape_type,
            shape_scale, shape_src, ignored, origin, direction,
            ray_len, min_range, best, shape, normal,
        )
        best = d2
        shape = s2
        normal = n2

    dist = float(ray_len)
    if shape >= 0:
        dist = best

    out_dist[i] = dist
    out_end[i] = origin + direction * dist
    out_shape[i] = shape
    out_normal[i] = normal

    wp.atomic_min(sensor_min, ray_sensor[i], dist)


# =============================================================
# SHARED BVH MANAGEMENT (one build / refit per model per update)
# =============================================================

def _frame_token():
    try:
        import omni.kit.app

        return omni.kit.app.get_app().get_update_number()
    except Exception:
        return None


def _bvh_acquire(model, state):
    entry = _BVH_ENTRIES.get(id(model))

    if entry is None or entry["model"] is not model:
        model.bvh_build_shapes(
            state, bvh_constructor="sah", shape_flags=COLLIDE_SHAPES_FLAG
        )
        entry = {"model": model, "refs": 0, "frame": None}
        _BVH_ENTRIES[id(model)] = entry
        count = int(getattr(model, "bvh_shape_count_enabled", 0))
        print(f"Newton raycast BVH built: {count} collision shape(s).")

    entry["refs"] += 1


def _bvh_release(model):
    if model is None:
        return

    entry = _BVH_ENTRIES.get(id(model))

    if entry is None or entry["model"] is not model:
        return

    entry["refs"] -= 1

    if entry["refs"] <= 0:
        _BVH_ENTRIES.pop(id(model), None)


def _bvh_refit(model, state):
    entry = _BVH_ENTRIES.get(id(model))
    token = _frame_token()

    if entry is not None and token is not None and entry["frame"] == token:
        return

    model.bvh_refit_shapes(state)

    if entry is not None:
        entry["frame"] = token


# =============================================================
# RESULT
# =============================================================

@dataclass
class RaycastResult:
    sensor_min: np.ndarray            # (S,) min distance per sensor
    num_rays: int
    sensor_ray_counts: list
    origins: np.ndarray               # (N, 3)
    endpoints: np.ndarray             # (N, 3)
    per_ray: np.ndarray = None        # (N,) distances, -1 = invalid ray
    hit_shape_ids: np.ndarray = None  # (N,) -1 = miss
    hit_normals: np.ndarray = None    # (N, 3)


def _empty3():
    return np.empty((0, 3), dtype=np.float32)


# =============================================================
# DETECTOR
# =============================================================

class RaycastTargetDetector:

    def __init__(
        self,
        sensor_paths,
        max_range=500.0,
        min_range=0.0,
        ignore_collision_group_zero=True,
        filter_refresh_interval=0,
    ):
        self.sensor_paths = list(sensor_paths)
        self.max_range = float(max_range)
        self.min_range = float(min_range)
        self.ignore_group_zero = bool(ignore_collision_group_zero)
        self.filter_refresh_interval = int(filter_refresh_interval)

        self.stage = None
        self.sensors = []
        self.newton_stage = None
        self.model = None
        self.state = None
        self.state_source = None
        self.wp_device = None

        self.sensor_worlds = np.zeros(len(self.sensor_paths), dtype=np.int32)
        self.gpu_ignored_shape_ids = None

        self._usd_cache = {}
        self._calls = 0

        self.ray_capacity = 0
        self._meta_key = None
        self.gpu_sensor_min = None

    # ---------------------------------------------------------
    # USD helpers
    # ---------------------------------------------------------

    def _usd_collision_enabled(self, path):
        """Nearest ancestor-or-self with CollisionAPI decides; default True."""
        cache = self._usd_cache

        if path in cache:
            return cache[path]

        chain = []
        result = True
        current = path

        while True:
            if current in cache:
                result = cache[current]
                break

            chain.append(current)

            try:
                prim = self.stage.GetPrimAtPath(current)
            except Exception:
                break

            if not prim or not prim.IsValid():
                break

            if prim.HasAPI(UsdPhysics.CollisionAPI):
                attr = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr()
                value = attr.Get() if attr else None

                if value is not None:
                    result = bool(value)
                    break

            if "/" not in current.strip("/"):
                break

            current = current.rsplit("/", 1)[0]

            if not current:
                break

        for item in chain:
            cache[item] = result

        return result

    def _find_articulation_roots(self):
        roots = []

        for sensor_path in self.sensor_paths:
            prim = self.stage.GetPrimAtPath(sensor_path)
            current = prim

            while current and current.IsValid():
                if current.HasAPI(UsdPhysics.ArticulationRootAPI):
                    root = str(current.GetPath())

                    if root not in roots:
                        roots.append(root)

                    break

                current = current.GetParent()

        return roots

    @staticmethod
    def _under(path, root):
        return path == root or path.startswith(root + "/")

    # ---------------------------------------------------------
    # Articulation membership via joint connectivity
    # ---------------------------------------------------------

    def _articulation_body_set(self):
        model = self.model
        labels = getattr(model, "body_label", None)
        j_parent = getattr(model, "joint_parent", None)
        j_child = getattr(model, "joint_child", None)

        if labels is None or j_parent is None or j_child is None:
            _log_once(
                "no_joint_graph",
                "WARNING: model has no body_label/joint_parent/joint_child; "
                "self-filtering falls back to USD path prefixes only.",
            )
            return set()

        labels = [str(x) for x in labels]
        parent = j_parent.numpy()
        child = j_child.numpy()

        adjacency = {}

        for p, c in zip(parent.tolist(), child.tolist()):
            if p < 0 or c < 0:
                continue

            adjacency.setdefault(p, []).append(c)
            adjacency.setdefault(c, []).append(p)

        seeds = set()

        for sensor_path in self.sensor_paths:
            best, best_len = None, -1

            for index, label in enumerate(labels):
                if self._under(sensor_path, label) and len(label) > best_len:
                    best, best_len = index, len(label)

            if best is not None:
                seeds.add(best)

        members = set(seeds)
        queue = deque(seeds)

        while queue:
            body = queue.popleft()

            for other in adjacency.get(body, ()):
                if other not in members:
                    members.add(other)
                    queue.append(other)

        return members

    # ---------------------------------------------------------
    # Shape filter
    # ---------------------------------------------------------

    def _build_ignored_shape_mask(self):
        model = self.model
        shape_count = int(getattr(model, "shape_count", 0))

        if shape_count <= 0:
            self.gpu_ignored_shape_ids = wp.zeros(
                1, dtype=wp.int32, device=self.wp_device
            )
            print("Newton shape filter: model contains no shapes.")
            return

        ignored = np.zeros(shape_count, dtype=np.int32)

        labels = [str(x) for x in getattr(model, "shape_label", [])]
        flags = model.shape_flags.numpy()
        candidates = np.nonzero((flags & COLLIDE_SHAPES_FLAG) != 0)[0]

        groups = getattr(model, "shape_collision_group", None)
        groups = groups.numpy() if groups is not None else None

        shape_body = getattr(model, "shape_body", None)
        shape_body = shape_body.numpy() if shape_body is not None else None

        roots = self._find_articulation_roots()
        body_set = self._articulation_body_set()

        n_group = n_usd = n_art = 0

        for index in candidates.tolist():
            reason = None

            if self.ignore_group_zero and groups is not None and index < len(groups):
                if int(groups[index]) == 0:
                    reason = "group"

            path = labels[index] if index < len(labels) else None

            if reason is None and path is not None:
                try:
                    if not self._usd_collision_enabled(path):
                        reason = "usd"
                except Exception:
                    pass

            if reason is None:
                in_art = False

                if shape_body is not None and int(shape_body[index]) in body_set:
                    in_art = True
                elif path is not None and any(self._under(path, r) for r in roots):
                    in_art = True

                if in_art:
                    reason = "art"

            if reason is not None:
                ignored[index] = 1
                n_group += reason == "group"
                n_usd += reason == "usd"
                n_art += reason == "art"

        self.gpu_ignored_shape_ids = wp.array(
            ignored, dtype=wp.int32, device=self.wp_device
        )

        print(
            f"Newton shape filter: {int(ignored.sum())}/{len(candidates)} "
            f"collision shape(s) ignored "
            f"(group0={n_group}, usd_disabled={n_usd}, sensor_articulation={n_art})."
        )

        if groups is None and self.ignore_group_zero:
            _log_once(
                "no_collision_group",
                "WARNING: model has no shape_collision_group.",
            )

    def invalidate_filters(self):
        """Call after toggling collision enabled / reparenting at runtime."""
        self._usd_cache.clear()

        if self.model is not None:
            self._build_ignored_shape_mask()
            self._resolve_sensor_worlds()

    # ---------------------------------------------------------
    # World resolution per sensor
    # ---------------------------------------------------------

    def _resolve_sensor_worlds(self):
        model = self.model
        count = len(self.sensor_paths)
        worlds = np.zeros(count, dtype=np.int32)
        world_count = int(getattr(model, "world_count", 1))
        shape_world = getattr(model, "shape_world", None)

        if world_count <= 1:
            self.sensor_worlds = worlds
            self._meta_key = None
            return

        if shape_world is None:
            _log_once(
                "no_shape_world",
                "WARNING: model has no shape_world; all sensors use world 0.",
            )
            self.sensor_worlds = worlds
            self._meta_key = None
            return

        sw = shape_world.numpy()
        labels = [str(x) for x in getattr(model, "shape_label", [])]
        label_parts = [label.strip("/").split("/") for label in labels]

        for k, sensor_path in enumerate(self.sensor_paths):
            sensor_parts = sensor_path.strip("/").split("/")
            best_len, best_worlds = 0, set()

            for index, parts in enumerate(label_parts):
                if index >= len(sw) or int(sw[index]) < 0:
                    continue

                common = 0

                for a, b in zip(sensor_parts, parts):
                    if a != b:
                        break

                    common += 1

                if common > best_len:
                    best_len, best_worlds = common, {int(sw[index])}
                elif common == best_len and common > 0:
                    best_worlds.add(int(sw[index]))

            if best_len >= 2 and len(best_worlds) == 1:
                worlds[k] = next(iter(best_worlds))
            else:
                worlds[k] = min(best_worlds) if best_worlds else 0
                print(
                    f"WARNING: could not uniquely resolve the world of "
                    f"{sensor_path} (matched {sorted(best_worlds)}); "
                    f"using world {int(worlds[k])}."
                )

        self.sensor_worlds = worlds
        self._meta_key = None
        print(f"Sensor worlds: {dict(zip(self.sensor_paths, worlds.tolist()))}")

    # ---------------------------------------------------------
    # Newton references
    # ---------------------------------------------------------

    def _get_state(self):
        for name in ("state", "state_0"):
            state = getattr(self.newton_stage, name, None)

            if state is not None:
                self.state_source = name
                return state

        _log_once(
            "fallback_state",
            "WARNING: Newton stage exposes no live state; using a fresh "
            "model.state() (initial pose). Raycasts will NOT follow moving "
            "bodies. Check the attribute name used by this Isaac Sim build.",
        )
        self.state_source = "model.state() fallback"
        return self.model.state()

    def _refresh_newton_references(self, force=False):
        newton_stage = isaac_newton.acquire_stage()

        if newton_stage is None:
            raise RuntimeError("Could not acquire Newton stage.")

        model = newton_stage.model

        if model is None:
            raise RuntimeError("Newton stage does not provide a model.")

        changed = force or (model is not self.model)

        self.newton_stage = newton_stage
        old_model = self.model
        self.model = model
        self.state = self._get_state()

        if not changed:
            return

        print("Newton model (re)bound. Refreshing filter, worlds and BVH.")

        if old_model is not None and old_model is not model:
            _bvh_release(old_model)

        if not hasattr(model, "bvh_build_shapes"):
            raise RuntimeError("Newton model does not provide bvh_build_shapes().")

        _bvh_acquire(model, self.state)
        self._usd_cache.clear()
        self._build_ignored_shape_mask()
        self._resolve_sensor_worlds()

        roots = getattr(model, "bvh_shapes_group_roots", None)
        world_count = int(getattr(model, "world_count", 1))

        if roots is not None and roots.shape[0] != world_count + 1:
            print(
                f"WARNING: bvh_shapes_group_roots has {roots.shape[0]} entries "
                f"for {world_count} world(s); expected world_count + 1. "
                "Per-world raycasting may be wrong."
            )

    # ---------------------------------------------------------
    # Sensor ray data
    # ---------------------------------------------------------

    def get_sensor_ray_data(self, sensor):
        reading = sensor.get_sensor_reading()

        if reading is None:
            return _empty3(), _empty3()

        origins = getattr(reading, "ray_origins_world", None)
        endpoints = getattr(reading, "ray_end_points_world", None)

        if origins is None or endpoints is None:
            return _empty3(), _empty3()

        origins = np.asarray(origins, dtype=np.float32)
        endpoints = np.asarray(endpoints, dtype=np.float32)

        if origins.size == 0:
            return _empty3(), _empty3()

        origins = origins.reshape(-1, 3)
        endpoints = endpoints.reshape(-1, 3)
        count = min(len(origins), len(endpoints))

        if count <= 0:
            return _empty3(), _empty3()

        return (
            np.ascontiguousarray(origins[:count]),
            np.ascontiguousarray(endpoints[:count]),
        )

    def _collect_all_rays(self):
        origin_arrays, endpoint_arrays, counts = [], [], []

        for sensor in self.sensors:
            origins, endpoints = self.get_sensor_ray_data(sensor)
            counts.append(len(origins))

            if len(origins):
                origin_arrays.append(origins)
                endpoint_arrays.append(endpoints)

        if not origin_arrays:
            return _empty3(), _empty3(), counts

        return (
            np.ascontiguousarray(np.concatenate(origin_arrays), dtype=np.float32),
            np.ascontiguousarray(np.concatenate(endpoint_arrays), dtype=np.float32),
            counts,
        )

    # ---------------------------------------------------------
    # Buffers
    # ---------------------------------------------------------

    def _ensure_ray_capacity(self, ray_count):
        if ray_count <= self.ray_capacity:
            return

        cap = max(ray_count, 256, self.ray_capacity * 2)
        dev = self.wp_device
        self.ray_capacity = cap
        self._meta_key = None

        self.gpu_ray_origins = wp.zeros(cap, dtype=wp.vec3, device=dev)
        self.gpu_ray_endpoints = wp.zeros(cap, dtype=wp.vec3, device=dev)
        self.gpu_ray_world = wp.zeros(cap, dtype=wp.int32, device=dev)
        self.gpu_ray_sensor = wp.zeros(cap, dtype=wp.int32, device=dev)
        self.gpu_distances = wp.zeros(cap, dtype=wp.float32, device=dev)
        self.gpu_endpoints = wp.zeros(cap, dtype=wp.vec3, device=dev)
        self.gpu_shape_ids = wp.zeros(cap, dtype=wp.int32, device=dev)
        self.gpu_normals = wp.zeros(cap, dtype=wp.vec3, device=dev)

    def _ensure_ray_meta(self, counts):
        key = (tuple(counts), tuple(self.sensor_worlds.tolist()))

        if key == self._meta_key:
            return

        n = int(sum(counts))
        ids = np.repeat(np.arange(len(counts), dtype=np.int32), counts)
        worlds = np.repeat(self.sensor_worlds[: len(counts)], counts).astype(np.int32)

        self.gpu_ray_sensor[:n].assign(ids)
        self.gpu_ray_world[:n].assign(worlds)
        self._meta_key = key

    # ---------------------------------------------------------
    # Setup
    # ---------------------------------------------------------

    def setup(self):
        version = str(getattr(newton, "__version__", "unknown"))

        if not version.startswith(TESTED_NEWTON_PREFIX):
            print(
                f"WARNING: tested with Newton {TESTED_NEWTON_PREFIX}.x, "
                f"found {version}. Private/raycast APIs may differ."
            )

        self.stage = omni.usd.get_context().get_stage()

        if self.stage is None:
            raise RuntimeError("Could not get active USD stage.")

        cuda_devices = wp.get_cuda_devices()

        if not cuda_devices:
            raise RuntimeError("No CUDA device is available for Warp.")

        self.wp_device = cuda_devices[0]

        self.sensors = []

        for sensor_path in self.sensor_paths:
            print(f"Attaching to existing RaycastSensor: {sensor_path}")
            self.sensors.append(RaycastSensor(sensor_path))

        print(f"Attached to {len(self.sensors)} existing raycast sensor(s).")

        self.gpu_sensor_min = wp.zeros(
            max(1, len(self.sensors)), dtype=wp.float32, device=self.wp_device
        )

        self._refresh_newton_references(force=True)

        print(
            f"Newton model ready: {int(getattr(self.model, 'shape_count', 0))} "
            f"shape(s), {int(getattr(self.model, 'world_count', 1))} world(s). "
            f"State source: {self.state_source}."
        )

        return True

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    def get_results(self, per_ray=False, extras=False):
        """Trace all rays.

        per_ray: also download per-ray distances (-1 marks invalid rays).
        extras:  also download hit shape ids and world-space normals.
        """
        self._refresh_newton_references()

        self._calls += 1

        if self.filter_refresh_interval and self._calls % self.filter_refresh_interval == 0:
            self.invalidate_filters()

        origins, endpoints, counts = self._collect_all_rays()
        n = len(origins)
        sensor_min = np.full(len(self.sensors), self.max_range, dtype=np.float32)

        if n == 0:
            return RaycastResult(sensor_min, 0, counts, _empty3(), _empty3())

        model = self.model

        if getattr(model, "bvh_shapes", None) is None:
            raise RuntimeError("Newton shape BVH is unavailable.")

        _bvh_refit(model, self.state)

        self._ensure_ray_capacity(n)
        self._ensure_ray_meta(counts)

        self.gpu_ray_origins[:n].assign(origins)
        self.gpu_ray_endpoints[:n].assign(endpoints)
        self.gpu_sensor_min.fill_(_BIG)

        wp.launch(
            kernel=raycast_kernel,
            dim=n,
            inputs=[
                model.bvh_shapes.id,
                model.bvh_shapes_group_roots,
                model.bvh_shape_enabled,
                model.bvh_shape_world_transforms,
                model.shape_type,
                model.shape_scale,
                model.shape_source_ptr,
                self.gpu_ignored_shape_ids,
                self.gpu_ray_origins[:n],
                self.gpu_ray_endpoints[:n],
                self.gpu_ray_world[:n],
                self.gpu_ray_sensor[:n],
                self.max_range,
                self.min_range,
                self.gpu_distances[:n],
                self.gpu_endpoints[:n],
                self.gpu_shape_ids[:n],
                self.gpu_normals[:n],
                self.gpu_sensor_min,
            ],
            device=self.wp_device,
        )

        mins = self.gpu_sensor_min.numpy()[: len(self.sensors)]
        sensor_min = np.where(mins >= 0.5 * _BIG, self.max_range, mins).astype(np.float32)

        result = RaycastResult(
            sensor_min=sensor_min,
            num_rays=n,
            sensor_ray_counts=counts,
            origins=origins,
            endpoints=self.gpu_endpoints[:n].numpy(),
        )

        if per_ray:
            result.per_ray = self.gpu_distances[:n].numpy()

        if extras:
            result.hit_shape_ids = self.gpu_shape_ids[:n].numpy()
            result.hit_normals = self.gpu_normals[:n].numpy()

        return result

    def shape_label(self, shape_id):
        labels = getattr(self.model, "shape_label", None)

        if labels is None or shape_id < 0 or shape_id >= len(labels):
            return ""

        return str(labels[shape_id])

    # Compatibility with the first version.
    def get_sensor_result(self, sensor_index):
        result = self.get_results(per_ray=True)
        start = sum(result.sensor_ray_counts[:sensor_index])
        end = start + result.sensor_ray_counts[sensor_index]

        return (
            result.per_ray[start:end].tolist(),
            result.origins[start:end].tolist(),
            result.endpoints[start:end].tolist(),
        )

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------

    def cleanup(self):
        _bvh_release(self.model)

        self.sensors = []
        self.sensor_paths = []
        self.gpu_ignored_shape_ids = None
        self.gpu_sensor_min = None

        for name in (
            "gpu_ray_origins", "gpu_ray_endpoints", "gpu_ray_world",
            "gpu_ray_sensor", "gpu_distances", "gpu_endpoints",
            "gpu_shape_ids", "gpu_normals",
        ):
            setattr(self, name, None)

        self.ray_capacity = 0
        self._meta_key = None
        self._usd_cache = {}
        self.state = None
        self.model = None
        self.newton_stage = None
        self.stage = None
        self.wp_device = None
