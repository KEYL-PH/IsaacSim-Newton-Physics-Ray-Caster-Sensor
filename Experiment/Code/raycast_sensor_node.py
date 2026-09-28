import sys
import importlib
from pathlib import Path

import omni.usd
import omni.graph.core as og


# =============================================================
# HELPERS
# =============================================================

def normalize_targets(
    value,
):

    if value is None:
        return []

    if isinstance(
        value,
        str,
    ):

        value = value.strip()

        if not value:
            return []

        return [
            value
        ]

    try:
        values = list(value)

    except TypeError:

        return [
            str(value)
        ]

    targets = []

    for item in values:

        item = str(
            item
        ).strip()

        if item:
            targets.append(
                item
            )

    return targets


# =============================================================
# IMPORT DETECTOR
# =============================================================

def load_detector_class():

    stage = (
        omni.usd
        .get_context()
        .get_stage()
    )

    if stage is None:
        raise RuntimeError(
            "Could not get active USD stage."
        )

    root_layer = (
        stage.GetRootLayer()
    )

    usd_path = (
        root_layer.realPath
    )

    if not usd_path:

        raise RuntimeError(
            "The current USD stage does not "
            "have a valid file path."
        )

    code_dir = (
        Path(usd_path).parent
        / "Code"
    )

    if not code_dir.exists():

        raise RuntimeError(
            f"Code directory not found: "
            f"{code_dir}"
        )

    code_dir_string = str(
        code_dir
    )

    if code_dir_string not in sys.path:
        sys.path.insert(
            0,
            code_dir_string,
        )

    import raycast_target_detector

    raycast_target_detector = (
        importlib.reload(
            raycast_target_detector
        )
    )

    return (
        raycast_target_detector
        .RaycastTargetDetector
    )


# =============================================================
# CREATE DETECTOR
# =============================================================

def create_detector(
    sensor_paths,
):

    if not sensor_paths:

        raise RuntimeError(
            "No target_front raycast sensor "
            "paths were provided."
        )

    RaycastTargetDetector = (
        load_detector_class()
    )

    detector = (
        RaycastTargetDetector(
            sensor_paths=sensor_paths,
            max_range=500.0,
        )
    )

    detector.setup()

    return detector


# =============================================================
# SETUP
# =============================================================

def setup(
    db: og.Database,
):

    state = (
        db.per_instance_state
    )

    state.stage = (
        omni.usd
        .get_context()
        .get_stage()
    )

    if state.stage is None:

        raise RuntimeError(
            "Could not get active USD stage."
        )

    state.target_front = None

    state.sensor_paths = []

    state.target_body_paths = []

    state.detector = None

    state.distance = []

    state.num_rays = 0

    state.beam_origin = []

    state.beam_endpoint = []

    return True


# =============================================================
# COMPUTE
# =============================================================

def compute(
    db: og.Database,
):

    state = (
        db.per_instance_state
    )

    current_sensor_paths = (
        normalize_targets(
            db.inputs.target_front
        )
    )

    # ---------------------------------------------------------
    # No sensor paths.
    # ---------------------------------------------------------

    if not current_sensor_paths:

        if state.detector is not None:

            try:
                state.detector.cleanup()
            except Exception:
                pass

        state.target_front = (
            db.inputs.target_front
        )

        state.sensor_paths = []

        state.target_body_paths = []

        state.detector = None

        state.distance = []

        state.num_rays = 0

        state.beam_origin = []

        state.beam_endpoint = []

        db.outputs.distance = []
        db.outputs.num_rays = 0
        db.outputs.beam_origins = []
        db.outputs.beam_end_points = []

        return True

    # ---------------------------------------------------------
    # Create / recreate detector when sensor paths change.
    # ---------------------------------------------------------

    if (
        state.detector is None
        or
        state.sensor_paths
        != current_sensor_paths
    ):

        if state.detector is not None:

            try:
                state.detector.cleanup()
            except Exception:
                pass

        try:

            state.detector = (
                create_detector(
                    current_sensor_paths
                )
            )

            state.sensor_paths = list(
                current_sensor_paths
            )

            state.target_front = (
                db.inputs.target_front
            )

            state.target_body_paths = []

        except Exception as error:

            print(
                "WARNING: Could not create "
                "raycast target detector: "
                f"{error}"
            )

            state.detector = None

            state.sensor_paths = list(
                current_sensor_paths
            )

            state.target_body_paths = []

            state.distance = []

            state.num_rays = 0

            state.beam_origin = []

            state.beam_endpoint = []

            db.outputs.distance = []
            db.outputs.num_rays = 0
            db.outputs.beam_origins = []
            db.outputs.beam_end_points = []

            return False

    # ---------------------------------------------------------
    # Detector not available.
    # ---------------------------------------------------------

    if state.detector is None:

        db.outputs.distance = []
        db.outputs.num_rays = 0
        db.outputs.beam_origins = []
        db.outputs.beam_end_points = []

        return True

    # ---------------------------------------------------------
    # Perform Newton raycast.
    # ---------------------------------------------------------

    try:

        (
            ray_distances,
            state.num_rays,
            sensor_ray_counts,
            state.beam_origin,
            state.beam_endpoint,
        ) = (
            state.detector.get_results()
        )

    except Exception as error:

        print(
            "WARNING: Could not get "
            "raycast results: "
            f"{error}"
        )

        state.distance = []

        state.num_rays = 0

        state.beam_origin = []

        state.beam_endpoint = []

        db.outputs.distance = []
        db.outputs.num_rays = 0
        db.outputs.beam_origins = []
        db.outputs.beam_end_points = []

        return False

    # ---------------------------------------------------------
    # Convert per-ray distances to one minimum distance
    # per sensor.
    # ---------------------------------------------------------

    state.distance = []

    distance_index = 0

    for ray_count in (
        sensor_ray_counts
    ):

        if ray_count <= 0:

            state.distance.append(
                500.0
            )

            continue

        sensor_distances = (
            ray_distances[
                distance_index:
                distance_index + ray_count
            ]
        )

        if sensor_distances:

            state.distance.append(
                float(
                    min(
                        sensor_distances
                    )
                )
            )

        else:

            state.distance.append(
                500.0
            )

        distance_index += ray_count

    # ---------------------------------------------------------
    # Outputs.
    # ---------------------------------------------------------

    db.outputs.distance = (
        state.distance
    )

    db.outputs.num_rays = (
        state.num_rays
    )

    db.outputs.beam_origins = (
        state.beam_origin
    )

    db.outputs.beam_end_points = (
        state.beam_endpoint
    )

    return True


# =============================================================
# CLEANUP
# =============================================================

def cleanup(
    db: og.Database,
):

    state = (
        db.per_instance_state
    )

    if state.detector is not None:

        try:
            state.detector.cleanup()
        except Exception:
            pass

    state.stage = None

    state.target_front = None

    state.sensor_paths = []

    state.target_body_paths = []

    state.detector = None

    state.distance = []

    state.num_rays = 0

    state.beam_origin = []

    state.beam_endpoint = []