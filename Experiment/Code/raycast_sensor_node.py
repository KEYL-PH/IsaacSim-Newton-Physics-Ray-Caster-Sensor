import importlib
import os
import sys
import time
from pathlib import Path

import omni.graph.core as og
import omni.usd

# =============================================================
# CONFIG
# =============================================================

MAX_RANGE = 500.0
MIN_RANGE = 0.0

# Optional override: set this env var to the folder holding
# raycast_target_detector.py. Otherwise <stage folder>/Code is used.
CODE_DIR_ENV = "RAYCAST_CODE_DIR"

RETRY_INITIAL_S = 2.0
RETRY_MAX_S = 30.0


# =============================================================
# HELPERS
# =============================================================

def normalize_targets(value):
    if value is None:
        return []

    if isinstance(value, str):
        value = value.strip()
        return [value] if value else []

    try:
        values = list(value)
    except TypeError:
        return [str(value)]

    return [str(v).strip() for v in values if str(v).strip()]


def _find_code_dir():
    env = os.environ.get(CODE_DIR_ENV)

    if env and Path(env).is_dir():
        return Path(env)

    stage = omni.usd.get_context().get_stage()

    if stage is None:
        raise RuntimeError("Could not get active USD stage.")

    usd_path = stage.GetRootLayer().realPath

    if usd_path:
        candidate = Path(usd_path).parent / "Code"

        if candidate.is_dir():
            return candidate

    raise RuntimeError(
        "Detector code not found. Save the stage next to a 'Code' folder "
        f"or set the {CODE_DIR_ENV} environment variable."
    )


def load_detector_class():
    code_dir = str(_find_code_dir())

    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)

    import raycast_target_detector

    module = importlib.reload(raycast_target_detector)
    return module.RaycastTargetDetector


def create_detector(sensor_paths):
    detector = load_detector_class()(
        sensor_paths=sensor_paths,
        max_range=MAX_RANGE,
        min_range=MIN_RANGE,
    )
    detector.setup()
    return detector


def _warn_once(state, message):
    if message != state.last_warning:
        print(f"WARNING: {message}")
        state.last_warning = message


def _set_output(state, name, value):
    """Prefer numpy (no list conversion); fall back to lists if rejected."""
    out = db_outputs = state.db_outputs

    if not state.use_lists:
        try:
            setattr(out, name, value)
            return
        except Exception:
            state.use_lists = True
            print("Node outputs rejected numpy arrays; falling back to lists.")

    setattr(out, name, value.tolist() if hasattr(value, "tolist") else value)


def _write_empty(state):
    _set_output(state, "distance", [])
    state.db_outputs.num_rays = 0
    _set_output(state, "beam_origins", [])
    _set_output(state, "beam_end_points", [])


def _dispose_detector(state):
    if state.detector is not None:
        try:
            state.detector.cleanup()
        except Exception:
            pass

    state.detector = None


# =============================================================
# SETUP
# =============================================================

def setup(db: og.Database):
    state = db.per_instance_state

    state.stage = omni.usd.get_context().get_stage()

    if state.stage is None:
        raise RuntimeError("Could not get active USD stage.")

    state.sensor_paths = []
    state.detector = None
    state.retry_at = 0.0
    state.retry_delay = RETRY_INITIAL_S
    state.last_warning = None
    state.use_lists = False
    state.db_outputs = db.outputs

    return True


# =============================================================
# COMPUTE
# =============================================================

def compute(db: og.Database):
    state = db.per_instance_state
    state.db_outputs = db.outputs

    paths = normalize_targets(db.inputs.target_front)

    # ---------------------------------------------------------
    # No sensor paths.
    # ---------------------------------------------------------

    if not paths:
        _dispose_detector(state)
        state.sensor_paths = []
        state.retry_at = 0.0
        state.retry_delay = RETRY_INITIAL_S
        _write_empty(state)
        return True

    # ---------------------------------------------------------
    # Paths changed: reset failure backoff, drop old detector.
    # ---------------------------------------------------------

    if paths != state.sensor_paths:
        _dispose_detector(state)
        state.sensor_paths = list(paths)
        state.retry_at = 0.0
        state.retry_delay = RETRY_INITIAL_S
        state.last_warning = None

    # ---------------------------------------------------------
    # Create detector (with exponential backoff on failure).
    # ---------------------------------------------------------

    if state.detector is None:
        now = time.monotonic()

        if now < state.retry_at:
            _write_empty(state)
            return True

        try:
            state.detector = create_detector(paths)
            state.retry_delay = RETRY_INITIAL_S
            state.last_warning = None

        except Exception as error:
            _warn_once(state, f"Could not create raycast target detector: {error}")
            state.retry_at = now + state.retry_delay
            state.retry_delay = min(state.retry_delay * 2.0, RETRY_MAX_S)
            _dispose_detector(state)
            _write_empty(state)
            return False

    # ---------------------------------------------------------
    # Raycast.
    # ---------------------------------------------------------

    try:
        result = state.detector.get_results()

    except Exception as error:
        _warn_once(state, f"Could not get raycast results: {error}")

        # A broken detector (e.g. model replaced mid-run) is rebuilt with backoff.
        _dispose_detector(state)
        state.retry_at = time.monotonic() + state.retry_delay
        state.retry_delay = min(state.retry_delay * 2.0, RETRY_MAX_S)
        _write_empty(state)
        return False

    state.last_warning = None

    _set_output(state, "distance", result.sensor_min)
    db.outputs.num_rays = int(result.num_rays)
    _set_output(state, "beam_origins", result.origins)
    _set_output(state, "beam_end_points", result.endpoints)

    return True


# =============================================================
# CLEANUP
# =============================================================

def cleanup(db: og.Database):
    state = db.per_instance_state

    _dispose_detector(state)

    state.stage = None
    state.sensor_paths = []
    state.db_outputs = None
