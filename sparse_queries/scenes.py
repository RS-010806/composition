"""Procedural dynamic scenes for a small D4RT-style experiment.

Each clip has T frames from a moving camera (translation plus a slow pan), a
ground plane with random tiles, a few static spheres and a few moving spheres.
For every pixel of every frame we keep the object it hits and its 3D world
position, which is enough to answer any D4RT query q = (u, v, t_src, t_tgt, t_cam):
where the surface point seen at pixel (u, v) of frame t_src is at time t_tgt, in
the coordinates of the camera of frame t_cam.

World: y up. Camera: x right, y down, z forward; P_cam = R (X - eye).
"""
import numpy as np

T, H, W = 6, 32, 32
SS = 2                      # supersampling for appearance (ground truth uses pixel centres)
FOV = np.deg2rad(60.0)
FOCAL = (W / 2) / np.tan(FOV / 2)
N_STATIC, N_MOVING = 3, 2
SKY = 255
PALETTE = np.array([[0.90, 0.30, 0.25], [0.20, 0.55, 0.90], [0.95, 0.75, 0.20], [0.30, 0.75, 0.40],
                    [0.70, 0.35, 0.80], [0.95, 0.55, 0.20], [0.20, 0.75, 0.75], [0.85, 0.40, 0.60]])
TILES = np.array([[0.86, 0.84, 0.78], [0.33, 0.36, 0.42], [0.62, 0.66, 0.60], [0.74, 0.62, 0.52],
                  [0.48, 0.52, 0.62], [0.56, 0.48, 0.42]])
LIGHT = np.array([-0.4, 1.0, -0.3]) / np.linalg.norm([-0.4, 1.0, -0.3])


def frame_from_forward(f):
    f = f / np.linalg.norm(f)
    r = np.cross(f, [0.0, 1.0, 0.0])
    r = r / np.linalg.norm(r)
    d = np.cross(f, r)
    return np.stack([r, d, f])            # rows: camera x (right), y (down), z (forward) in world coordinates


def look_at(eye, target):
    return frame_from_forward(target - eye)


def yaw(v, a):
    c, s = np.cos(a), np.sin(a)
    return np.array([c * v[0] + s * v[2], v[1], -s * v[0] + c * v[2]])


def sample_scene(rng):
    """Camera path and object trajectories for one clip."""
    target = np.array([rng.uniform(-0.4, 0.4), 0.0, rng.uniform(-0.4, 0.4)])
    theta0 = rng.uniform(0, 2 * np.pi)
    # like a vehicle-mounted camera: fixed height (4.3) and tilt (about 47 degrees down), so metric depth is
    # recoverable from a single frame; no horizon in view and depths stay below ~16
    eye0 = target + np.array([4.0 * np.cos(theta0), 4.3, 4.0 * np.sin(theta0)])
    R0 = look_at(eye0, target)
    # camera velocity per frame: sideways and forward/back in the ground plane, plus a slow pan
    fwd = np.array([R0[2][0], 0.0, R0[2][2]]) / np.linalg.norm([R0[2][0], R0[2][2]])
    vel = (rng.choice([-1, 1]) * rng.uniform(0.25, 0.5)) * R0[0] + rng.uniform(-0.2, 0.2) * fwd
    pan = rng.uniform(-0.06, 0.06)
    eyes, rots = [], []
    for t in range(T):
        eyes.append(eye0 + vel * t)
        rots.append(frame_from_forward(yaw(R0[2], pan * t)))
    # spheres: static ones first, then moving ones; centres per frame
    radii, colors, starts = [], [], []
    for k in range(N_STATIC + N_MOVING):
        for _ in range(100):
            r = rng.uniform(0.35, 0.65) if k < N_STATIC else rng.uniform(0.55, 0.8)
            lim = 3.5 if k < N_STATIC else 2.2
            p = rng.uniform(-lim, lim, size=2)
            if np.linalg.norm(p) < lim and all(np.linalg.norm(p - q[[0, 2]]) > r + rr + 0.3 for q, rr in zip(starts, radii)):
                break
        radii.append(r)
        starts.append(np.array([p[0], r, p[1]]))
        colors.append(PALETTE[rng.integers(len(PALETTE))])
    centers = np.zeros((T, N_STATIC + N_MOVING, 3))
    for k in range(N_STATIC + N_MOVING):
        if k < N_STATIC:
            centers[:, k] = starts[k]
            continue
        ang, speed = rng.uniform(0, 2 * np.pi), rng.uniform(0.2, 0.4)
        v = speed * np.array([np.cos(ang), 0.0, np.sin(ang)])
        amp, w, ph = rng.uniform(0.0, 0.5), rng.uniform(0.5, 1.2), rng.uniform(0, np.pi)
        for t in range(T):
            centers[t, k] = starts[k] + v * t
            centers[t, k, 1] = radii[k] + amp * abs(np.sin(w * t + ph))
    tiles = rng.integers(len(TILES), size=(64, 64))      # a fresh, non-repeating floor for every clip
    return dict(eye=np.array(eyes), R=np.array(rots), centers=centers, radii=np.array(radii), colors=np.array(colors), tiles=tiles)


def _cast(eye, dirs, centers, radii):
    """Nearest hit along unit rays. Returns distance, object id (0 = ground, 1.. = spheres, SKY) and normal."""
    n = dirs.shape[0]
    dist = np.full(n, np.inf)
    oid = np.full(n, SKY, dtype=np.int32)
    with np.errstate(divide="ignore", invalid="ignore"):
        tp = -eye[1] / dirs[:, 1]
    hit = (dirs[:, 1] < -1e-6) & (tp > 0) & (tp < 40.0)
    dist[hit], oid[hit] = tp[hit], 0
    for k, (c, r) in enumerate(zip(centers, radii)):
        oc = eye - c
        b = dirs @ oc
        disc = b * b - (oc @ oc - r * r)
        ok = disc > 0
        tk = np.where(ok, -b - np.sqrt(np.where(ok, disc, 0.0)), np.inf)
        better = ok & (tk > 1e-4) & (tk < dist)
        dist[better], oid[better] = tk[better], k + 1
    pts = eye + dirs * np.where(np.isfinite(dist), dist, 0.0)[:, None]
    normal = np.zeros_like(pts)
    normal[oid == 0] = [0.0, 1.0, 0.0]
    for k, (c, r) in enumerate(zip(centers, radii)):
        m = oid == k + 1
        normal[m] = (pts[m] - c) / r
    return dist, oid, pts, normal


def _shade(oid, pts, normal, dist, colors, tiles):
    col = np.empty(pts.shape)
    sky = oid == SKY
    col[sky] = [0.62, 0.74, 0.92]
    g = oid == 0
    ix = np.floor(pts[g, 0]).astype(np.int64) % 64
    iz = np.floor(pts[g, 2]).astype(np.int64) % 64
    col[g] = TILES[tiles[ix, iz]]
    s = (oid > 0) & ~sky
    col[s] = colors[oid[s] - 1]
    lit = 0.38 + 0.62 * np.clip(normal @ LIGHT, 0.0, 1.0)
    col[~sky] *= lit[~sky, None]
    fog = np.exp(-np.where(np.isfinite(dist), dist, 0.0) / 30.0)
    col[~sky] = col[~sky] * fog[~sky, None] + np.array([0.62, 0.74, 0.92]) * (1 - fog[~sky, None])
    return col


def _rays(R, res):
    ys, xs = np.mgrid[0:res, 0:res]
    f = FOCAL * res / W
    d = np.stack([(xs + 0.5 - res / 2) / f, (ys + 0.5 - res / 2) / f, np.ones_like(xs, dtype=float)], -1).reshape(-1, 3)
    d = d @ R                             # camera to world: R^T d, written as d @ R for row vectors
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def render(scene):
    """RGB frames (T, H, W, 3) uint8, object ids (T, H, W) uint8 and world points (T, H, W, 3) float32."""
    rgb = np.zeros((T, H, W, 3), np.uint8)
    ids = np.zeros((T, H, W), np.uint8)
    xyz = np.zeros((T, H, W, 3), np.float32)
    for t in range(T):
        eye, R, cen = scene["eye"][t], scene["R"][t], scene["centers"][t]
        d_hi = _rays(R, H * SS)
        dist, oid, pts, nrm = _cast(eye, d_hi, cen, scene["radii"])
        col = _shade(oid, pts, nrm, dist, scene["colors"], scene["tiles"]).reshape(H, SS, W, SS, 3).mean((1, 3))
        rgb[t] = np.clip(col * 255 + 0.5, 0, 255).astype(np.uint8)
        d_lo = _rays(R, H)
        dist, oid, pts, _ = _cast(eye, d_lo, cen, scene["radii"])
        ids[t] = oid.reshape(H, W)
        xyz[t] = np.where((oid != SKY)[:, None], pts, np.nan).reshape(H, W, 3)
    return rgb, ids, xyz


def make_dataset(n, seed):
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ("rgb", "ids", "xyz", "eye", "R", "centers")}
    for _ in range(n):
        s = sample_scene(rng)
        rgb, ids, xyz = render(s)
        for k, v in (("rgb", rgb), ("ids", ids), ("xyz", xyz), ("eye", s["eye"]), ("R", s["R"]), ("centers", s["centers"])):
            out[k].append(v)
    return {k: np.stack(v).astype(np.float32 if k in ("eye", "R", "centers") else v[0].dtype) for k, v in out.items()}


def is_moving(ids):
    return (ids > N_STATIC) & (ids != SKY)


def ground_truth(data, clip, t_src, py, px, t_tgt, t_cam):
    """3D position (camera t_cam coordinates) at time t_tgt of the point seen at pixel (py, px) of frame t_src."""
    X = data["xyz"][clip, t_src, py, px]
    oid = data["ids"][clip, t_src, py, px].astype(np.int64)
    mov = is_moving(oid)
    k = np.clip(oid - 1, 0, N_STATIC + N_MOVING - 1)
    shift = data["centers"][clip, t_tgt, k] - data["centers"][clip, t_src, k]
    X = X + np.where(mov[..., None], shift, 0.0)
    R, eye = data["R"][clip, t_cam], data["eye"][clip, t_cam]
    return np.einsum("...ij,...j->...i", R, X - eye).astype(np.float32), mov


if __name__ == "__main__":
    import time
    t0 = time.time()
    d = make_dataset(20, 0)
    print({k: v.shape for k, v in d.items()}, f"{time.time() - t0:.1f}s for 20 clips")
    print("moving pixel share", is_moving(d["ids"]).mean(), "sky share", (d["ids"] == SKY).mean())
