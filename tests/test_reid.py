"""Re-ID: crop sampling, track descriptors, association rules, metrics, the online service."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from lgep.reid import (AssociationConfig, CameraTopology, IdentityManager, SamplingConfig, TrackFeatureBank,
                       TrackletRecord, TransitWindow)
from lgep.reid.features import occlusion
from lgep.reid.metrics import merge_cameras, retrieval_metrics, same_vs_different_auroc
from lgep.reid.service import ReIDService


def unit(*v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


def tracklet(key, t0, t1, emb, phrase="person", samples=3):
    cam, lid = key.split("/")
    r = TrackletRecord(key, cam, int(lid), phrase, int(t0 * 10), int(t1 * 10), t0, t1)
    r.embeddings = [np.asarray(emb, np.float32)] * samples
    return r


def track(local_id, box, score=0.9, cam="cam01", phrase="person"):
    return SimpleNamespace(key=f"{cam}/{local_id}", local_id=local_id, box_xyxy=box, score=score, phrase=phrase)


# ------------------------------------------------------------------------------------------- sampling
def test_occlusion_is_the_largest_covered_fraction():
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [100, 100, 110, 110]], float)
    assert occlusion(b).tolist() == [0.5, 0.5, 0.0]


def test_sampling_rules():
    bank = TrackFeatureBank(SamplingConfig(min_height_px=50, min_score=0.5, border_margin_px=4, max_occlusion=0.3,
                                           every_n_frames=5))
    tracks = [track(1, (100, 100, 140, 200)),               # good
              track(2, (300, 100, 340, 130)),               # too small (30 px)
              track(3, (500, 100, 540, 200), score=0.3),    # low score
              track(4, (0, 100, 40, 200)),                  # touches the left border
              track(5, (700, 100, 740, 200)), track(6, (710, 100, 750, 200))]  # 75 % overlap with each other
    want = bank.observe("cam01", 1, 0.0, tracks, (1920, 1080))
    assert [t.local_id for t in want] == [1]
    bank.add("cam01/1", 1, unit(1, 0))
    assert bank.observe("cam01", 3, 0.1, tracks[:1], (1920, 1080)) == []  # 2 frames later: too soon
    assert [t.local_id for t in bank.observe("cam01", 6, 0.2, tracks[:1], (1920, 1080))] == [1]
    rec = bank.tracks["cam01/1"]
    assert (rec.first_frame, rec.last_frame, rec.frames_seen) == (1, 6, 3)


def test_mean_centring_removes_the_common_component():
    bank = TrackFeatureBank(SamplingConfig(center=True))
    common = np.array([3.0, 0, 0], np.float32)  # every crop of this camera shares it
    bank.observe("cam01", 1, 0.0, [track(1, (10, 10, 50, 200)), track(2, (300, 10, 340, 200))], (640, 480))
    for f in range(1, 4):
        bank.add("cam01/1", f, unit(*(common + [0, 1, 0])))
        bank.add("cam01/2", f, unit(*(common + [0, 0, 1])))
    raw_1, raw_2 = bank.tracks["cam01/1"].descriptor, bank.tracks["cam01/2"].descriptor
    a, b = bank.pop("cam01/1"), bank.pop("cam01/2")
    assert float(raw_1 @ raw_2) > 0.9  # dominated by the shared component
    assert float(a.descriptor @ b.descriptor) < -0.9  # after centring: opposite directions
    assert np.allclose(a.center, bank.camera_mean("cam01"))


def test_samples_are_thinned_and_descriptor_is_normalised():
    bank = TrackFeatureBank(SamplingConfig(max_samples=4))
    bank.observe("cam01", 1, 0.0, [track(1, (10, 10, 50, 200))], (640, 480))
    for f in range(1, 6):
        bank.add("cam01/1", f, unit(1, f))
    rec = bank.tracks["cam01/1"]
    assert rec.sample_frames == [1, 3, 5] and len(rec.embeddings) == 3  # 5 > 4 -> every second kept
    assert np.isclose(np.linalg.norm(rec.descriptor), 1.0)


# ---------------------------------------------------------------------------------------- association
def test_same_camera_fragments_are_relinked():
    m = IdentityManager(AssociationConfig(appearance_threshold=0.9, max_gap_same_camera_s=5))
    a = m.assign(tracklet("cam01/1", 0, 10, unit(1, 0)))
    b = m.assign(tracklet("cam01/2", 12, 20, unit(1, 0.1)))      # 2 s later, similar: same person
    c = m.assign(tracklet("cam01/3", 15, 25, unit(1, 0.05)))     # overlaps tracklet 2 in time: cannot be them
    d = m.assign(tracklet("cam01/4", 40, 50, unit(1, 0)))        # 15 s after the last fragment: gap too long
    e = m.assign(tracklet("cam01/5", 52, 60, unit(0, 1)))        # right after 4, but looks different
    assert (a.decision, b.decision, c.decision, d.decision, e.decision) == ("new", "matched", "new", "new", "new")
    assert b.global_id == a.global_id and len({a.global_id, c.global_id, d.global_id, e.global_id}) == 4
    assert b.evidence["top"][0]["appearance"] > 0.99


def test_cross_camera_transit_windows():
    topo = CameraTopology({("cam01", "cam02"): TransitWindow(5, 20)}, default=None)
    m = IdentityManager(AssociationConfig(appearance_threshold=0.9), topo)
    a = m.assign(tracklet("cam01/1", 0, 10, unit(1, 0)))
    assert m.assign(tracklet("cam02/1", 20, 30, unit(1, 0))).global_id == a.global_id  # transit 10 s
    assert m.assign(tracklet("cam03/1", 35, 40, unit(1, 0))).decision == "new"  # cam03 not connected
    late = IdentityManager(AssociationConfig(appearance_threshold=0.9), topo)
    late.assign(tracklet("cam01/1", 0, 10, unit(1, 0)))
    assert late.assign(tracklet("cam02/1", 45, 50, unit(1, 0))).decision == "new"  # transit 35 s > 20 s


def test_overlapping_cameras_allow_simultaneous_presence():
    plain = CameraTopology({("a", "b"): TransitWindow(0, 30)}, default=None)
    overlap = CameraTopology({("a", "b"): TransitWindow(-5, 30, overlapping=True)}, default=None)
    for topo, expected in ((plain, "new"), (overlap, "matched")):
        m = IdentityManager(AssociationConfig(appearance_threshold=0.9), topo)
        m.assign(tracklet("a/1", 0, 10, unit(1, 0)))
        assert m.assign(tracklet("b/1", 5, 15, unit(1, 0))).decision == expected


def test_phrase_threshold_and_ablation_switches():
    m = IdentityManager(AssociationConfig(appearance_threshold=0.9))
    m.assign(tracklet("cam01/1", 0, 10, unit(1, 0), phrase="car"))
    assert m.assign(tracklet("cam01/2", 11, 20, unit(1, 0), phrase="truck")).decision == "new"
    assert m.assign(tracklet("cam01/3", 21, 30, unit(1, 0.6), phrase="car")).decision == "new"  # cos 0.86 < 0.9
    no_time = IdentityManager(AssociationConfig(appearance_threshold=0.9, use_time=False, use_phrase=False))
    no_time.assign(tracklet("cam01/1", 0, 10, unit(1, 0), phrase="car"))
    assert no_time.assign(tracklet("cam01/2", 5, 15, unit(1, 0), phrase="truck")).decision == "matched"


def test_attribute_cue_rejects_different_colours():
    red, blue = np.array([0.9, 0.05, 0.05]), np.array([0.05, 0.05, 0.9])
    for use, expected in ((False, "matched"), (True, "new")):
        m = IdentityManager(AssociationConfig(appearance_threshold=0.9, use_attributes=use, attribute_min_overlap=0.5))
        a = tracklet("cam01/1", 0, 10, unit(1, 0))
        a.attributes = red
        m.assign(a)
        b = tracklet("cam01/2", 12, 20, unit(1, 0))
        b.attributes = blue  # same appearance vector, different colour distribution
        assert m.assign(b).decision == expected


def test_low_evidence_merge_and_split():
    m = IdentityManager(AssociationConfig(appearance_threshold=0.9, min_samples=2))
    assert m.assign(tracklet("cam01/1", 0, 5, unit(1, 0), samples=1)).decision == "new_low_evidence"
    a = m.assign(tracklet("cam01/2", 6, 10, unit(1, 0)))
    b = m.assign(tracklet("cam01/3", 50, 60, unit(0, 1)))
    m.merge(a.global_id, b.global_id, reason="operator")
    assert m.global_id_of("cam01/3") == a.global_id and b.global_id not in m.identities
    new = m.split("cam01/3")
    assert m.global_id_of("cam01/3") == new != a.global_id
    assert [r.decision for r in m.records][-2:] == ["merged", "split"]
    with pytest.raises(ValueError, match="already assigned"):
        m.assign(tracklet("cam01/2", 6, 10, unit(1, 0)))


def test_peek_predicts_assign_without_changing_anything():
    m = IdentityManager(AssociationConfig(appearance_threshold=0.9, max_gap_same_camera_s=5))
    a = m.assign(tracklet("cam01/1", 0, 10, unit(1, 0)))
    open_track = tracklet("cam01/2", 12, 15, unit(1, 0.1))
    n_records, n_ids = len(m.records), len(m.identities)
    gid, score = m.peek(open_track)
    assert gid == a.global_id and score > 0.99
    assert (len(m.records), len(m.identities), m.global_id_of("cam01/2")) == (n_records, n_ids, None)
    assert m.peek(tracklet("cam01/3", 12, 15, unit(0, 1))) is None  # looks different
    assert m.assign(open_track).global_id == gid


def test_time_horizon_skips_identities_that_can_no_longer_match():
    topo = CameraTopology({("a", "b"): TransitWindow(0, 40)}, default=TransitWindow(0, 20))
    m = IdentityManager(AssociationConfig(appearance_threshold=0.9, max_gap_same_camera_s=5), topo)
    assert m.time_horizon_s == 40
    m.assign(tracklet("a/1", 0, 10, unit(1, 0)))
    rec = m.assign(tracklet("b/1", 60, 70, unit(1, 0)))  # 50 s after: beyond every window
    assert rec.decision == "new" and rec.evidence["candidates"] == 0
    assert IdentityManager(AssociationConfig(use_time=False)).time_horizon_s is None


def test_vectorised_decisions_equal_checking_every_identity():
    """Random tracklets: the manager's decisions equal a brute-force loop over all identities."""
    rng = np.random.default_rng(0)
    topo = CameraTopology({("c0", "c1"): TransitWindow(0, 15), ("c1", "c2"): TransitWindow(5, 30, overlapping=True)},
                          default=TransitWindow(0, 10))
    for reduction in ("max", "mean"):
        cfg = AssociationConfig(appearance_threshold=0.8, member_reduction=reduction, max_gap_same_camera_s=8)
        m = IdentityManager(cfg, topo)
        protos = [unit(*rng.normal(size=8)) for _ in range(6)]  # 6 "objects", noisy views of them
        t = 0.0
        for i in range(150):
            t += rng.uniform(0, 3)
            emb = unit(*(protos[rng.integers(6)] + rng.normal(scale=0.25, size=8)))
            tr = tracklet(f"c{rng.integers(3)}/{i}", t, t + rng.uniform(1, 6), emb)
            # brute force with the original rule: best feasible identity by appearance, then the threshold
            best, best_s = None, -1.0
            for ident in m.identities.values():
                if m._rejection(ident, tr) is not None:
                    continue
                sims = [float(tr.descriptor @ x.descriptor) for x in ident.members if x.descriptor is not None]
                s = max(sims) if reduction == "max" else float(np.mean(sims))
                if s > best_s + 1e-6 or (abs(s - best_s) <= 1e-6 and best is not None and ident.global_id < best):
                    best, best_s = ident.global_id, s
            rec = m.assign(tr)
            expected = best if best is not None and best_s >= cfg.appearance_threshold else None
            assert (rec.global_id if rec.decision == "matched" else None) == expected, (reduction, i)


def test_topology_from_config():
    topo = CameraTopology.from_dict({"default": None, "pairs": [{"from": "a", "to": "b", "min_s": 5, "max_s": 60}]})
    assert topo.window("a", "b").max_s == 60 and topo.window("b", "a").min_s == 5 and topo.window("a", "c") is None


# -------------------------------------------------------------------------------------------- metrics
def test_retrieval_metrics_hand_computed():
    g = np.eye(4, dtype=np.float32)
    g_ids, g_cams = [1, 1, 2, 3], ["a", "b", "b", "b"]
    q = np.array([[0.1, 0.9, 0.0, 0.4], [0.0, 0.0, 0.2, 0.9]], np.float32)  # query 1 (cam a), query 2 (cam a)
    m = retrieval_metrics(q, [1, 2], ["a", "a"], g, g_ids, g_cams, ranks=(1, 2))
    # query 1: gallery (1,a) is ignored (same id + camera); order 1b(0.9), 3(0.4), 2(0) -> rank 1, AP 1
    # query 2: order 3(0.9), 2(0.2), ... -> first hit at rank 2, AP 0.5
    assert m == {"queries": 2, "mAP": 0.75, "rank1": 0.5, "rank2": 1.0}


def test_merge_cameras_offsets_frames():
    rows, n = merge_cameras({"b": [(1, 7, 0, 0, 1, 1)], "a": [(2, 5, 0, 0, 1, 1)]}, {"a": 10, "b": 4})
    assert n == 14 and rows == [(2, 5, 0, 0, 1, 1), (11, 7, 0, 0, 1, 1)]


def test_auroc_perfect_and_chance():
    f = np.array([unit(1, 0), unit(1, 0.01), unit(0, 1), unit(0.01, 1)])
    assert same_vs_different_auroc(f, [1, 1, 2, 2])["auroc"] == 1.0
    assert same_vs_different_auroc(f, [1, 2, 1, 2])["auroc"] < 0.5


# ------------------------------------------------------------------------------------------- service
def test_service_closes_idle_tracks_and_links_them():
    calls = []

    class Emb:
        def embed_images(self, crops):
            return np.array([unit(*c) for c in calls.pop(0)], np.float32)

    svc = ReIDService(Emb(), SamplingConfig(every_n_frames=1, min_score=0.0, center=False),
                      AssociationConfig(appearance_threshold=0.9, max_gap_same_camera_s=10), close_after_s=1.0)
    img = np.zeros((480, 640, 3), np.uint8)

    def frame(i, t, tracks, dirs):
        calls.append(dirs)
        return svc.on_result(SimpleNamespace(camera_id="cam01", index=i, t_capture=t, image=img), None,
                             SimpleNamespace(tracks=tracks))

    assert frame(1, 0.0, [track(1, (100, 100, 150, 300))], [(1, 0)]) == []
    assert frame(2, 0.5, [track(1, (100, 100, 150, 300))], [(1, 0)]) == []
    recs = frame(3, 2.0, [track(2, (120, 100, 170, 300))], [(1, 0.02)])  # track 1 idle for 1.5 s -> closed
    assert [r.track_key for r in recs] == ["cam01/1"]
    final = svc.close_all()
    assert final[0].decision == "matched" and svc.global_id("cam01/2") == svc.global_id("cam01/1")
    assert svc.stats()["crops_embedded"] == 3 and svc.stats()["global_identities"] == 1


def test_service_gives_a_track_refound_after_it_ended_a_new_segment():
    """ByteTrack counts lost frames, the service counts seconds: when processing is slow, the tracker can re-find
    a track the service already ended. The key comes back; it must become a new tracklet, not crash."""
    from lgep.tracking.tracker import Track

    class Emb:
        def embed_images(self, crops):
            return np.array([unit(1, 0)] * len(crops), np.float32)

    svc = ReIDService(Emb(), SamplingConfig(every_n_frames=1, min_score=0.0, center=False),
                      AssociationConfig(appearance_threshold=0.9, max_gap_same_camera_s=10), close_after_s=1.0)
    img = np.zeros((480, 640, 3), np.uint8)
    t1 = Track("cam01", 1, "person", (100.0, 100.0, 150.0, 300.0), 0.9, 1)

    def frame(i, t, tracks):
        return svc.on_result(SimpleNamespace(camera_id="cam01", index=i, t_capture=t, image=img), None,
                             SimpleNamespace(tracks=tracks))

    frame(1, 0.0, [t1])
    assert [r.track_key for r in frame(2, 2.0, [])] == ["cam01/1"]  # idle 2 s > 1 s: ended and assigned
    frame(3, 2.5, [t1])  # the tracker re-finds track 1
    final = svc.close_all()
    assert [r.track_key for r in final] == ["cam01/1~2"] and final[0].decision == "matched"
    assert svc.segments.reopened == 1 and svc.stats()["tracks_refound_after_end"] == 1


def test_reid_worker_never_blocks_and_closes_on_stop():
    import threading

    from lgep.reid.service import ReIDWorker
    from lgep.tracking.tracker import Track

    gate = threading.Event()

    class SlowEmb:
        def embed_images(self, crops):
            gate.wait(5)
            return np.array([unit(1, 0)] * len(crops), np.float32)

    svc = ReIDService(SlowEmb(), SamplingConfig(every_n_frames=1, min_score=0.0, center=False),
                      AssociationConfig(appearance_threshold=0.9), close_after_s=1.0)
    got = []
    w = ReIDWorker(svc, on_records=got.extend, queue_size=2).start()
    img = np.zeros((480, 640, 3), np.uint8)
    t1 = Track("cam01", 1, "person", (100.0, 100.0, 150.0, 300.0), 0.9, 1)
    for i in range(6):  # the worker is stuck in the first embedding: the queue overflows
        w.submit(SimpleNamespace(camera_id="cam01", index=i + 1, t_capture=0.1 * i, image=img), None,
                 SimpleNamespace(tracks=[t1]))
    assert w.submitted == 6 and w.dropped >= 3  # oldest frames dropped, submit returned immediately
    gate.set()
    final = w.stop()
    assert [r.track_key for r in final] == ["cam01/1"] and w.stats()["processed"] >= 2


# --- Milestone 9: a dedicated Re-ID model for identity, CLIP for search -----------------------------------------


def test_identity_embeddings_drive_the_descriptor_and_are_thinned_with_clip():
    bank = TrackFeatureBank(SamplingConfig(max_samples=4, every_n_frames=1, min_score=0.0))
    bank.observe("cam01", 1, 0.0, [track(1, (100, 100, 150, 300))], (640, 480))
    for f in range(1, 12):  # sample f: CLIP embedding along x, identity embedding along y, both tagged with f
        bank.add("cam01/1", f, unit(1, 0, f / 100), identity=unit(0, 1, f / 100))
    rec = bank.tracks["cam01/1"]
    assert len(rec.embeddings) == len(rec.identity_embeddings) <= 4
    assert [round(e[2] / e[0] * 100) for e in rec.embeddings] == [round(i[2] / i[1] * 100) for i in rec.identity_embeddings]
    assert rec.descriptor[1] > 0.9 and rec.mean_embedding[0] > 0.9  # identity from the Re-ID model, search from CLIP
    assert bank.camera_mean("cam01", identity=True)[1] > 0.9 and bank.camera_mean("cam01")[0] > 0.9
    closed = bank.pop("cam01/1")
    assert closed.center is not None and closed.identity_center is not None
    plain = tracklet("cam02/1", 0, 1, unit(1, 0, 0))  # no identity model: descriptor from CLIP, as before
    assert plain.descriptor[0] > 0.99


def test_service_links_by_the_identity_model_when_clip_disagrees():
    """Two tracks that CLIP finds different but the identity model finds the same are linked, and the search
    index keeps CLIP's embeddings."""
    clip_calls, id_calls = [], []

    class Emb:
        def __init__(self, q):
            self.q = q

        def embed_images(self, crops):
            return np.array([unit(*c) for c in self.q.pop(0)], np.float32)

    svc = ReIDService(Emb(clip_calls), SamplingConfig(every_n_frames=1, min_score=0.0, center=False),
                      AssociationConfig(appearance_threshold=0.9, max_gap_same_camera_s=10), close_after_s=1.0,
                      identity_embedder=Emb(id_calls))
    img = np.zeros((480, 640, 3), np.uint8)

    def frame(i, t, tracks, clip, ident):
        clip_calls.append(clip)
        id_calls.append(ident)
        return svc.on_result(SimpleNamespace(camera_id="cam01", index=i, t_capture=t, image=img), None,
                             SimpleNamespace(tracks=tracks))

    frame(1, 0.0, [track(1, (100, 100, 150, 300))], [(1, 0, 0)], [(0, 1, 0)])
    frame(2, 2.0, [track(2, (120, 100, 170, 300))], [(0, 0, 1)], [(0, 1, 0.01)])  # CLIP: orthogonal; Re-ID: same
    svc.close_all()
    assert svc.global_id("cam01/2") == svc.global_id("cam01/1")
    assert not clip_calls and not id_calls  # both models embedded every sampled crop
