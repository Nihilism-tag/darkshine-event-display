#!/usr/bin/env python3
"""Export FULL static WebGL data bundle for GitHub Pages (all 3 modes, zero subsampling).

Exports:
  1) Mode 'cloud' (N = 5,491 4-cohort 3D clouds):
       - data/cloud_5491.bin + data/cloud_meta.json (~1.1 MB; loaded on initial paint)
  2) Mode 'second_track' (ALL 5,491 DS smear=1 multi-track single events + 9 cohorts):
       - data/st_5491_geom.bin + data/st_5491_meta.json (~11.5 MB; lazy-loaded)
       - data/st_5491_routes.json (~2.8 MB; all 5,491 V5/V6/DS candidates + metrics)
  3) Mode 'full_gnn' (ALL 149,859 full GNN-digitized events + MC truth):
       - data/full_gnn_s0.bin .. full_gnn_s3.bin (~20 MB each; lazy-loaded per shard)
       - data/full_gnn_meta.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np


def _write_aligned_bin(bin_path: Path, sections: list[tuple[str, np.ndarray, str]]) -> tuple[int, dict]:
    layout: dict[str, dict] = {}
    cursor = 0
    with bin_path.open("wb") as fout:
        for name, arr, dtype_str in sections:
            rem = cursor % 4
            if rem != 0:
                pad = 4 - rem
                fout.write(b"\x00" * pad)
                cursor += pad
            raw = np.ascontiguousarray(arr).tobytes()
            fout.write(raw)
            layout[name] = {
                "byteOffset": cursor,
                "length": int(arr.size),
                "dtype": dtype_str,
            }
            cursor += len(raw)
    return cursor, layout


def _pack_tracks(track_list: list[np.ndarray | None]) -> tuple[np.ndarray, np.ndarray]:
    offs = np.zeros(len(track_list) + 1, dtype=np.uint32)
    blocks: list[np.ndarray] = []
    cur = 0
    for i, arr in enumerate(track_list):
        if arr is not None and len(arr) > 0:
            a32 = np.asarray(arr, dtype=np.float32)
            blocks.append(a32)
            cur += len(a32)
        offs[i + 1] = cur
    xyz = np.vstack(blocks).astype(np.float32) if blocks else np.zeros((0, 3), dtype=np.float32)
    return offs, xyz


def _quantize_prob_u16(prob: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(np.asarray(prob, dtype=np.float32) * 65535.0), 0, 65535).astype(np.uint16)


def _export_event_shard(
    bin_path: Path,
    ev_start: int,
    ev_end: int,
    pred_npz_dict,
    mc_npz_dict,
) -> dict:
    """Pack a contiguous slice [ev_start, ev_end) of events (geometry + edges + MC CSR)."""
    pos_off_all = pred_npz_dict["pos_offsets"]
    edge_off_all = pred_npz_dict["edge_offsets"]

    p0, p1 = int(pos_off_all[ev_start]), int(pos_off_all[ev_end])
    e0, e1 = int(edge_off_all[ev_start]), int(edge_off_all[ev_end])

    pos_off = (pos_off_all[ev_start:ev_end + 1] - p0).astype(np.uint32)
    positions = np.asarray(pred_npz_dict["positions"][p0:p1], dtype=np.float32).reshape(-1)
    layer_id = np.asarray(pred_npz_dict["layer_id"][p0:p1], dtype=np.uint8)

    edge_off = (edge_off_all[ev_start:ev_end + 1] - e0).astype(np.uint32)
    ei_slice = pred_npz_dict["edge_index"][:, e0:e1]
    edge_u = np.asarray(ei_slice[0], dtype=np.uint16)
    edge_v = np.asarray(ei_slice[1], dtype=np.uint16)
    edge_truth = np.asarray(pred_npz_dict["edge_truth"][e0:e1], dtype=np.uint8)
    prob_v5 = _quantize_prob_u16(pred_npz_dict["prob_v5"][e0:e1])
    prob_v6 = _quantize_prob_u16(pred_npz_dict["prob_v6"][e0:e1])

    # Pack sorted >=2-hit MC tracks per event cleanly
    n_slice = ev_end - ev_start
    mc_ev_off = np.zeros(n_slice + 1, dtype=np.uint32)
    mc_trk_off_list = [0]
    mc_hits_blocks: list[np.ndarray] = []
    mc_ids_list: list[int] = []

    id_off = mc_npz_dict["mc_id_offsets"]
    hit_off = mc_npz_dict["mc_hit_offsets"]
    t_off_off = mc_npz_dict["mc_track_off_off"]
    mc_id_arr = mc_npz_dict["mc_id"]
    mc_hit_arr = mc_npz_dict["mc_hit"]
    mc_toff_arr = mc_npz_dict["mc_track_off"]
    map_ok = mc_npz_dict["mapping_ok"]

    pos_2d = pred_npz_dict["positions"]
    lay_1d = pred_npz_dict["layer_id"]

    trk_count = 0
    hit_cursor = 0
    for local_i, g_i in enumerate(range(ev_start, ev_end)):
        if bool(map_ok[g_i]):
            ia, ib = int(id_off[g_i]), int(id_off[g_i + 1])
            ha = int(hit_off[g_i])
            ta, tb = int(t_off_off[g_i]), int(t_off_off[g_i + 1])
            ids_ev = mc_id_arr[ia:ib]
            toffs_ev = mc_toff_arr[ta:tb]
            ev_p0 = int(pos_off_all[g_i])
            ev_nh = int(pos_off_all[g_i + 1] - ev_p0)
            if len(toffs_ev) == len(ids_ev) + 1:
                for ti in range(len(ids_ev)):
                    raw_h = mc_hit_arr[ha + int(toffs_ev[ti]):ha + int(toffs_ev[ti + 1])]
                    valid_h = [int(h) for h in raw_h if 0 <= int(h) < ev_nh]
                    if len(valid_h) >= 2:
                        valid_h.sort(key=lambda h: (int(lay_1d[ev_p0 + h]), float(pos_2d[ev_p0 + h, 2]), h))
                        arr_h = np.asarray(valid_h, dtype=np.uint16)
                        mc_hits_blocks.append(arr_h)
                        hit_cursor += len(arr_h)
                        mc_trk_off_list.append(hit_cursor)
                        mc_ids_list.append(int(ids_ev[ti]))
                        trk_count += 1
        mc_ev_off[local_i + 1] = trk_count

    mc_trk_off = np.asarray(mc_trk_off_list, dtype=np.uint32)
    mc_hits = np.concatenate(mc_hits_blocks).astype(np.uint16) if mc_hits_blocks else np.zeros(0, dtype=np.uint16)
    mc_ids = np.asarray(mc_ids_list, dtype=np.int32)

    sections = [
        ("pos_off", pos_off, "uint32"),
        ("positions", positions, "float32"),
        ("layer_id", layer_id, "uint8"),
        ("edge_off", edge_off, "uint32"),
        ("edge_u", edge_u, "uint16"),
        ("edge_v", edge_v, "uint16"),
        ("edge_truth", edge_truth, "uint8"),
        ("prob_v5", prob_v5, "uint16"),
        ("prob_v6", prob_v6, "uint16"),
        ("mc_ev_off", mc_ev_off, "uint32"),
        ("mc_trk_off", mc_trk_off, "uint32"),
        ("mc_hits", mc_hits, "uint16"),
        ("mc_ids", mc_ids, "int32"),
    ]
    nbytes, layout = _write_aligned_bin(bin_path, sections)
    return {
        "file": bin_path.name,
        "ev_start": ev_start,
        "ev_end": ev_end,
        "n_events": n_slice,
        "total_bytes": nbytes,
        "sections": layout,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-dir", required=True,
                    help="Directory containing viewer modules and datasets")
    ap.add_argument("--out-dir", required=True,
                    help="Target data/ directory inside the static web repository")
    args = ap.parse_args()

    src_dir = Path(args.source_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    # Remove old curated-only file if present
    old_sc = out_dir / "showcase_events.json"
    if old_sc.exists():
        old_sc.unlink()

    sys.path.insert(0, str(src_dir))
    from mc_particles import MCParticleIndex
    import second_track_cloud as stc
    import second_track_mode as stm

    bundle = stm.SecondTrackBundle(src_dir / "second_track_data" / "events_5491.jsonl")
    if not bundle.ok:
        raise RuntimeError(f"Failed to load second-track bundle: {bundle.error}")

    st_pred_path = str(src_dir / "second_track_data" / "second_track_smear1_predictions.npz")
    st_mc_path = str(src_dir / "second_track_data" / "second_track_smear1_mc.npz")
    st_mp_path = str(src_dir / "second_track_data" / "second_track_smear1_mc_particles.npz")

    mp_idx = MCParticleIndex(st_mp_path, n_events=len(bundle.rows))
    eng = stc.SecondTrackCloudEngine(bundle, st_pred_path, mp_idx)

    # ── 1) Export 4-Cohort 3D Cloud Bundle (5,491 events, ~1.1 MB) ──
    for mname, expected in (("v6", (1909, 1329, 232, 2021)),
                            ("v5", (2000, 1549, 141, 1801))):
        masks = eng.cohort_masks(mname)
        got = tuple(int(masks[k].sum()) for k in stc.COHORT_KEYS)
        if got != expected:
            raise AssertionError(f"Cohort mismatch for {mname}: got {got}, expected {expected}")

    n_ev = eng.n_events
    flags = (
        (eng.u_v5.astype(np.uint8) << 0)
        | (eng.u_v6.astype(np.uint8) << 1)
        | (eng.u_ds.astype(np.uint8) << 2)
        | (eng.has_shared.astype(np.uint8) << 3)
        | ((eng.pdg == -11).astype(np.uint8) << 4)
        | ((eng.pdg == 11).astype(np.uint8) << 5)
    )
    t_off, t_xyz = _pack_tracks(eng.truth_tracks)
    v5_off, v5_xyz = _pack_tracks(eng.reco_tracks["V5"])
    v6_off, v6_xyz = _pack_tracks(eng.reco_tracks["V6"])
    ds_off, ds_xyz = _pack_tracks(eng.reco_tracks["DS"])

    cloud_sections = [
        ("p_true", eng.p_true.astype(np.float32), "float32"),
        ("dx_bend", eng.dx_bend.astype(np.float32), "float32"),
        ("n_layers", eng.n_layers.astype(np.uint8), "uint8"),
        ("n_ev_hits", eng.n_ev_hits.astype(np.uint16), "uint16"),
        ("flags", flags.astype(np.uint8), "uint8"),
        ("truth_off", t_off, "uint32"),
        ("truth_xyz", t_xyz.reshape(-1), "float32"),
        ("v5_off", v5_off, "uint32"),
        ("v5_xyz", v5_xyz.reshape(-1), "float32"),
        ("v6_off", v6_off, "uint32"),
        ("v6_xyz", v6_xyz.reshape(-1), "float32"),
        ("ds_off", ds_off, "uint32"),
        ("ds_xyz", ds_xyz.reshape(-1), "float32"),
    ]
    cloud_bytes, cloud_layout = _write_aligned_bin(out_dir / "cloud_5491.bin", cloud_sections)
    (out_dir / "cloud_meta.json").write_text(json.dumps({
        "schema_version": 2,
        "n_events": int(n_ev),
        "station_z_mm": [7.83, 38.83, 89.83, 120.83, 151.83, 180.33],
        "verified_cohorts": {
            "v6": {"both": 1909, "gnn_only": 1329, "ds_only": 232, "neither": 2021},
            "v5": {"both": 2000, "gnn_only": 1549, "ds_only": 141, "neither": 1801},
        },
        "binary_file": "cloud_5491.bin",
        "total_bytes": cloud_bytes,
        "sections": cloud_layout,
    }, indent=2) + "\n")
    print(f"[1/3] Cloud bundle: {cloud_bytes / 1024:.1f} KB")

    # ── 2) Export ALL 5,491 Second-Track Single Events (Geometry + Routes + 9 Cohorts) ──
    st_pred_npz = np.load(st_pred_path, allow_pickle=False)
    st_mc_npz = np.load(st_mc_path, allow_pickle=False)
    st_shard_meta = _export_event_shard(out_dir / "st_5491_geom.bin", 0, n_ev, st_pred_npz, st_mc_npz)
    (out_dir / "st_5491_meta.json").write_text(json.dumps(st_shard_meta, indent=2) + "\n")

    st_pos_all = st_pred_npz["positions"]
    st_poff_all = st_pred_npz["pos_offsets"]
    st_lay_all = st_pred_npz["layer_id"]

    compact_rows = []
    for i, row in enumerate(bundle.rows):
        p0, p1 = int(st_poff_all[i]), int(st_poff_all[i + 1])
        nh = p1 - p0

        def _sort_h(hlist):
            v = [int(h) for h in hlist if 0 <= int(h) < nh]
            v.sort(key=lambda h: (int(st_lay_all[p0 + h]), float(st_pos_all[p0 + h, 2]), h))
            return v

        tid = int(row["target_id"])
        tmeta = mp_idx.lookup(i, tid)
        sp_name = tmeta.name if (tmeta is not None and tmeta.name) else "e±"
        t_hits = _sort_h(row["target_hits"])

        r_pack = []
        for rname in ("V5", "V6", "DS"):
            rr = row[rname]
            m_idx = rr.get("matched_index")
            m_hits = []
            o_cands = []
            for ci, cand in enumerate(rr.get("candidates", [])):
                sh = _sort_h(cand["hits"])
                if m_idx is not None and int(ci) == int(m_idx):
                    m_hits = sh
                elif len(sh) >= 2:
                    o_cands.append(sh)
            r_pack.append([
                1 if stm.route_usable(rr) else 0,
                m_hits,
                o_cands,
                stm.format_route_line(rname, rr),
                stm.format_route_debug(rname, rr),
            ])

        compact_rows.append([
            tid,
            round(float(row["p_true"]), 1),
            sp_name,
            len(row["target_layers"]),
            int(row.get("ambiguous_target_hits", 0)),
            t_hits,
            r_pack[0],
            r_pack[1],
            r_pack[2],
        ])

    showcase_defs = [
        {"key": c.key, "label": c.label, "viewer_idx": c.viewer_idx, "summary": c.summary}
        for c in stm.SHOWCASE_CASES
    ]
    routes_payload = {
        "n_events": n_ev,
        "showcase_cases": showcase_defs,
        "cohort_labels": bundle.cohort_labels,
        "cohorts": bundle.cohorts,
        "rows": compact_rows,
    }
    routes_path = out_dir / "st_5491_routes.json"
    routes_path.write_text(json.dumps(routes_payload, separators=(",", ":")) + "\n")
    print(f"[2/3] Full 5,491 2nd-track events: geom={st_shard_meta['total_bytes']/1024/1024:.2f} MB, routes={routes_path.stat().st_size/1024/1024:.2f} MB")

    # ── 3) Export ALL 149,859 Full GNN-Digitized Events in 4 Shards (<21 MB each) ──
    full_pred_npz = np.load(str(src_dir / "predictions_full.npz"), allow_pickle=False)
    full_mc_npz = np.load(str(src_dir / "predictions_full_mc.npz"), allow_pickle=False)
    n_full = int(full_pred_npz["n_events"])
    n_shards = 4
    shard_size = (n_full + n_shards - 1) // n_shards
    shard_metas = []
    for s_idx in range(n_shards):
        s_start = s_idx * shard_size
        s_end = min(n_full, (s_idx + 1) * shard_size)
        s_meta = _export_event_shard(
            out_dir / f"full_gnn_s{s_idx}.bin", s_start, s_end, full_pred_npz, full_mc_npz
        )
        shard_metas.append(s_meta)
        print(f"      Full shard {s_idx} [{s_start}..{s_end}): {s_meta['total_bytes']/1024/1024:.2f} MB")

    (out_dir / "full_gnn_meta.json").write_text(json.dumps({
        "n_events": n_full,
        "shard_size": shard_size,
        "shards": shard_metas,
    }, indent=2) + "\n")
    print(f"[3/3] Full 149,859 GNN events exported across {n_shards} shards.")


if __name__ == "__main__":
    main()
