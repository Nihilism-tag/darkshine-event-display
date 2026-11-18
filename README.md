# DarkSHINE Recoil Tracker — Interactive 3D Event Cloud & Track Display

Zero-backend WebGL2 interactive 3D event display for the DarkSHINE Recoil Tracker (`N1–N6` silicon strip stations), comparing **GNN (`V5` / `V6`)** and **DS native combinatorial track finding**.

## Datasets & Modes (100% Full Sample, Lazy-Loaded Binary Shards)

1. **`CLOUD` — 4-Cohort 3D Cross-Event Clouds (`N = 5,491`, `1.1 MB` initial load)**:
   - Mutually exclusive & exhaustive recall cohorts on the DS `smear=1` multi-track sample: **Both recalled**, **GNN-only recalled**, **DS-only recalled**, and **Neither recalled**.
   - **Dual Synchronized 3D Viewports** (`GNN vs DS Recall` or `4-Cohort Split`) sharing a live camera + **Single Overlay** mode.
   - Real-time slicing by true momentum (`p_true`), event hit occupancy, particle species (`e+` / `e−`), and coordinate source (`Truth` vs `Reconstructed`).
2. **`EVENT` — Full DS `smear=1` Second-Track Single Events (`N = 5,491`, `#0–#5,490`)**:
   - All 9 evaluation cohorts, direct event-index navigation, Showcase trio (**Case A `#1613`**, **Case B `#2589`**, **Case C `#86`**), 3D `V5 / V6 / DS` candidate trajectories, MC reference tracks, and live GNN edge-probability thresholding.
3. **`FULL` — Full GNN-Digitized Sample (`N = 149,859`, `#0–#149,858`)**:
   - Streamed on demand via 4 compact binary shards (`< 23 MB` each), with 6-station hit coloring, edge classification overlays (`TP / FP / FN / Candidate`), and MC truth tracks.

## Local Preview

```bash
python3 -m http.server 8080
```

Then open `http://127.0.0.1:8080/` in any WebGL2-capable browser.
