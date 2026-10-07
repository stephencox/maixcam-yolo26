# MaixCAM YOLO26 detector app — design

Date: 2026-10-07
Status: approved in conversation, awaiting written-spec review

## 1. Goal

A public, general-purpose COCO object-detection demo for **MaixCAM v1**
(SOPHGO SG2002), published on MaixHub as a new app. It succeeds MaixHub app
123 (`stephencox/miaxcamYolov11n320`, MaixPy + YOLOv11n 320), which stays
up and gets a pointer to this one.

"Done properly" means:

- written in C++ on **MaixCDK** (not MaixPy);
- models built by a reproducible, verified pipeline — not hand-made files;
- hardware-independent logic covered by host tests;
- versioned releases built by CI;
- no known defects carried over from the old app (backslash paths in
  `app.yaml`, wrong `name[zh]` size, inverted FPS maths with a
  divide-by-zero).

### Non-goals

- MaixCAM-Pro / MaixCAM2 support (may work; not tested or claimed).
- Custom-trained models, class filtering, recording video, streaming.
- Automated MaixHub upload (no API exists).

## 2. Key decisions

| Decision | Choice | Why |
|---|---|---|
| SDK | MaixCDK **v4.12.5**, pinned | Matches the board image (MaixPy v4.12.5, flashed 2026-10-07). A mismatched SDK fails at exec with `CVI_*: symbol not found`. |
| Decoder | Built-in `maix::nn::YOLO26` | Present in v4.12.5 (`components/nn/include/maix_nn_yolo26.hpp`, RVV-optimised). cat_repel's custom decoder only existed because its SDK pin predates YOLO26 support. Fallback if checkpoint 1 fails: port cat_repel's `yolo26_decode.hpp` (approach B). |
| Model sizes | YOLO26n at 320×320 **and** 640×640, switchable at runtime | Speed vs accuracy is the interesting comparison for a demo. |
| Model provenance | Built in-repo by `model/` pipeline on GPU box .30 | Reproducible and verified; see §5. |
| Binary storage | **Git LFS** for `*.cvimodel`, `*.png`, `*.jpg` | Same as the old repo; models are ~3–4 MB each. |
| Repo | New public repo `stephencox/maixcam-yolo26`, Apache-2.0 | Clean name; licence matches MaixCDK. |
| App id | `yolo26_detect` (new MaixHub listing) | Old id `yolov11n_object_320` describes a different model. |
| Releases | GitHub Actions on tag; manual MaixHub upload | §6. |

## 3. Repository layout

```
app.yaml                 id, name/name[zh], version, icon, desc/desc[zh], files
main/
  CMakeLists.txt
  Kconfig
  src/main.cpp           hardware wiring only (camera, display, touch, YOLO26, loop)
  include/
    fps_meter.hpp        rolling FPS + inference latency            (pure C++)
    controls.hpp         button layout, hit-test, threshold, toggle (pure C++)
    strings.hpp          en/zh string table + locale selection       (pure C++)
assets/
  app.png                                             (LFS)
  yolo26n_320.mud, yolo26n_640.mud
  yolo26n_320_int8.cvimodel, yolo26n_640_int8.cvimodel (LFS)
test/                    host tests for the pure headers (plain g++, no SDK)
model/                   export + quantisation pipeline (§5), MANIFEST.md
.github/workflows/       ci.yml, release.yml
.gitattributes           LFS patterns
justfile                 test, build, deploy, smoke, models, release
CHANGELOG.md
README.md                usage, build, manual test checklist, release checklist
```

Rule: `main.cpp` is the only translation unit that includes MaixCDK headers.
Anything that makes a decision lives in a pure header and is host-tested.

## 4. Runtime behaviour

### 4.1 Frame loop

1. `cam.read()` at display resolution (552×368), pixel format from
   `detector.input_format()`.
2. `detector.detect(img, conf_th, iou_th = 0.45)`; time **only this call** →
   inference latency.
3. Draw each box plus `"<label> <score>"`.
4. Draw overlay (buttons, status line), `disp.show()`.
5. Whole iteration time → FPS.

`FpsMeter`: ring of the last 30 frame durations and 30 latencies. Reports
"no data" (rendered "–") until at least one sample exists. Never divides by
zero; ignores non-positive durations.

### 4.2 Screen layout and controls

| Position | Control | Behaviour |
|---|---|---|
| top-left | `< Exit` | `app::set_exit_flag(true)` |
| top-right | `320` / `640` | switch model (§4.3) |
| bottom-left | `conf −` / value / `conf +` | step 0.05, clamp to [0.05, 0.95], default 0.50 |
| bottom-right | `Snapshot` | §4.4 |
| status line | `FPS 24.1 · 38 ms · 320` | from `FpsMeter` and current model |

Touch targets are at least 48 px tall. A press triggers on the press edge
only (no repeat while held). Layout rectangles and hit-testing are defined in
`controls.hpp` and computed from screen width/height, not hard-coded to
552×368.

Language: `maix::i18n::get_locale()`; `zh*` → Chinese, anything else →
English. All on-screen text comes from `strings.hpp`.

### 4.3 Model switching

- Only one model is loaded at a time (NPU memory is tight on SG2002).
- Toggle: show "Loading…", destroy the current `YOLO26`, construct the
  other.
- If loading 640 fails (missing file, OOM, exception), show an error
  message for ~2 s and reload 320.
- The selected size is persisted with `app::set_app_config_kv` and restored
  at startup (default 320).
- Confidence threshold is also persisted.

### 4.4 Snapshot

- Save the **annotated** frame as JPEG to
  `/root/snapshots/yolo26_<n>.jpg`, where `<n>` is one more than the
  highest existing index (the RTC may not be set, so no timestamps).
- Create the directory if it is missing.
- Show "Saved" or "Save failed" as a ~1.5 s toast.

### 4.5 Error handling

- No model loadable at startup: full-screen message naming the missing
  file; Exit still works; no crash loop.
- Any exception inside an iteration: log it, skip that frame, continue.
- The loop ends only on Exit or `app::need_exit()`.

## 5. Model pipeline (`model/`, runs on GPU box .30)

`just models` runs these steps end to end:

1. **Fetch.** Download Ultralytics `yolo26n.pt`; verify its sha256 against
   `model/sources.toml`.
2. **Export.** ONNX at 320×320 and 640×640, opset and Ultralytics version
   pinned. Export **without** the end-to-end NMS block (TopK /
   GatherElements / Mod do not compile for cv181x). The output layout must
   be whatever `maix::nn::YOLO26` expects — determined from
   `maix_nn_yolo26.cpp` and Sipeed's YOLO26 conversion docs during
   implementation, and recorded in MANIFEST.md.
3. **Quantise.** TPU-MLIR in a pinned Docker image:
   `model_transform` with mean `0,0,0` and scale `1/255` fused into the model
   so the `.mud` preprocessing values apply, then `model_deploy` to INT8 for
   `cv181x`. Calibration: 500 COCO val2017 images chosen by a fixed seed;
   the list of file names is committed.
4. **Verify.**
   - Simulator: INT8 vs FP32 on a fixed set of 50 COCO val images; mAP50
     drop ≤ 3 points, and TPU-MLIR's per-layer cosine check passes.
   - Board: the same 10 committed test images produce detections on the
     board that match the simulator output (same classes; box IoU ≥ 0.9).
5. **Record.** Write `model/MANIFEST.md`: source hashes, tool versions,
   calibration list hash, measured mAP50 (FP32 and INT8), and board FPS and
   latency per size.

Model changes bump the app's minor version and update MANIFEST.md in the
same commit.

### Checkpoint 1 (gate before any UI work)

Export and quantise 320 only, install it on the board with a minimal
`main.cpp`, and confirm sensible COCO detections on the 10 test images
via the built-in decoder. If it fails and the export cannot be fixed, switch
to approach B (port cat_repel's decoder, extended to 80 classes) and update
this spec.

## 6. Build, test, CI, release

### Host tests (`just test`)

Plain g++, `-Wall -Wextra -Werror`, ASan + UBSan.

- `fps_meter`: empty window, single sample, ring wrap-around, latency
  independent of FPS, non-positive durations ignored (old-app regression).
- `controls`: hit-testing at every button edge for 552×368 and one other
  resolution; threshold step and clamp; press-edge triggering; toggle and
  fallback state.
- `strings`: every key has en and zh text; unknown locale → English.

### On the board

- `just deploy`: build, `scp` to the board over USB (`<board-usb-ip>`),
  `app_store install`.
- `just smoke`: run the §5 board verification against the installed models.
- README manual checklist: each button, model switch both ways, snapshot
  written, both languages render, Exit returns to the launcher.

### CI (`.github/workflows/ci.yml`, every push and PR)

- Host tests.
- `maixcdk build -p maixcam` in the MaixCDK v4.12.5 toolchain container
  (proves the app compiles for the board).
- Checkout with `lfs: true`.

### Release (`.github/workflows/release.yml`, on tag `vX.Y.Z`)

- Fail unless the tag, `app.yaml` `version`, and the top `CHANGELOG.md`
  entry all match.
- `maixcdk release -P maixcam`; attach `yolo26_detect_vX.Y.Z.zip` and its
  `.sha256` to the GitHub Release.
- The model pipeline does **not** run in CI; CI packages the committed LFS
  models.

### Publishing

Manual upload of the release zip to maixhub.com/app, following the README
release checklist, which ends with "board smoke test passed". Add a line to
app 123's description pointing to the new app.

### Versioning

SemVer. 1.0.0 is the first MaixHub release. Model changes → minor bump.

## 7. Risks

| Risk | Mitigation |
|---|---|
| Our ONNX layout does not match `nn::YOLO26` | Checkpoint 1 before UI work; approach B fallback. |
| 640 model too large or slow for the NPU | Load failure falls back to 320; MANIFEST records real FPS; if 640 is unusable, ship 320 only and say so. |
| MaixCDK toolchain image is unavailable to CI | Build the image from the pinned MaixCDK tag in a workflow and cache it in GHCR. |
| GitHub LFS bandwidth quota | Models are small; CI uses a cached LFS checkout. |
