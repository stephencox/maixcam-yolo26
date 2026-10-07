![YOLO26 Detector: INT8 detections on a COCO val2017 photo](docs/art/cover.png)

# YOLO26 Detector

Real-time COCO object detection on the MaixCAM using YOLO26n, with an INT8
model at 320 or 640 input, on-screen touch controls, and snapshots. The UI is
available in English and Chinese. It is the successor to my MaixHub app 123 (YOLOv11n 320, MaixPy); the MaixHub
listing link will be added once the app is published.

## Controls

| Control | Action |
|---|---|
| `Exit` | Quit to the launcher |
| `-` / `+` | Confidence threshold down / up. Range 0.45 to 0.95, step 0.05, default 0.50 |
| `320` / `640` | Switch model size (the button shows the current size). Shows "Loading..." during the switch |
| `Snapshot` | Save the current frame to `/root/snapshots/yolo26_N.jpg` |

The threshold and model size are remembered between launches. The status
pill at the top shows FPS and latency, e.g. `8.9 fps | 111 ms` (`-` until the
first numbers arrive). Boxes are coloured per class with a label tab such as
`person 91%`.

Notes:

- 640 is slow by design (about 641 ms per inference, roughly 1.6 FPS). It is an
  accuracy-versus-speed demo; 320 runs at about 111 ms. The 640 model also
  leaves only about 7 MB of free RAM on the board after loading (see
  `model/MANIFEST.md`).
- If 640 fails to load, the app falls back to 320 and shows a yellow message.
- Snapshots contain the frame with detection boxes and labels, but without the
  buttons and status overlay. The UI chrome is not part of the picture.

## Building

Needs Docker, Git LFS, and an ssh alias `maixcam` for the board.

- `just test` runs the host tests for the pure headers in `main/include/`.
- `just build` cross-compiles the app for the MaixCAM (MaixCDK in Docker).
- `just deploy` packages, copies to the board and installs via the app store CLI.

## Rebuilding models

Prerequisites: a Linux GPU box reachable over ssh (default `GPU=192.168.12.30`;
override with `GPU=host just models 640`) with Docker, COCO val2017 and its
annotations at `~/coco`, and `model/.venv` created with `uv` on the box.

`just models 320` or `just models 640` exports, converts (INT8 with a BF16
head) and runs the simulator mAP gate on the GPU box. Then run
`just smoke 320` (or `640`) for the board-vs-simulator compare and update
`model/MANIFEST.md` with the results.

`just smoke` copies the model and the 10 images in `model/board_images/` to the
board, runs `model/board_detect.py` there and compares with the simulator output
(`model/compare_board.py`). Run it only with the app closed (the launcher is
fine: it uses the NPU, not the camera), never while `yolo26_detect` is running.
Both `just deploy` and `just smoke` use the `BOARD` ssh host (default
`maixcam`; override with `BOARD=host just deploy`).

`model/board_images/` are 10 COCO val2017 images under their original Flickr
licences (COCO terms of use), not Apache-2.0.

## Credits

Cover photo: "Montreal Motorcycle Cop" by Gerry Lauzon
([Flickr](https://www.flickr.com/photos/bikeman04/7933423348/)), licensed
[CC BY 2.0](https://creativecommons.org/licenses/by/2.0/), via COCO 2017 val
(image 538364). The detections drawn on it are real INT8 640 model output
(`docs/art/cover_dets.json`). The cover image `docs/art/cover.png` is a derivative
(cropped into a layout, boxes and text added) and is shared under the same
licence, CC BY 2.0. `just art` re-renders the cover and the launcher icon.

## Manual test checklist

Run after `just deploy`, from the launcher. Tick each item.

1. [ ] Launch from the launcher; the status pill shows `-` then numbers within 2 s.
2. [ ] Point at a person: a coral box with a `person NN%` tab.
3. [ ] `+` up to 0.95: fewer boxes. `-` down to 0.45, and it stops there.
4. [ ] Exit and relaunch: threshold and model size are remembered.
5. [ ] Tap `320`: "Loading..." appears, then the button reads `640`; FPS drops and the status pill shows the lower FPS. Hold the finger on the button through the load: it switches once only.
6. [ ] Rename `/maixapp/apps/yolo26_detect/assets/yolo26n_640_int8.cvimodel`, then tap `320`: a yellow fallback message appears and it stays on 320. Restore the file.
7. [ ] Tap Snapshot twice: "Saved" appears twice; `ssh maixcam ls /root/snapshots` shows `yolo26_1.jpg` and `yolo26_2.jpg`; open one: boxes present, no buttons.
8. [ ] Settings, Language, 中文, then relaunch: Chinese labels render (no boxes or tofu).
9. [ ] Exit returns to the launcher.

## Release checklist

1. `just test`; `just deploy`; full manual checklist passes; board vs sim compare passes for both sizes.
2. Bump `app.yaml` `version`, add the `CHANGELOG.md` entry with today's date, commit.
3. `git tag vX.Y.Z && git push origin main vX.Y.Z`; wait for the `release` workflow to go green.
4. Download `yolo26_detect_vX.Y.Z.zip` from the GitHub Release; verify its sha256.
5. Upload it at maixhub.com/app (new version of `yolo26_detect`), paste the CHANGELOG entry as notes.
6. Install from MaixHub on the board via the app store and run checklist items 1, 2, 9.
