# YOLO26 Detector

Real-time COCO object detection on the MaixCAM using YOLO26n, with an INT8
model at 320 or 640 input, on-screen touch controls, and snapshots. The UI is
available in English and Chinese. It follows the stock detector app
(app 123), which is the predecessor; the MaixHub listing link will be added
once the app is published.

## Controls

| Control | Action |
|---|---|
| `Exit` | Quit to the launcher |
| `-` / `+` | Confidence threshold down / up. Range 0.45 to 0.95, step 0.05, default 0.50 |
| `320` / `640` | Switch model size (the button shows the current size). Shows "Loading..." during the switch |
| `Snapshot` | Save the current frame to `/root/snapshots/yolo26_N.jpg` |

The threshold and model size are remembered between launches. The status line
shows FPS, latency and the model size (`-` until the first numbers arrive).

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

`just models 320` or `just models 640` exports, converts (INT8 with a BF16
head) and runs the simulator mAP gate on the GPU box. Then run the board
compare (`model/compare_board.py`, see `model/MANIFEST.md`) and update
`model/MANIFEST.md` with the results.

## Manual test checklist

Run after `just deploy`, from the launcher. Tick each item.

1. [ ] Launch from the launcher; the status line shows `-` then numbers within 2 s.
2. [ ] Point at a person: a red box labelled `person`.
3. [ ] `+` up to 0.95: fewer boxes. `-` down to 0.45, and it stops there.
4. [ ] Exit and relaunch: threshold and model size are remembered.
5. [ ] Tap `320`: "Loading..." appears, then the button reads `640`; FPS drops and the status shows `640`. Hold the finger on the button through the load: it switches once only.
6. [ ] Rename `/maixapp/apps/yolo26_detect/assets/yolo26n_640_int8.cvimodel`, then tap `320`: a yellow fallback message appears and it stays on 320. Restore the file.
7. [ ] Tap Snapshot twice: "Saved" appears twice; `ssh maixcam ls /root/snapshots` shows `yolo26_1.jpg` and `yolo26_2.jpg`; open one: boxes present, no buttons.
8. [ ] Settings, Language, 中文, then relaunch: Chinese labels render (no boxes or tofu).
9. [ ] Exit returns to the launcher.

## Release checklist

To be filled in by the release task (Task 11).
