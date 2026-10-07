# MaixCAM YOLO26 Detector App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A MaixCDK C++ app (`yolo26_detect`) that runs COCO YOLO26n at 320 or 640 on MaixCAM v1, with models built by a reproducible, verified pipeline, host-tested logic, and CI-built releases.

**Architecture:** `main/src/main.cpp` is the only file touching MaixCDK; every decision (FPS maths, layout/hit-testing, threshold, model choice, strings, snapshot naming) lives in pure headers under `main/include/` tested with plain g++. Models come from `model/` (Ultralytics → cut ONNX → TPU-MLIR INT8) run on GPU box <gpu-box> and are committed via Git LFS. Inference/decoding uses MaixCDK's built-in `maix::nn::YOLO26`.

**Tech Stack:** C++17, MaixCDK v4.12.5 (commit `db6b17821b900c97eaa899be548ddc1cba0dd635`), Docker `sipeed/maixcdk-builder`, Python 3.12 + ultralytics + onnx + onnxruntime + pycocotools, TPU-MLIR `tpu_mlir[onnx]==1.28.1` on `sophgo/tpuc_dev`, just, GitHub Actions, Git LFS.

**Spec:** `docs/superpowers/specs/2026-10-07-maixcam-yolo26-design.md`

## Global Constraints

- Target: MaixCAM v1 (SG2002), board image `maixcam-2026-01-24-maixpy-v4.12.5`.
- MaixCDK pinned to commit `db6b17821b900c97eaa899be548ddc1cba0dd635` (v4.12.5). Never build against another commit. Mount it at a nested path (`/opt/MaixCDK`), never `/<dir>`.
- App id `yolo26_detect`; repo `stephencox/maixcam-yolo26`; licence Apache-2.0.
- Only `main/src/main.cpp` may include MaixCDK headers.
- Git LFS for `*.cvimodel`, `*.png`, `*.jpg`.
- Model input sizes: 320×320 and 640×640. Only one model loaded at a time.
- Confidence threshold: step 0.05, range [0.45, 0.95], default 0.50.
- IoU passed to `detect()`: 0.45 (unused by the decoder; kept for API parity).
- TPU-MLIR: `model_deploy` WITHOUT `--fuse_preprocess`; `.mud` has `mean = 0, 0, 0`, `scale = 0.00392156862745098` ×3, `input_type = rgb`, `model_type = yolo26`.
- Calibration 500 / eval 50 / board 10 COCO val2017 images, disjoint, fixed seed `20261007`.
- Acceptance: INT8 mAP50 within 3 points of FP32 on the 50 eval images; board vs simulator on the 10 board images: same class and IoU ≥ 0.9 for every matched detection, and equal detection counts.
- Heavy compute (export, calibration, simulation) runs on <gpu-box>, never on the home media server.
- Status line format: `FPS 24.1 | 38 ms | 320`, `-` for missing values, ASCII only.
- Snapshots: `/root/snapshots/yolo26_<n>.jpg`, n = max existing + 1, starting at 1.
- Persisted config: `app::set_app_config_kv("yolo26_detect", "model", "320"|"640")` and `("yolo26_detect", "conf", "<pct>")`.
- Commit trailer on every commit: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Touch held across a model switch** — the finger is still down when the new model finishes loading; it must not toggle again. Pinned by `PressEdge` tests in Task 2 (held press yields exactly one edge).
2. **Corrupt or out-of-range persisted config** (`conf=abc`, `conf=7`, `model=1024`) — must fall back to defaults, not crash or clamp to nonsense. Pinned by `Threshold::from_pct_string` / `parse_model_size` tests in Task 2.
3. **Snapshot directory containing odd names** (`yolo26_.jpg`, `yolo26_12a.jpg`, `yolo26_99999999999999.jpg`, other files) — must ignore them and still produce a fresh index. Pinned in Task 4.
4. **Clock jumps / zero-length frames** (`ticks_us` difference 0, or the first frame) — FPS must show `-`, never `inf`/`nan`. Pinned in Task 1.
5. **Small screens** — layout must stay non-overlapping and in-bounds at 320×240 as well as 552×368. Pinned in Task 2.

---

## File Structure

```
.gitattributes .gitignore LICENSE README.md CHANGELOG.md justfile app.yaml
main/CMakeLists.txt  main/Kconfig
main/src/main.cpp                  Task 8, 9  (only MaixCDK user)
main/include/fps_meter.hpp         Task 1
main/include/controls.hpp          Task 2
main/include/strings.hpp           Task 3
main/include/snapshot.hpp          Task 4
test/check.hpp                     Task 1  (tiny assert harness)
test/test_fps_meter.cpp            Task 1
test/test_controls.cpp             Task 2
test/test_strings.cpp              Task 3
test/test_snapshot.cpp             Task 4
assets/app.png                     Task 8 (LFS)
assets/yolo26n_{320,640}.mud       Task 7
assets/yolo26n_{320,640}_int8.cvimodel  Task 6, 7 (LFS)
model/pyproject.toml model/sources.toml model/lists/*.txt
model/select_images.py             Task 5
model/export.py                    Task 5
model/decode.py                    Task 5  (Python mirror of nn::YOLO26 post-process)
model/eval_map.py                  Task 5
model/convert.sh                   Task 6
model/sim_detect.py                Task 6  (runs inside tpu_mlir container)
model/board_detect.py              Task 6  (runs on the board under MaixPy)
model/compare_board.py             Task 6
model/tests/test_*.py              Task 5, 6
model/MANIFEST.md                  Task 6, 7
scripts/build.sh scripts/check_release.sh  Task 8, 11
.github/workflows/ci.yml           Task 10
.github/workflows/release.yml      Task 11
```

---

### Task 0: Board access after the reflash

The board was reflashed on 2026-10-07; the board still has its factory root credentials and no SSH key is installed. ssh alias `maixcam` (→ <board-ip>) is already in `~/.ssh/config`.

- [ ] **Step 1: Install the laptop's key on the board**

Run: `ssh-copy-id -o PubkeyAuthentication=no -o PreferredAuthentications=password root@<board-ip>` (factory default password).

- [ ] **Step 2: Verify key login and image version**

Run: `ssh maixcam 'cat /boot/ver; ls /maixapp/share/font/SourceHanSansCN-Regular.otf; python3 -c "from maix import nn; print(nn.YOLO26)"'`
Expected: a version string containing `4.12.5` (or the image date 2026-01-24), the font path printed, and `<class 'maix._maix.nn.YOLO26'>`.

- [ ] **Step 3: Change the root password**

Run: `ssh -t maixcam passwd` and set a new one (the user chooses). No commit.

---

### Task 1: Repo scaffold + FpsMeter

**Files:**
- Create: `.gitattributes`, `.gitignore`, `LICENSE`, `justfile`, `test/check.hpp`, `main/include/fps_meter.hpp`, `test/test_fps_meter.cpp`

**Interfaces:**
- Produces: `namespace y26 { class FpsMeter { void add(double frame_ms, double infer_ms); std::optional<double> fps() const; std::optional<double> latency_ms() const; }; std::string format_status(std::optional<double> fps, std::optional<double> latency_ms, int model_px); }`
- Produces: `test/check.hpp` with `CHECK(cond)`, `CHECK_EQ(a,b)`, `CHECK_NEAR(a,b,eps)`, `TEST(name)` and `int run_all()`.

- [ ] **Step 1: Scaffold**

`.gitattributes`:
```
*.cvimodel filter=lfs diff=lfs merge=lfs -text
*.png filter=lfs diff=lfs merge=lfs -text
*.jpg filter=lfs diff=lfs merge=lfs -text
```

`.gitignore`:
```
build/
dist/
test/bin/
model/work/
model/.venv/
__pycache__/
*.onnx
*.pt
```

`LICENSE`: copy the Apache-2.0 text: `cp ~/code/MaixCDK-v4125/LICENSE LICENSE` then check `head -3 LICENSE` shows "Apache License".

`justfile`:
```make
set shell := ["bash", "-euo", "pipefail", "-c"]

cxx := env_var_or_default("CXX", "g++")
cxxflags := "-std=c++17 -O1 -g -Wall -Wextra -Werror -fsanitize=address,undefined -fno-omit-frame-pointer -Imain/include -Itest"

# Host tests for the pure headers
test:
    mkdir -p test/bin
    for t in test/test_*.cpp; do \
        n=$(basename "$t" .cpp); \
        {{cxx}} {{cxxflags}} "$t" -o "test/bin/$n" && "test/bin/$n"; \
    done
```

`test/check.hpp`:
```cpp
#pragma once
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace check
{
    struct Case { const char *name; std::function<void()> fn; };
    inline std::vector<Case> &cases() { static std::vector<Case> c; return c; }
    inline int &failures() { static int f = 0; return f; }
    struct Reg { Reg(const char *n, std::function<void()> f) { cases().push_back({n, std::move(f)}); } };
    inline void fail(const char *file, int line, const std::string &msg)
    {
        std::fprintf(stderr, "%s:%d: FAIL %s\n", file, line, msg.c_str());
        ++failures();
    }
    inline int run_all()
    {
        for (auto &c : cases()) c.fn();
        std::printf("%zu tests, %d failures\n", cases().size(), failures());
        return failures() ? 1 : 0;
    }
}

#define CHECK_CAT2(a, b) a##b
#define CHECK_CAT(a, b) CHECK_CAT2(a, b)
#define TEST(name)                                                        \
    static void name();                                                   \
    static check::Reg CHECK_CAT(reg_, name)(#name, name);                 \
    static void name()
#define CHECK(c) do { if (!(c)) check::fail(__FILE__, __LINE__, #c); } while (0)
#define CHECK_EQ(a, b) do { if (!((a) == (b))) check::fail(__FILE__, __LINE__, #a " == " #b); } while (0)
#define CHECK_NEAR(a, b, eps) do { if (std::fabs((a) - (b)) > (eps)) check::fail(__FILE__, __LINE__, #a " ~= " #b); } while (0)
```

- [ ] **Step 2: Write the failing test** — `test/test_fps_meter.cpp`:
```cpp
#include "check.hpp"
#include "fps_meter.hpp"

using y26::FpsMeter;

TEST(empty_has_no_data)
{
    FpsMeter m;
    CHECK(!m.fps().has_value());
    CHECK(!m.latency_ms().has_value());
}

TEST(single_sample)
{
    FpsMeter m;
    m.add(40.0, 25.0);
    CHECK_NEAR(*m.fps(), 25.0, 1e-9);
    CHECK_NEAR(*m.latency_ms(), 25.0, 1e-9);
}

TEST(non_positive_durations_ignored)   // old app: inverted ticks_diff, then 0/0
{
    FpsMeter m;
    m.add(0.0, 0.0);
    m.add(-12.0, -3.0);
    CHECK(!m.fps().has_value());
    CHECK(!m.latency_ms().has_value());
    m.add(50.0, 0.0);
    CHECK_NEAR(*m.fps(), 20.0, 1e-9);
    CHECK(!m.latency_ms().has_value());
}

TEST(window_wraps_to_last_30)
{
    FpsMeter m;
    for (int i = 0; i < 30; ++i) m.add(100.0, 10.0);
    for (int i = 0; i < 30; ++i) m.add(20.0, 5.0);
    CHECK_NEAR(*m.fps(), 50.0, 1e-9);
    CHECK_NEAR(*m.latency_ms(), 5.0, 1e-9);
}

TEST(latency_independent_of_fps)
{
    FpsMeter m;
    m.add(100.0, 30.0);
    m.add(100.0, 50.0);
    CHECK_NEAR(*m.fps(), 10.0, 1e-9);
    CHECK_NEAR(*m.latency_ms(), 40.0, 1e-9);
}

TEST(format_status_values)
{
    CHECK_EQ(y26::format_status(24.06, 38.4, 320), std::string("FPS 24.1 | 38 ms | 320"));
}

TEST(format_status_missing)
{
    CHECK_EQ(y26::format_status(std::nullopt, std::nullopt, 640), std::string("FPS - | - ms | 640"));
}

int main() { return check::run_all(); }
```

- [ ] **Step 3: Run test to verify it fails**

Run: `just test`
Expected: compile error `fps_meter.hpp: No such file or directory`.

- [ ] **Step 4: Implement** — `main/include/fps_meter.hpp`:
```cpp
// Rolling frame-rate and inference-latency meter. Pure C++, host-tested.
#pragma once
#include <array>
#include <cstddef>
#include <cstdio>
#include <optional>
#include <string>

namespace y26
{
    class FpsMeter
    {
    public:
        static constexpr std::size_t kWindow = 30;

        // Non-positive durations (first frame, clock steps) are dropped.
        void add(double frame_ms, double infer_ms)
        {
            if (frame_ms > 0) frames_.push(frame_ms);
            if (infer_ms > 0) infers_.push(infer_ms);
        }

        std::optional<double> fps() const
        {
            auto m = frames_.mean();
            if (!m) return std::nullopt;
            return 1000.0 / *m;
        }

        std::optional<double> latency_ms() const { return infers_.mean(); }

    private:
        struct Ring
        {
            std::array<double, kWindow> v{};
            std::size_t n = 0, i = 0;
            void push(double x) { v[i] = x; i = (i + 1) % kWindow; if (n < kWindow) ++n; }
            std::optional<double> mean() const
            {
                if (n == 0) return std::nullopt;
                double s = 0;
                for (std::size_t k = 0; k < n; ++k) s += v[k];
                return s / static_cast<double>(n);
            }
        };
        Ring frames_, infers_;
    };

    inline std::string format_status(std::optional<double> fps, std::optional<double> latency_ms, int model_px)
    {
        char f[16] = "-", l[16] = "-";
        if (fps) std::snprintf(f, sizeof f, "%.1f", *fps);
        if (latency_ms) std::snprintf(l, sizeof l, "%.0f", *latency_ms);
        char out[64];
        std::snprintf(out, sizeof out, "FPS %s | %s ms | %d", f, l, model_px);
        return out;
    }
}
```

- [ ] **Step 5: Run tests**

Run: `just test`
Expected: `7 tests, 0 failures`.

- [ ] **Step 6: Commit**
```bash
git lfs install --local
git add .gitattributes .gitignore LICENSE justfile test/check.hpp test/test_fps_meter.cpp main/include/fps_meter.hpp
git commit -m "Add host test harness and FpsMeter" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Controls (layout, hit-test, press edge, threshold, model size)

**Files:**
- Create: `main/include/controls.hpp`, `test/test_controls.cpp`

**Interfaces:**
- Produces:
  - `struct Rect { int x, y, w, h; bool contains(int px, int py) const; };`
  - `enum class Button { None, Exit, Model, ConfDown, ConfUp, Snapshot };`
  - `struct Layout { Rect exit, model, conf_down, conf_value, conf_up, snapshot; int status_x, status_y; };`
  - `Layout make_layout(int screen_w, int screen_h);`
  - `Button hit_test(const Layout &, int x, int y);`
  - `class PressEdge { bool update(bool pressed); };` — true only on the press transition.
  - `class Threshold { Threshold(); int pct() const; float value() const; void up(); void down(); std::string to_pct_string() const; static Threshold from_pct_string(const std::string &); static constexpr int kMin = 45, kMax = 95, kStep = 5, kDefault = 50; };`
  - `enum class ModelSize { S320, S640 }; int pixels(ModelSize); ModelSize other(ModelSize); std::optional<ModelSize> parse_model_size(const std::string &); std::string to_string(ModelSize);`

- [ ] **Step 1: Write the failing test** — `test/test_controls.cpp`:
```cpp
#include "check.hpp"
#include "controls.hpp"
#include <vector>

using namespace y26;

static std::vector<Rect> buttons(const Layout &l)
{
    return {l.exit, l.model, l.conf_down, l.conf_value, l.conf_up, l.snapshot};
}

static bool overlap(const Rect &a, const Rect &b)
{
    return a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;
}

static void check_layout(int sw, int sh)
{
    Layout l = make_layout(sw, sh);
    auto r = buttons(l);
    for (auto &b : r)
    {
        CHECK(b.x >= 0 && b.y >= 0 && b.x + b.w <= sw && b.y + b.h <= sh);
        CHECK(b.h >= 48);
        CHECK(b.w > 0);
    }
    for (size_t i = 0; i < r.size(); ++i)
        for (size_t j = i + 1; j < r.size(); ++j)
            CHECK(!overlap(r[i], r[j]));
    CHECK(l.status_y > l.exit.y + l.exit.h);
    CHECK(l.status_y < l.conf_down.y);
}

TEST(layout_552x368) { check_layout(552, 368); }
TEST(layout_320x240) { check_layout(320, 240); }

TEST(hit_test_edges)
{
    Layout l = make_layout(552, 368);
    const Rect &e = l.exit;
    CHECK(hit_test(l, e.x, e.y) == Button::Exit);
    CHECK(hit_test(l, e.x + e.w - 1, e.y + e.h - 1) == Button::Exit);
    CHECK(hit_test(l, e.x + e.w, e.y) != Button::Exit);
    CHECK(hit_test(l, e.x, e.y + e.h) != Button::Exit);
    CHECK(hit_test(l, l.model.x + 1, l.model.y + 1) == Button::Model);
    CHECK(hit_test(l, l.conf_down.x + 1, l.conf_down.y + 1) == Button::ConfDown);
    CHECK(hit_test(l, l.conf_up.x + 1, l.conf_up.y + 1) == Button::ConfUp);
    CHECK(hit_test(l, l.snapshot.x + 1, l.snapshot.y + 1) == Button::Snapshot);
    CHECK(hit_test(l, l.conf_value.x + 1, l.conf_value.y + 1) == Button::None);  // label, not a button
    CHECK(hit_test(l, 276, 184) == Button::None);
    CHECK(hit_test(l, -1, -1) == Button::None);
}

TEST(press_edge_fires_once_while_held)   // Review Focus 1
{
    PressEdge p;
    CHECK(!p.update(false));
    CHECK(p.update(true));
    for (int i = 0; i < 50; ++i) CHECK(!p.update(true));   // held through a slow model load
    CHECK(!p.update(false));
    CHECK(p.update(true));
}

TEST(threshold_default_and_clamp)
{
    Threshold t;
    CHECK_EQ(t.pct(), 50);
    CHECK_NEAR(t.value(), 0.50f, 1e-6);
    for (int i = 0; i < 20; ++i) t.down();
    CHECK_EQ(t.pct(), 45);
    for (int i = 0; i < 20; ++i) t.up();
    CHECK_EQ(t.pct(), 95);
    t.down();
    CHECK_EQ(t.pct(), 90);
}

TEST(threshold_persistence_round_trip)
{
    Threshold t;
    t.up(); t.up();
    CHECK_EQ(t.to_pct_string(), std::string("60"));
    CHECK_EQ(Threshold::from_pct_string("60").pct(), 60);
}

TEST(threshold_bad_persisted_values_fall_back)   // Review Focus 2
{
    CHECK_EQ(Threshold::from_pct_string("").pct(), 50);
    CHECK_EQ(Threshold::from_pct_string("abc").pct(), 50);
    CHECK_EQ(Threshold::from_pct_string("7").pct(), 50);      // below range
    CHECK_EQ(Threshold::from_pct_string("100").pct(), 50);    // above range
    CHECK_EQ(Threshold::from_pct_string("62").pct(), 50);     // off-step
    CHECK_EQ(Threshold::from_pct_string("60x").pct(), 50);
    CHECK_EQ(Threshold::from_pct_string("99999999999999999999").pct(), 50);
}

TEST(model_size)
{
    CHECK_EQ(pixels(ModelSize::S320), 320);
    CHECK_EQ(pixels(ModelSize::S640), 640);
    CHECK(other(ModelSize::S320) == ModelSize::S640);
    CHECK(other(ModelSize::S640) == ModelSize::S320);
    CHECK(parse_model_size("320") == ModelSize::S320);
    CHECK(parse_model_size("640") == ModelSize::S640);
    CHECK(!parse_model_size("1024").has_value());   // Review Focus 2
    CHECK(!parse_model_size("").has_value());
    CHECK_EQ(to_string(ModelSize::S640), std::string("640"));
}

int main() { return check::run_all(); }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `just test`
Expected: compile error `controls.hpp: No such file or directory`.

- [ ] **Step 3: Implement** — `main/include/controls.hpp`:
```cpp
// Touch controls: layout, hit-testing, press edge, confidence threshold,
// model size. Pure C++, host-tested; main.cpp only draws and reads touch.
#pragma once
#include <algorithm>
#include <optional>
#include <string>

namespace y26
{
    struct Rect
    {
        int x, y, w, h;
        bool contains(int px, int py) const { return px >= x && px < x + w && py >= y && py < y + h; }
    };

    enum class Button { None, Exit, Model, ConfDown, ConfUp, Snapshot };

    struct Layout
    {
        Rect exit, model, conf_down, conf_value, conf_up, snapshot;
        int status_x, status_y;
    };

    // Top row: Exit (left), Model (right). Bottom row: conf -, value, conf + (left),
    // Snapshot (right). Status line under the top row. Sizes scale with the screen
    // but never drop below 48 px tall.
    inline Layout make_layout(int sw, int sh)
    {
        const int m = 4;
        const int bh = std::max(48, sh / 7);
        const int unit = sw / 8;               // 69 px on 552, 40 on 320
        const int top = m, bottom = sh - m - bh;
        Layout l{};
        l.exit = {m, top, unit * 2, bh};
        l.model = {sw - m - unit * 3 / 2, top, unit * 3 / 2, bh};
        l.conf_down = {m, bottom, unit, bh};
        l.conf_value = {m + unit, bottom, unit * 3 / 2, bh};
        l.conf_up = {m + unit + unit * 3 / 2, bottom, unit, bh};
        l.snapshot = {sw - m - unit * 2, bottom, unit * 2, bh};
        l.status_x = m * 2;
        l.status_y = top + bh + m * 2;
        return l;
    }

    inline Button hit_test(const Layout &l, int x, int y)
    {
        if (l.exit.contains(x, y)) return Button::Exit;
        if (l.model.contains(x, y)) return Button::Model;
        if (l.conf_down.contains(x, y)) return Button::ConfDown;
        if (l.conf_up.contains(x, y)) return Button::ConfUp;
        if (l.snapshot.contains(x, y)) return Button::Snapshot;
        return Button::None;
    }

    class PressEdge
    {
    public:
        bool update(bool pressed)
        {
            bool edge = pressed && !was_;
            was_ = pressed;
            return edge;
        }

    private:
        bool was_ = false;
    };

    // Stored as integer percent so repeated steps never drift.
    class Threshold
    {
    public:
        static constexpr int kMin = 45, kMax = 95, kStep = 5, kDefault = 50;

        Threshold() = default;
        int pct() const { return pct_; }
        float value() const { return static_cast<float>(pct_) / 100.0f; }
        void up() { pct_ = std::min(kMax, pct_ + kStep); }
        void down() { pct_ = std::max(kMin, pct_ - kStep); }
        std::string to_pct_string() const { return std::to_string(pct_); }

        static Threshold from_pct_string(const std::string &s)
        {
            Threshold t;
            if (s.empty() || s.size() > 3) return t;
            int v = 0;
            for (char c : s)
            {
                if (c < '0' || c > '9') return t;
                v = v * 10 + (c - '0');
            }
            if (v < kMin || v > kMax || v % kStep != 0) return t;
            t.pct_ = v;
            return t;
        }

    private:
        int pct_ = kDefault;
    };

    enum class ModelSize { S320, S640 };

    inline int pixels(ModelSize m) { return m == ModelSize::S320 ? 320 : 640; }
    inline ModelSize other(ModelSize m) { return m == ModelSize::S320 ? ModelSize::S640 : ModelSize::S320; }
    inline std::string to_string(ModelSize m) { return std::to_string(pixels(m)); }
    inline std::optional<ModelSize> parse_model_size(const std::string &s)
    {
        if (s == "320") return ModelSize::S320;
        if (s == "640") return ModelSize::S640;
        return std::nullopt;
    }
}
```

- [ ] **Step 4: Run tests**

Run: `just test`
Expected: both binaries report `0 failures`. If `layout_320x240` fails on the status line check, the arithmetic above is wrong — fix `make_layout`, not the test.

- [ ] **Step 5: Commit**
```bash
git add main/include/controls.hpp test/test_controls.cpp
git commit -m "Add touch layout, hit-testing, press edge, threshold and model size" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Strings (en/zh)

**Files:**
- Create: `main/include/strings.hpp`, `test/test_strings.cpp`

**Interfaces:**
- Produces: `enum class Lang { En, Zh }; Lang lang_from_locale(const std::string &); enum class Str { Exit, Snapshot, Conf, Saved, SaveFailed, Loading, Fallback320, NoModel, Count_ }; const char *tr(Str, Lang);`

- [ ] **Step 1: Write the failing test** — `test/test_strings.cpp`:
```cpp
#include "check.hpp"
#include "strings.hpp"
#include <cstring>

using namespace y26;

TEST(every_key_has_both_languages)
{
    for (int i = 0; i < static_cast<int>(Str::Count_); ++i)
    {
        auto k = static_cast<Str>(i);
        CHECK(tr(k, Lang::En) != nullptr && std::strlen(tr(k, Lang::En)) > 0);
        CHECK(tr(k, Lang::Zh) != nullptr && std::strlen(tr(k, Lang::Zh)) > 0);
    }
}

TEST(locale_selection)
{
    CHECK(lang_from_locale("zh") == Lang::Zh);
    CHECK(lang_from_locale("zh_CN") == Lang::Zh);
    CHECK(lang_from_locale("zh-TW") == Lang::Zh);
    CHECK(lang_from_locale("en") == Lang::En);
    CHECK(lang_from_locale("ja") == Lang::En);
    CHECK(lang_from_locale("") == Lang::En);
    CHECK(lang_from_locale("z") == Lang::En);
}

TEST(english_text)
{
    CHECK_EQ(std::string(tr(Str::Exit, Lang::En)), std::string("< Exit"));
    CHECK_EQ(std::string(tr(Str::Saved, Lang::En)), std::string("Saved"));
}

int main() { return check::run_all(); }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `just test` — Expected: compile error `strings.hpp: No such file or directory`.

- [ ] **Step 3: Implement** — `main/include/strings.hpp`:
```cpp
// On-screen text in English and Chinese. Pure C++, host-tested.
#pragma once
#include <array>
#include <string>

namespace y26
{
    enum class Lang { En, Zh };

    inline Lang lang_from_locale(const std::string &loc)
    {
        return loc.rfind("zh", 0) == 0 ? Lang::Zh : Lang::En;
    }

    enum class Str { Exit, Snapshot, Conf, Saved, SaveFailed, Loading, Fallback320, NoModel, Count_ };

    inline const char *tr(Str s, Lang l)
    {
        static const std::array<std::array<const char *, 2>, static_cast<size_t>(Str::Count_)> t = {{
            {"< Exit", "< 退出"},
            {"Snapshot", "拍照"},
            {"conf", "置信度"},
            {"Saved", "已保存"},
            {"Save failed", "保存失败"},
            {"Loading...", "加载中..."},
            {"640 model failed, using 320", "640 模型加载失败，改用 320"},
            {"Model not found:", "找不到模型："},
        }};
        return t[static_cast<size_t>(s)][l == Lang::Zh ? 1 : 0];
    }
}
```

- [ ] **Step 4: Run tests** — `just test` — Expected: all binaries `0 failures`.

- [ ] **Step 5: Commit**
```bash
git add main/include/strings.hpp test/test_strings.cpp
git commit -m "Add en/zh string table" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Snapshot naming

**Files:**
- Create: `main/include/snapshot.hpp`, `test/test_snapshot.cpp`

**Interfaces:**
- Produces: `int next_snapshot_index(const std::vector<std::string> &filenames); std::string snapshot_name(int index);` (`"yolo26_<n>.jpg"`)

- [ ] **Step 1: Write the failing test** — `test/test_snapshot.cpp`:
```cpp
#include "check.hpp"
#include "snapshot.hpp"

using namespace y26;

TEST(empty_dir_starts_at_1) { CHECK_EQ(next_snapshot_index({}), 1); }

TEST(max_plus_one)
{
    CHECK_EQ(next_snapshot_index({"yolo26_1.jpg", "yolo26_7.jpg", "yolo26_3.jpg"}), 8);
    CHECK_EQ(next_snapshot_index({"yolo26_007.jpg"}), 8);
}

TEST(ignores_odd_names)   // Review Focus 3
{
    CHECK_EQ(next_snapshot_index({"yolo26_.jpg", "yolo26_12a.jpg", "yolo26_5.png", "notes.txt",
                                  "yolo26_-3.jpg", "Yolo26_9.jpg", "yolo26_99999999999999.jpg"}), 1);
}

TEST(name_format) { CHECK_EQ(snapshot_name(42), std::string("yolo26_42.jpg")); }

int main() { return check::run_all(); }
```

- [ ] **Step 2: Run to fail** — `just test` — Expected: `snapshot.hpp: No such file or directory`.

- [ ] **Step 3: Implement** — `main/include/snapshot.hpp`:
```cpp
// Snapshot file naming. Numbered, not timestamped: the RTC may be unset.
#pragma once
#include <string>
#include <vector>

namespace y26
{
    inline std::string snapshot_name(int index) { return "yolo26_" + std::to_string(index) + ".jpg"; }

    // Highest valid "yolo26_<digits>.jpg" + 1. Names with non-digits, no digits,
    // or more than 9 digits (would overflow int) are ignored.
    inline int next_snapshot_index(const std::vector<std::string> &names)
    {
        const std::string pre = "yolo26_", suf = ".jpg";
        int best = 0;
        for (const auto &n : names)
        {
            if (n.size() <= pre.size() + suf.size()) continue;
            if (n.compare(0, pre.size(), pre) != 0) continue;
            if (n.compare(n.size() - suf.size(), suf.size(), suf) != 0) continue;
            std::string digits = n.substr(pre.size(), n.size() - pre.size() - suf.size());
            if (digits.empty() || digits.size() > 9) continue;
            int v = 0;
            bool ok = true;
            for (char c : digits)
            {
                if (c < '0' || c > '9') { ok = false; break; }
                v = v * 10 + (c - '0');
            }
            if (ok && v > best) best = v;
        }
        return best + 1;
    }
}
```

- [ ] **Step 4: Run tests** — `just test` — Expected: all `0 failures`.

- [ ] **Step 5: Commit**
```bash
git add main/include/snapshot.hpp test/test_snapshot.cpp
git commit -m "Add snapshot naming" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Model pipeline — image lists, export, Python decoder, mAP

Runs on the GPU box. Set up once:
```bash
rsync -a --exclude .git ~/code/maixcam-yolo26/ <gpu-box>:~/maixcam-yolo26/
ssh <gpu-box> 'cd ~/maixcam-yolo26/model && uv venv --python 3.12 .venv && uv pip install -p .venv -e ".[dev]"'
ssh <gpu-box> 'mkdir -p ~/coco && cd ~/coco && [ -d val2017 ] || (curl -sLO http://images.cocodataset.org/zips/val2017.zip && curl -sLO http://images.cocodataset.org/annotations/annotations_trainval2017.zip && unzip -q val2017.zip && unzip -q annotations_trainval2017.zip)'
```
Develop locally, run tests both locally and on .30 (pure tests need no GPU).

**Files:**
- Create: `model/pyproject.toml`, `model/sources.toml`, `model/select_images.py`, `model/export.py`, `model/decode.py`, `model/eval_map.py`, `model/tests/test_select_images.py`, `model/tests/test_decode.py`, `model/tests/test_eval_map.py`, `model/lists/{calib,eval,board}.txt`

**Interfaces:**
- Produces: `decode.letterbox(img_rgb: np.ndarray, size: int) -> tuple[np.ndarray, float, float, float]` (CHW float32 /255, scale, pad_x, pad_y — black padding, matches `FIT_CONTAIN`)
- Produces: `decode.decode(outputs: dict[str, np.ndarray], names: dict, size: int, conf: float, img_w: int, img_h: int) -> list[Det]` where `Det = (x, y, w, h, class_id, score)` in original-image pixels; mirrors `maix_nn_yolo26.hpp` `_generate_proposals` + `_correct_bbox` including `LOGIT_THRESHOLD = -0.2` and clamp-to-input.
- Produces: `model/work/<size>/output_names.json` — `{"bbox": [s8, s16, s32], "cls": [s8, s16, s32]}` tensor names; `model/work/<size>/yolo26n_<size>_cut.onnx`.
- Produces: `eval_map.map50(dets_by_image: dict[int, list[Det]], ann_file: str, image_ids: list[int]) -> float`
- Produces: `COCO80_TO_91: list[int]` in `eval_map.py`.

- [ ] **Step 1: `model/pyproject.toml`**
```toml
[project]
name = "yolo26-model"
version = "0.0.0"
requires-python = ">=3.12"
dependencies = [
  "ultralytics==8.4.*",
  "onnx>=1.16",
  "onnxruntime>=1.18",
  "onnxsim>=0.4.36",
  "numpy>=1.26",
  "opencv-python-headless>=4.9",
  "pycocotools>=2.0.7",
]
[project.optional-dependencies]
dev = ["pytest>=8"]
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```
After installing on .30 run `.venv/bin/python -c "import ultralytics; print(ultralytics.__version__)"` and replace `8.4.*` with that exact version (`==X.Y.Z`). Ultralytics must be a release that ships `yolo26n.pt`.

- [ ] **Step 2: Write failing tests** — `model/tests/test_select_images.py`:
```python
from select_images import select

NAMES = [f"{i:012d}.jpg" for i in range(5000)]

def test_sizes_and_disjoint():
    s = select(NAMES, seed=20261007)
    assert (len(s["calib"]), len(s["eval"]), len(s["board"])) == (500, 50, 10)
    a, b, c = map(set, (s["calib"], s["eval"], s["board"]))
    assert not (a & b or a & c or b & c)

def test_deterministic_and_order_independent():
    assert select(NAMES, seed=20261007) == select(list(reversed(NAMES)), seed=20261007)

def test_seed_changes_selection():
    assert select(NAMES, seed=1)["eval"] != select(NAMES, seed=2)["eval"]
```
`model/tests/test_decode.py`:
```python
import numpy as np
from decode import decode, letterbox

NAMES = {"bbox": ["b8", "b16", "b32"], "cls": ["c8", "c16", "c32"]}

def blank(size, nc=80):
    out = {}
    for s, b, c in zip((8, 16, 32), NAMES["bbox"], NAMES["cls"]):
        g = size // s
        out[b] = np.zeros((1, 4, g, g), np.float32)
        out[c] = np.full((1, nc, g, g), -10.0, np.float32)
    return out

def test_letterbox_square_is_plain_resize():
    img = np.full((100, 100, 3), 255, np.uint8)
    x, scale, px, py = letterbox(img, 320)
    assert x.shape == (3, 320, 320) and x.dtype == np.float32
    assert (scale, px, py) == (3.2, 0.0, 0.0)
    assert np.allclose(x, 1.0)

def test_letterbox_pads_black():
    img = np.full((160, 320, 3), 255, np.uint8)       # h=160, w=320
    x, scale, px, py = letterbox(img, 320)
    assert (scale, px, py) == (1.0, 0.0, 80.0)
    assert np.allclose(x[:, :80, :], 0.0) and np.allclose(x[:, 80:240, :], 1.0)

def test_single_cell_decode_and_letterbox_inverse():
    out = blank(320)
    # stride-32 grid is 10x10; cell (ax=3, ay=4): centre (112, 144) in input px
    out["c32"][0, 15, 4, 3] = 2.0               # class 15 (cat), sigmoid(2)=0.881
    out["b32"][0, :, 4, 3] = [1.0, 1.0, 1.0, 1.0]  # l,t,r,b in stride units
    dets = decode(out, NAMES, 320, conf=0.5, img_w=640, img_h=640)
    assert len(dets) == 1
    x, y, w, h, cid, score = dets[0]
    assert cid == 15 and abs(score - 0.8808) < 1e-3
    # input box (80,112,64,64) -> scale 0.5 back to 640x640
    assert np.allclose((x, y, w, h), (160, 224, 128, 128), atol=1e-4)

def test_logit_floor_drops_below_045_even_with_low_conf():
    out = blank(320)
    out["c32"][0, 0, 0, 0] = -0.3               # sigmoid=0.4256 < floor
    out["c16"][0, 0, 0, 0] = -0.1               # sigmoid=0.475, kept at conf 0.3
    out["b32"][0, :, 0, 0] = 1; out["b16"][0, :, 0, 0] = 1
    dets = decode(out, NAMES, 320, conf=0.3, img_w=320, img_h=320)
    assert len(dets) == 1 and abs(dets[0][5] - 0.4750) < 1e-3

def test_clamp_to_input():
    out = blank(320)
    out["c8"][0, 1, 0, 0] = 3.0
    out["b8"][0, :, 0, 0] = [5.0, 5.0, 1.0, 1.0]   # extends left/up past 0
    (x, y, w, h, *_), = decode(out, NAMES, 320, conf=0.5, img_w=320, img_h=320)
    assert x == 0 and y == 0 and w > 0 and h > 0
```
`model/tests/test_eval_map.py`:
```python
import json
from eval_map import COCO80_TO_91, map50

def test_coco_mapping():
    assert len(COCO80_TO_91) == 80 and COCO80_TO_91[0] == 1 and COCO80_TO_91[15] == 17 and COCO80_TO_91[79] == 90

def test_perfect_and_empty(tmp_path):
    ann = {
        "images": [{"id": 1, "width": 100, "height": 100}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 17, "bbox": [10, 10, 30, 30], "area": 900, "iscrowd": 0}],
        "categories": [{"id": c} for c in COCO80_TO_91],
    }
    p = tmp_path / "ann.json"; p.write_text(json.dumps(ann))
    assert map50({1: [(10, 10, 30, 30, 15, 0.9)]}, str(p), [1]) > 0.99
    assert map50({1: []}, str(p), [1]) == 0.0
```

- [ ] **Step 3: Run to fail** — `cd model && .venv/bin/pytest -q` — Expected: `ModuleNotFoundError: No module named 'select_images'` (and decode, eval_map).

- [ ] **Step 4: Implement** — `model/select_images.py`:
```python
"""Deterministic, disjoint COCO val2017 subsets: calib 500, eval 50, board 10."""
import argparse
import random
from pathlib import Path

SEED = 20261007

def select(names: list[str], seed: int = SEED) -> dict[str, list[str]]:
    pool = sorted(names)
    random.Random(seed).shuffle(pool)
    return {"calib": sorted(pool[:500]), "eval": sorted(pool[500:550]), "board": sorted(pool[550:560])}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--out", default=str(Path(__file__).parent / "lists"))
    a = ap.parse_args()
    s = select([p.name for p in Path(a.val_dir).glob("*.jpg")])
    Path(a.out).mkdir(exist_ok=True)
    for k, v in s.items():
        (Path(a.out) / f"{k}.txt").write_text("\n".join(v) + "\n")

if __name__ == "__main__":
    main()
```
`model/decode.py`:
```python
"""Python mirror of maix::nn::YOLO26 post-processing (MaixCDK v4.12.5,
components/nn/include/maix_nn_yolo26.hpp: _generate_proposals, _correct_bbox).
Used to score FP32 (onnxruntime) and INT8 (TPU-MLIR simulator) outputs the
same way the board does. Keep in lockstep with that header."""
import cv2
import numpy as np

LOGIT_THRESHOLD = -0.2

def letterbox(img_rgb: np.ndarray, size: int):
    h, w = img_rgb.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = round(w * scale), round(h * scale)
    px, py = (size - nw) / 2.0, (size - nh) / 2.0
    canvas = np.zeros((size, size, 3), np.uint8)
    x0, y0 = int(px), int(py)
    canvas[y0:y0 + nh, x0:x0 + nw] = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x = canvas.astype(np.float32).transpose(2, 0, 1) / 255.0
    return x, float(scale), float(px), float(py)

def _sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))

def decode(outputs, names, size, conf, img_w, img_h):
    dets = []
    for bname, cname in zip(names["bbox"], names["cls"]):
        b = outputs[bname][0]            # 4,H,W
        c = outputs[cname][0]            # C,H,W
        fh, fw = c.shape[1:]
        stride = size / fw
        max_logit = c.max(axis=0)
        cls_id = c.argmax(axis=0)
        ys, xs = np.nonzero(max_logit >= LOGIT_THRESHOLD)
        for ay, ax in zip(ys, xs):
            score = float(_sigmoid(max_logit[ay, ax]))
            if score <= conf:
                continue
            l, t, r, btm = b[:, ay, ax]
            cx, cy = (ax + 0.5) * stride, (ay + 0.5) * stride
            x, y = max(0.0, cx - l * stride), max(0.0, cy - t * stride)
            w = min((l + r) * stride, size - x)
            h = min((t + btm) * stride, size - y)
            if w > 0 and h > 0:
                dets.append([x, y, w, h, int(cls_id[ay, ax]), score])
    scale = min(size / img_w, size / img_h)
    pad_w, pad_h = (size - img_w * scale) / 2.0, (size - img_h * scale) / 2.0
    out = []
    for x, y, w, h, k, s in dets:
        out.append(((x - pad_w) / scale, (y - pad_h) / scale, w / scale, h / scale, k, s))
    return out
```
`model/eval_map.py`:
```python
"""COCO mAP50 over a fixed image subset, using pycocotools."""
import contextlib
import io
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

COCO80_TO_91 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51,
                52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77,
                78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

def map50(dets_by_image, ann_file, image_ids):
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(ann_file)
        res = [{"image_id": i, "category_id": COCO80_TO_91[k], "bbox": [x, y, w, h], "score": s}
               for i, ds in dets_by_image.items() for (x, y, w, h, k, s) in ds]
        if not res:
            return 0.0
        dt = gt.loadRes(res)
        ev = COCOeval(gt, dt, "bbox")
        ev.params.imgIds = list(image_ids)
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return float(ev.stats[1])
```

- [ ] **Step 5: Run tests** — `cd model && .venv/bin/pytest -q` — Expected: all pass. (If `test_letterbox_*` is off by one on odd sizes, match the board: `FIT_CONTAIN` centres with integer offset `int(pad)`.)

- [ ] **Step 6: Export script** — `model/export.py`:
```python
"""yolo26n.pt -> ONNX at <size> -> cut at the six one2one head convs.
The e2e NMS-free selection block (TopK/GatherElements/Mod) does not compile
for cv181x; maix::nn::YOLO26 decodes the six head outputs itself."""
import argparse
import hashlib
import json
import re
import shutil
import tomllib
from pathlib import Path

import onnx
from onnx.utils import Extractor
from ultralytics import YOLO

HERE = Path(__file__).parent
PAT = re.compile(r"/model\.\d+/one2one_cv([23])\.(\d)/one2one_cv\1\.\2\.2/Conv_output_0$")

def head_outputs(model: onnx.ModelProto) -> dict:
    found = {}
    for node in model.graph.node:
        for o in node.output:
            m = PAT.search(o)
            if m:
                found[("bbox" if m.group(1) == "2" else "cls", int(m.group(2)))] = o
    names = {"bbox": [found[("bbox", i)] for i in range(3)], "cls": [found[("cls", i)] for i in range(3)]}
    return names

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, choices=[320, 640], required=True)
    a = ap.parse_args()
    src = tomllib.loads((HERE / "sources.toml").read_text())
    work = HERE / "work" / str(a.size)
    work.mkdir(parents=True, exist_ok=True)
    pt = HERE / "work" / "yolo26n.pt"
    if not pt.exists():
        YOLO("yolo26n.pt")                      # ultralytics downloads into cwd
        shutil.move("yolo26n.pt", pt)
    digest = hashlib.sha256(pt.read_bytes()).hexdigest()
    if digest != src["yolo26n"]["sha256"]:
        raise SystemExit(f"yolo26n.pt sha256 {digest} != pinned {src['yolo26n']['sha256']}")
    full = Path(YOLO(str(pt)).export(format="onnx", imgsz=a.size, opset=src["export"]["opset"], simplify=True))
    shutil.move(full, work / f"yolo26n_{a.size}_full.onnx")
    m = onnx.load(work / f"yolo26n_{a.size}_full.onnx")
    names = head_outputs(m)
    cut = Extractor(m).extract_model(["images"], names["bbox"] + names["cls"])
    onnx.save(cut, work / f"yolo26n_{a.size}_cut.onnx")
    (work / "output_names.json").write_text(json.dumps(names, indent=2))
    print(json.dumps(names, indent=2))

if __name__ == "__main__":
    main()
```
`model/sources.toml` (first run pins the hash):
```toml
[yolo26n]
file = "yolo26n.pt"
sha256 = "PIN-ME"

[export]
opset = 17
```
On .30: `cd ~/maixcam-yolo26/model && .venv/bin/python -c "from ultralytics import YOLO; YOLO('yolo26n.pt')" && mkdir -p work && mv yolo26n.pt work/ && sha256sum work/yolo26n.pt` — write that hash into `sources.toml` replacing `PIN-ME`, then run `.venv/bin/python export.py --size 320`.
Expected: prints six tensor names; the `cls` ones are 80-channel. If `head_outputs` raises `KeyError`, inspect names with `python -c "import onnx;[print(o) for n in onnx.load('work/320/yolo26n_320_full.onnx').graph.node for o in n.output if 'one2one' in o]"` and fix `PAT`.

- [ ] **Step 7: FP32 sanity on the eval list** — add to `eval_map.py`:
```python
def fp32_map50(size: int, coco_dir: str, conf: float = 0.45) -> float:
    import json as _json
    from pathlib import Path
    import cv2
    import onnxruntime as ort
    from decode import decode, letterbox
    here = Path(__file__).parent
    work = here / "work" / str(size)
    names = _json.loads((work / "output_names.json").read_text())
    sess = ort.InferenceSession(str(work / f"yolo26n_{size}_cut.onnx"), providers=["CPUExecutionProvider"])
    outs = [o.name for o in sess.get_outputs()]
    files = (here / "lists" / "eval.txt").read_text().split()
    dets = {}
    for f in files:
        img = cv2.cvtColor(cv2.imread(f"{coco_dir}/val2017/{f}"), cv2.COLOR_BGR2RGB)
        x, *_ = letterbox(img, size)
        res = dict(zip(outs, sess.run(outs, {"images": x[None]})))
        dets[int(f[:-4])] = decode(res, names, size, conf, img.shape[1], img.shape[0])
    return map50(dets, f"{coco_dir}/annotations/instances_val2017.json", list(dets))

if __name__ == "__main__":
    import sys
    print(f"fp32 mAP50 @{sys.argv[1]}: {fp32_map50(int(sys.argv[1]), sys.argv[2]):.4f}")
```
Run on .30: `.venv/bin/python select_images.py --val-dir ~/coco/val2017 && .venv/bin/python eval_map.py 320 ~/coco`
Expected: a mAP50 well above 0.3 (YOLO26n COCO mAP50 is ~0.55 at 640 over the full set; 320 and conf 0.45 lower it). A value near 0 means the decoder or the cut outputs are wrong — stop and fix before Task 6.

- [ ] **Step 8: Commit** (lists are text; nothing under `work/` is committed)
```bash
git add model/pyproject.toml model/sources.toml model/select_images.py model/export.py model/decode.py model/eval_map.py model/tests model/lists
git commit -m "Model pipeline: image lists, ONNX export and cut, decoder mirror, mAP50" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: INT8 conversion, simulator check, and CHECKPOINT 1 on the board (320)

**Files:**
- Create: `model/convert.sh`, `model/sim_detect.py`, `model/board_detect.py`, `model/compare_board.py`, `model/tests/test_compare_board.py`, `assets/yolo26n_320.mud`, `assets/yolo26n_320_int8.cvimodel` (LFS), `model/MANIFEST.md`, `model/board_images/*.jpg` (LFS, the 10 board images)

**Interfaces:**
- Consumes: `model/work/<size>/yolo26n_<size>_cut.onnx`, `output_names.json`, `decode.decode`, `decode.letterbox`, `eval_map.map50`, `model/lists/*.txt`.
- Produces: `model/work/<size>/yolo26n_<size>_int8.cvimodel`, `model/work/<size>/sim_board.json`, `model/work/<size>/sim_eval_map50.txt`.
- Produces: `compare_board.compare(sim: dict, board: dict, iou_min=0.9) -> list[str]` (empty list = pass). JSON format for both: `{"<file>": [[x, y, w, h, class_id, score], ...]}`.

- [ ] **Step 1: `model/convert.sh`**
```bash
#!/usr/bin/env bash
# Cut ONNX -> INT8 .cvimodel for cv181x via TPU-MLIR in Docker.
# No --fuse_preprocess: the .mud mean/scale normalise input on the board
# (nn::YOLO26::detect passes them to forward_image).
set -euo pipefail
SIZE=${1:?usage: convert.sh 320|640}
COCO=${COCO:-$HOME/coco}
HERE="$(cd "$(dirname "$0")" && pwd)"
W="$HERE/work/$SIZE"
IMG=maixcam-yolo26/tpu_mlir:1.28.1
docker image inspect "$IMG" >/dev/null 2>&1 || docker build -t "$IMG" - <<'EOF'
FROM sophgo/tpuc_dev:latest
RUN pip install --no-cache-dir tpu_mlir[onnx]==1.28.1
EOF
rm -rf "$W/cali" && mkdir -p "$W/cali"
while read -r f; do ln -f "$COCO/val2017/$f" "$W/cali/$f" 2>/dev/null || cp "$COCO/val2017/$f" "$W/cali/"; done < "$HERE/lists/calib.txt"
cp "$COCO/val2017/$(head -1 "$HERE/lists/eval.txt")" "$W/test_input.jpg"
OUT=$(python3 -c "import json;d=json.load(open('$W/output_names.json'));print(','.join(d['bbox']+d['cls']))")
docker run --rm -v "$W":/work -w /work "$IMG" bash -c "
set -e
model_transform.py --model_name yolo26n_$SIZE --model_def yolo26n_${SIZE}_cut.onnx \
  --input_shapes [[1,3,$SIZE,$SIZE]] --output_names $OUT \
  --mean 0,0,0 --scale 0.00392156862745098,0.00392156862745098,0.00392156862745098 \
  --pixel_format rgb --keep_aspect_ratio --pad_value 0 \
  --test_input test_input.jpg --test_result top_outputs.npz --mlir yolo26n_$SIZE.mlir
run_calibration.py yolo26n_$SIZE.mlir --dataset cali --input_num 500 -o cali_table
model_deploy.py --mlir yolo26n_$SIZE.mlir --quantize INT8 --calibration_table cali_table \
  --processor cv181x --test_input yolo26n_${SIZE}_in_f32.npz --test_reference top_outputs.npz \
  --tolerance 0.85,0.45 --model yolo26n_${SIZE}_int8.cvimodel
" 2>&1 | tee "$W/convert.log"
ls -l "$W/yolo26n_${SIZE}_int8.cvimodel"
```
Run on .30: `chmod +x model/convert.sh && model/convert.sh 320`
Expected: ends with `npz compare PASSED` lines from model_deploy and lists the `.cvimodel` (~3 MB). A tolerance failure stops the task: report the log tail.

- [ ] **Step 2: Simulator detections** — `model/sim_detect.py` (runs inside the tpu_mlir container):
```python
"""INT8 detections from the TPU-MLIR simulator, decoded like the board.
Usage (in container): python3 sim_detect.py --size 320 --list lists/eval.txt --coco /coco --out work/320/sim_eval.json"""
import argparse
import glob
import json
from pathlib import Path

import cv2
from tpu_mlir.python.tools.model_runner import mlir_inference

from decode import decode, letterbox

ap = argparse.ArgumentParser()
ap.add_argument("--size", type=int, required=True)
ap.add_argument("--list", required=True)
ap.add_argument("--coco", required=True)
ap.add_argument("--conf", type=float, default=0.45)
ap.add_argument("--out", required=True)
a = ap.parse_args()
here = Path(__file__).parent
work = here / "work" / str(a.size)
mlir = glob.glob(str(work / "*_int8_sym_tpu.mlir"))
assert len(mlir) == 1, mlir
names = json.loads((work / "output_names.json").read_text())
res = {}
for f in Path(a.list).read_text().split():
    img = cv2.cvtColor(cv2.imread(f"{a.coco}/val2017/{f}"), cv2.COLOR_BGR2RGB)
    x, *_ = letterbox(img, a.size)
    out = mlir_inference({"images": x[None]}, mlir[0], dump_all=False)
    out = {k.removesuffix("_f32"): v for k, v in out.items()}
    res[f] = [list(map(float, d[:4])) + [int(d[4]), float(d[5])]
              for d in decode(out, names, a.size, a.conf, img.shape[1], img.shape[0])]
Path(a.out).write_text(json.dumps(res))
print(f"{len(res)} images -> {a.out}")
```
Run on .30:
```bash
cd ~/maixcam-yolo26/model
for L in eval board; do docker run --rm -v "$PWD":/m -v ~/coco:/coco -w /m maixcam-yolo26/tpu_mlir:1.28.1 \
  python3 sim_detect.py --size 320 --list lists/$L.txt --coco /coco --out work/320/sim_$L.json; done
.venv/bin/python - <<'EOF'
import json, os
from eval_map import map50, fp32_map50
COCO = os.path.expanduser("~/coco")
d = json.load(open("work/320/sim_eval.json"))
i8 = map50({int(k[:-4]): [tuple(v) for v in vs] for k, vs in d.items()}, f"{COCO}/annotations/instances_val2017.json", [int(k[:-4]) for k in d])
f = fp32_map50(320, COCO)
print(f"fp32 {f:.4f} int8 {i8:.4f} drop {100*(f-i8):.2f} pts")
open("work/320/sim_eval_map50.txt", "w").write(f"{f:.4f} {i8:.4f}\n")
assert f - i8 <= 0.03, "INT8 drop > 3 points"
EOF
```
Expected: `drop` ≤ 3.00 pts and no assertion. If the output keys don't match `output_names.json` (the runner may suffix names), print `out.keys()` once and adjust the `removesuffix` line.

- [ ] **Step 3: `.mud` and board script**

`assets/yolo26n_320.mud` (labels line = the 80 COCO names, exactly as in the old app's `.mud`):
```ini
[basic]
type = cvimodel
model = yolo26n_320_int8.cvimodel

[extra]
model_type = yolo26
input_type = rgb
mean = 0, 0, 0
scale = 0.00392156862745098, 0.00392156862745098, 0.00392156862745098
labels = person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush
```
`model/board_detect.py` (runs on the board with MaixPy, which wraps the same C++ `nn::YOLO26`):
```python
"""On-board detections for the board image list. Usage on the board:
python3 board_detect.py <model.mud> <image dir> <conf> > out.json"""
import json
import os
import sys
from maix import image, nn

det = nn.YOLO26(model=sys.argv[1], dual_buff=False)
conf = float(sys.argv[3])
res = {}
for f in sorted(os.listdir(sys.argv[2])):
    if not f.endswith(".jpg"):
        continue
    img = image.load(os.path.join(sys.argv[2], f), format=image.Format.FMT_RGB888)
    objs = det.detect(img, conf_th=conf, iou_th=0.45, fit=image.Fit.FIT_CONTAIN)
    res[f] = [[o.x, o.y, o.w, o.h, o.class_id, o.score] for o in objs]
print(json.dumps(res))
```

- [ ] **Step 4: Write failing comparison test** — `model/tests/test_compare_board.py`:
```python
from compare_board import compare, iou

def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 5, 5)) == 0.0

def test_pass_with_reordered_dets():
    sim = {"a.jpg": [[0, 0, 10, 10, 15, 0.9], [50, 50, 10, 10, 0, 0.8]]}
    board = {"a.jpg": [[50, 50, 10, 10.2, 0, 0.79], [0, 0, 10, 10, 15, 0.91]]}
    assert compare(sim, board) == []

def test_fail_count_class_iou_and_missing_image():
    sim = {"a.jpg": [[0, 0, 10, 10, 15, 0.9]], "b.jpg": [[0, 0, 10, 10, 1, 0.9]], "c.jpg": []}
    board = {"a.jpg": [[0, 0, 10, 10, 16, 0.9]], "b.jpg": [[0, 0, 5, 5, 1, 0.9]]}
    errs = compare(sim, board)
    assert any("a.jpg" in e for e in errs)
    assert any("b.jpg" in e for e in errs)
    assert any("c.jpg" in e for e in errs)
```
Run: `cd model && .venv/bin/pytest -q tests/test_compare_board.py` — Expected: `ModuleNotFoundError: compare_board`.

- [ ] **Step 5: Implement** — `model/compare_board.py`:
```python
"""Board vs simulator: same detection count, and greedy one-to-one matching
by IoU with the same class and IoU >= iou_min for every detection."""
import json
import sys

def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0

def compare(sim, board, iou_min=0.9):
    errs = []
    for f, sd in sim.items():
        if f not in board:
            errs.append(f"{f}: missing from board output")
            continue
        bd = list(board[f])
        if len(sd) != len(bd):
            errs.append(f"{f}: {len(sd)} sim dets vs {len(bd)} board dets")
            continue
        for s in sorted(sd, key=lambda d: -d[5]):
            best = max(bd, key=lambda b: iou(s, b) if b[4] == s[4] else -1.0, default=None)
            if best is None or best[4] != s[4] or iou(s, best) < iou_min:
                errs.append(f"{f}: no board match for class {s[4]} box {s[:4]}")
                continue
            bd.remove(best)
    return errs

if __name__ == "__main__":
    errs = compare(json.load(open(sys.argv[1])), json.load(open(sys.argv[2])))
    print("\n".join(errs) or "board matches simulator")
    sys.exit(1 if errs else 0)
```
Run: `.venv/bin/pytest -q` — Expected: all pass.

- [ ] **Step 6: CHECKPOINT 1 — run on the board**
```bash
cd ~/code/maixcam-yolo26
scp <gpu-box>:maixcam-yolo26/model/work/320/{yolo26n_320_int8.cvimodel,sim_board.json} model/work/320/ 2>/dev/null || { mkdir -p model/work/320 && scp <gpu-box>:maixcam-yolo26/model/work/320/{yolo26n_320_int8.cvimodel,sim_board.json} model/work/320/; }
cp model/work/320/yolo26n_320_int8.cvimodel assets/
mkdir -p model/board_images && for f in $(cat model/lists/board.txt); do scp -q <gpu-box>:coco/val2017/$f model/board_images/; done
ssh maixcam 'mkdir -p /root/y26/imgs'
scp -q assets/yolo26n_320.mud assets/yolo26n_320_int8.cvimodel model/board_detect.py maixcam:/root/y26/
scp -q model/board_images/*.jpg maixcam:/root/y26/imgs/
ssh maixcam 'cd /root/y26 && python3 board_detect.py yolo26n_320.mud imgs 0.45' > model/work/320/board.json
python3 model/compare_board.py model/work/320/sim_board.json model/work/320/board.json
```
Expected: `board matches simulator` and exit 0, and `board.json` shows sensible COCO classes (e.g. persons in images with people).

**GATE:** If the board fails to load the model (`need at least 6 outputs`, `need 3 bbox and 3 cls outputs`) or detections are garbage/empty while the simulator's are sensible, STOP. Report to the user with the board log. Per the spec, the fallback is approach B (port cat_repel's decoder, extended to 80 classes); that requires a spec update and user approval before continuing. Small IoU mismatches (0.85–0.9) on a few boxes: report the numbers and ask the user whether to relax the tolerance — do not change it unilaterally.

- [ ] **Step 7: Board FPS for MANIFEST**
```bash
ssh maixcam 'cd /root/y26 && python3 -c "
from maix import nn, image, time
d = nn.YOLO26(model=\"yolo26n_320.mud\", dual_buff=False)
img = image.load(\"imgs/$(ls /root/y26/imgs | head -1)\", format=image.Format.FMT_RGB888).resize(552, 368)
for _ in range(5): d.detect(img, conf_th=0.5)
t = time.ticks_ms(); n = 50
for _ in range(n): d.detect(img, conf_th=0.5)
print((time.ticks_ms() - t) / n)"'
```
Record the ms/inference.

- [ ] **Step 8: MANIFEST** — `model/MANIFEST.md`:
```markdown
# Model manifest

Regenerate with `just models` on <gpu-box> (see README). Any change to these
values bumps the app's minor version.

| | 320 | 640 |
|---|---|---|
| Source | yolo26n.pt sha256 `<from sources.toml>` | same |
| Ultralytics | `<exact version>` | same |
| ONNX opset | 17 | 17 |
| Cut outputs | `<six names from output_names.json>` | |
| TPU-MLIR | tpu_mlir[onnx]==1.28.1 on sophgo/tpuc_dev `<image digest: docker image inspect --format '{{.Id}}'>` | |
| Calibration | 500 imgs, `lists/calib.txt` sha256 `<sha256sum model/lists/calib.txt>` | |
| FP32 mAP50 (50 eval imgs, conf 0.45) | `<from sim_eval_map50.txt>` | |
| INT8 mAP50 (sim) | `<from sim_eval_map50.txt>` | |
| Board vs sim (10 imgs) | pass | |
| Board inference ms (552×368 input, FIT_CONTAIN) | `<step 7>` | |
| .cvimodel sha256 | `<sha256sum assets/yolo26n_320_int8.cvimodel>` | |
```
Fill every `<...>` with the measured value now (640 column in Task 7).

- [ ] **Step 9: Commit**
```bash
git add model/convert.sh model/sim_detect.py model/board_detect.py model/compare_board.py model/tests/test_compare_board.py model/MANIFEST.md model/board_images assets/yolo26n_320.mud assets/yolo26n_320_int8.cvimodel
git lfs ls-files   # must list the .cvimodel and the board .jpg files
git commit -m "YOLO26n 320 INT8 model, verified in simulator and on the board" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: 640 model + `just models`

**Files:**
- Create: `assets/yolo26n_640.mud`, `assets/yolo26n_640_int8.cvimodel` (LFS)
- Modify: `justfile` (add `models`), `model/MANIFEST.md` (640 column)

- [ ] **Step 1: Add the recipe to `justfile`**
```make
gpu := env_var_or_default("GPU", "<gpu-box>")

# Rebuild + verify one model size on the GPU box (export, INT8, simulator mAP check)
models size="320":
    rsync -a --exclude .git --exclude model/work ./ {{gpu}}:maixcam-yolo26/
    ssh {{gpu}} 'set -e; cd maixcam-yolo26/model; .venv/bin/pytest -q; \
      .venv/bin/python export.py --size {{size}}; ./convert.sh {{size}}; \
      for L in eval board; do docker run --rm -v "$PWD":/m -v ~/coco:/coco -w /m maixcam-yolo26/tpu_mlir:1.28.1 \
        python3 sim_detect.py --size {{size}} --list lists/$L.txt --coco /coco --out work/{{size}}/sim_$L.json; done'
    @echo "now run the mAP check and board compare for {{size}} (README: Rebuilding models)"
```

- [ ] **Step 2: Build 640** — `just models 640`, then run the Task 6 Step 2 mAP check with `320` replaced by `640`.
Expected: drop ≤ 3 pts.

- [ ] **Step 3: `.mud`** — `assets/yolo26n_640.mud`: identical to `yolo26n_320.mud` except `model = yolo26n_640_int8.cvimodel`.

- [ ] **Step 4: Board check** — repeat Task 6 Steps 6–7 with `640`. Expected: `board matches simulator`. If the board fails to load 640 (out of memory), record that in MANIFEST, tell the user, and per spec ship 320-only (the app's fallback already handles a missing 640 file). Do not proceed to delete code paths without the user's decision.

- [ ] **Step 5: Fill the 640 column in MANIFEST, commit**
```bash
git add justfile assets/yolo26n_640.mud assets/yolo26n_640_int8.cvimodel model/MANIFEST.md
git commit -m "YOLO26n 640 INT8 model and just models" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: MaixCDK app — build, detection loop, overlay, deploy

**Files:**
- Create: `app.yaml`, `main/CMakeLists.txt`, `main/Kconfig`, `main/src/main.cpp`, `scripts/build.sh`, `assets/app.png` (LFS)
- Modify: `justfile` (add `build`, `release-local`, `deploy`)

**Interfaces:**
- Consumes: `y26::FpsMeter`, `y26::format_status`, `y26::make_layout`, `y26::Layout`, `y26::Rect`, `y26::tr`, `y26::Str`, `y26::lang_from_locale`, `y26::ModelSize`, `y26::pixels`.

- [ ] **Step 1: `scripts/build.sh`** (clean SDK clone at the pin; never touches `~/code/MaixCDK-v4125`, which cat_repel patches)
```bash
#!/usr/bin/env bash
# Build or release with MaixCDK v4.12.5 in sipeed/maixcdk-builder.
# Usage: scripts/build.sh build|release
set -euo pipefail
MODE=${1:-build}
PIN=db6b17821b900c97eaa899be548ddc1cba0dd635
REPO="$(cd "$(dirname "$0")/.." && pwd)"
SDK=${MAIXCDK_CACHE:-$HOME/.cache/maixcam-yolo26/MaixCDK}
if [ ! -d "$SDK/.git" ]; then
    git clone --filter=blob:none https://github.com/sipeed/MaixCDK "$SDK"
fi
if [ "$(git -C "$SDK" rev-parse HEAD)" != "$PIN" ]; then
    git -C "$SDK" fetch origin "$PIN" && git -C "$SDK" checkout --detach "$PIN"
fi
[ -z "$(git -C "$SDK" status --porcelain)" ] || { echo "SDK cache $SDK is dirty; refusing" >&2; exit 1; }
case "$MODE" in
    build) CMD="maixcdk build -p maixcam" ;;
    release) CMD="maixcdk release -P maixcam" ;;
    *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
docker run --rm --network=host -e MAIXCDK_PATH=/opt/MaixCDK \
    -v "$SDK":/opt/MaixCDK -v "$REPO":/work/app \
    --user "$(id -u):$(id -g)" -e HOME=/tmp \
    --entrypoint bash sipeed/maixcdk-builder -c "cd /work/app && $CMD"
```
If the container fails as a non-root user (toolchain download into a root-owned path), drop the `--user`/`HOME` lines and add `sudo chown -R "$(id -u):$(id -g)" build dist` after the run — tell the user rather than running sudo yourself.

- [ ] **Step 2: `app.yaml`**
```yaml
id: yolo26_detect
name: YOLO26 Detector
name[zh]: YOLO26 物体检测
version: 1.0.0
icon: assets/app.png
author: Stephen Cox
desc: Real-time COCO object detection with YOLO26n (320 or 640), touch controls and snapshots
desc[zh]: YOLO26n 实时物体检测（320 或 640），支持触屏调节与拍照
files:
  app.yaml: app.yaml
  assets: assets
  README.md: README.md
```

- [ ] **Step 3: `main/CMakeLists.txt` and `main/Kconfig`**
```cmake
list(APPEND ADD_INCLUDE "include")
append_srcs_dir(ADD_SRCS "src")
list(APPEND ADD_REQUIREMENTS basic vision nn)
register_component()
```
`main/Kconfig`: copy `~/code/MaixCDK-v4125/examples/nn_yolo26_detect/main/Kconfig` verbatim.

- [ ] **Step 4: Icon** — create a 128×128 PNG `assets/app.png` (a simple bounding-box glyph with "26"):
```bash
python3 - <<'EOF'
from PIL import Image, ImageDraw, ImageFont
im = Image.new("RGBA", (128, 128), (20, 24, 32, 255))
d = ImageDraw.Draw(im)
d.rounded_rectangle((18, 22, 110, 106), radius=8, outline=(255, 70, 70, 255), width=6)
d.text((40, 44), "26", fill=(255, 255, 255, 255), font=ImageFont.load_default(size=40))
im.save("assets/app.png")
EOF
```

- [ ] **Step 5: `main/src/main.cpp` — detection loop with overlay (controls drawn, only Exit wired here)**
```cpp
// YOLO26 detector app. The only file that touches MaixCDK; all decisions
// live in main/include/*.hpp and are host-tested.
#include "maix_basic.hpp"
#include "maix_camera.hpp"
#include "maix_display.hpp"
#include "maix_image.hpp"
#include "maix_nn_yolo26.hpp"
#include "maix_touchscreen.hpp"

#include "controls.hpp"
#include "fps_meter.hpp"
#include "strings.hpp"

#include <memory>

using namespace maix;

static const char *kFont = "/maixapp/share/font/SourceHanSansCN-Regular.otf";

static std::string asset(const std::string &name)
{
    std::string base = app::get_app_path();
    return (base.empty() ? std::string(".") : base) + "/assets/" + name;
}

static std::string mud_for(y26::ModelSize m) { return asset("yolo26n_" + y26::to_string(m) + ".mud"); }

static void draw_button(image::Image &img, const y26::Rect &r, const std::string &text)
{
    img.draw_rect(r.x, r.y, r.w, r.h, image::COLOR_WHITE, 2);
    image::Size ts = image::string_size(text);
    img.draw_string(r.x + (r.w - ts.width()) / 2, r.y + (r.h - ts.height()) / 2, text, image::COLOR_WHITE);
}

int main()
{
    y26::Lang lang = y26::lang_from_locale(i18n::get_locale());
    if (image::load_font("ui", kFont, 22) == err::ERR_NONE)
        image::set_default_font("ui");

    display::Display disp;
    touchscreen::TouchScreen ts;
    const int sw = disp.width(), sh = disp.height();
    const y26::Layout layout = y26::make_layout(sw, sh);

    y26::ModelSize size = y26::ModelSize::S320;
    std::unique_ptr<nn::YOLO26> det;
    try
    {
        det = std::make_unique<nn::YOLO26>(mud_for(size), false);
    }
    catch (const std::exception &e)
    {
        log::error("model load failed: %s", e.what());
    }

    camera::Camera cam(sw, sh, image::FMT_RGB888);
    y26::FpsMeter meter;
    uint64_t last_us = time::ticks_us();

    while (!app::need_exit())
    {
        int tx = 0, ty = 0;
        bool pressed = false;
        ts.read(tx, ty, pressed);
        if (pressed && y26::hit_test(layout, tx, ty) == y26::Button::Exit)
            app::set_exit_flag(true);

        image::Image *raw = cam.read();
        if (!raw)
            continue;
        std::unique_ptr<image::Image> img(raw);

        double infer_ms = 0;
        if (det)
        {
            uint64_t t0 = time::ticks_us();
            std::unique_ptr<std::vector<nn::Object>> objs(det->detect(*img, 0.5f, 0.45f));
            infer_ms = (time::ticks_us() - t0) / 1000.0;
            for (const auto &o : *objs)
            {
                img->draw_rect(o.x, o.y, o.w, o.h, image::COLOR_RED, 3);
                char msg[64];
                snprintf(msg, sizeof msg, "%s %.2f", det->labels[o.class_id].c_str(), o.score);
                img->draw_string(o.x, std::max(0, o.y - 24), msg, image::COLOR_RED);
            }
        }
        else
        {
            img->draw_string(16, sh / 2, std::string(y26::tr(y26::Str::NoModel, lang)) + " " + mud_for(size), image::COLOR_RED);
        }

        draw_button(*img, layout.exit, y26::tr(y26::Str::Exit, lang));
        img->draw_string(layout.status_x, layout.status_y,
                         y26::format_status(meter.fps(), meter.latency_ms(), y26::pixels(size)), image::COLOR_GREEN);
        disp.show(*img);

        uint64_t now = time::ticks_us();
        meter.add((now - last_us) / 1000.0, infer_ms);
        last_us = now;
    }
    return 0;
}
```

- [ ] **Step 6: justfile recipes**
```make
board := env_var_or_default("BOARD", "maixcam")

build:
    scripts/build.sh build

release-local:
    scripts/build.sh release
    ls -l dist/*.zip

# Package, copy to the board, install via the app store CLI
deploy: release-local
    scp dist/yolo26_detect_v*.zip {{board}}:/tmp/yolo26_detect.zip
    ssh {{board}} 'app_store install /tmp/yolo26_detect.zip && rm /tmp/yolo26_detect.zip'
```

- [ ] **Step 7: Build** — `chmod +x scripts/build.sh && just build`
Expected: completes and produces `build/yolo26_detect` (check `file build/yolo26_detect` → `ELF 64-bit LSB ... RISC-V`). Compiler errors in `main.cpp` are fixed against the real headers in `$HOME/.cache/maixcam-yolo26/MaixCDK/components/` — e.g. confirm `image::load_font`, `image::set_default_font`, `time::ticks_us`, and `touchscreen::TouchScreen::read(int&, int&, bool&)` signatures there.

- [ ] **Step 8: Deploy and look** — `just deploy`, then launch "YOLO26 Detector" from the board's launcher.
Expected: live view with red boxes on people/objects, green status line `FPS xx.x | yy ms | 320` (first ~1 frame shows `-`), `< Exit` button top-left that returns to the launcher. If `app_store install` is not found on the board, run `ssh maixcam 'ls /maixapp/apps'` and report — do not install by copying files into `/maixapp/apps` without telling the user.

- [ ] **Step 9: Commit**
```bash
git add app.yaml main scripts/build.sh assets/app.png justfile
git commit -m "MaixCDK app: detection loop, overlay, status line, deploy" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Full controls — threshold, model switch + fallback, persistence, snapshot, errors

**Files:**
- Modify: `main/src/main.cpp`
- Create: `README.md` (usage + manual checklist)

**Interfaces:**
- Consumes: everything from Tasks 1–4: `PressEdge`, `Threshold`, `ModelSize`/`other`/`parse_model_size`/`to_string`, `hit_test`/`Button`, `Str::*`, `next_snapshot_index`, `snapshot_name`.

- [ ] **Step 1: Replace `main.cpp` with the full version**
```cpp
// YOLO26 detector app. The only file that touches MaixCDK; all decisions
// live in main/include/*.hpp and are host-tested.
#include "maix_basic.hpp"
#include "maix_camera.hpp"
#include "maix_display.hpp"
#include "maix_image.hpp"
#include "maix_nn_yolo26.hpp"
#include "maix_touchscreen.hpp"

#include "controls.hpp"
#include "fps_meter.hpp"
#include "snapshot.hpp"
#include "strings.hpp"

#include <dirent.h>
#include <sys/stat.h>

#include <memory>

using namespace maix;

static const char *kFont = "/maixapp/share/font/SourceHanSansCN-Regular.otf";
static const char *kSnapDir = "/root/snapshots";
static const char *kCfg = "yolo26_detect";

struct Toast
{
    std::string text;
    image::Color color = image::COLOR_WHITE;
    uint64_t until_ms = 0;
    void show(const std::string &t, image::Color c, uint64_t dur_ms)
    {
        text = t;
        color = c;
        until_ms = time::ticks_ms() + dur_ms;
    }
};

static std::string asset(const std::string &name)
{
    std::string base = app::get_app_path();
    return (base.empty() ? std::string(".") : base) + "/assets/" + name;
}

static std::string mud_for(y26::ModelSize m) { return asset("yolo26n_" + y26::to_string(m) + ".mud"); }

static std::unique_ptr<nn::YOLO26> try_load(y26::ModelSize m)
{
    try
    {
        return std::make_unique<nn::YOLO26>(mud_for(m), false);
    }
    catch (const std::exception &e)
    {
        log::error("load %s failed: %s", mud_for(m).c_str(), e.what());
        return nullptr;
    }
}

static void draw_button(image::Image &img, const y26::Rect &r, const std::string &text)
{
    img.draw_rect(r.x, r.y, r.w, r.h, image::COLOR_WHITE, 2);
    image::Size ts = image::string_size(text);
    img.draw_string(r.x + (r.w - ts.width()) / 2, r.y + (r.h - ts.height()) / 2, text, image::COLOR_WHITE);
}

static void show_message(display::Display &disp, int sw, int sh, const std::string &text, image::Color c)
{
    image::Image screen(sw, sh, image::FMT_RGB888);
    screen.clear();
    screen.draw_string(16, sh / 2 - 12, text, c);
    disp.show(screen);
}

static bool save_snapshot(image::Image &img)
{
    mkdir(kSnapDir, 0755);
    std::vector<std::string> names;
    if (DIR *d = opendir(kSnapDir))
    {
        while (dirent *e = readdir(d))
            names.emplace_back(e->d_name);
        closedir(d);
    }
    std::string path = std::string(kSnapDir) + "/" + y26::snapshot_name(y26::next_snapshot_index(names));
    return img.save(path.c_str()) == err::ERR_NONE;
}

int main()
{
    y26::Lang lang = y26::lang_from_locale(i18n::get_locale());
    if (image::load_font("ui", kFont, 22) == err::ERR_NONE)
        image::set_default_font("ui");
    else
        log::warn("font %s not loaded; Chinese text may not render", kFont);

    display::Display disp;
    touchscreen::TouchScreen ts;
    const int sw = disp.width(), sh = disp.height();
    const y26::Layout layout = y26::make_layout(sw, sh);

    y26::ModelSize size = y26::parse_model_size(app::get_app_config_kv(kCfg, "model", "320")).value_or(y26::ModelSize::S320);
    y26::Threshold conf = y26::Threshold::from_pct_string(app::get_app_config_kv(kCfg, "conf", "50"));

    show_message(disp, sw, sh, y26::tr(y26::Str::Loading, lang), image::COLOR_WHITE);
    std::unique_ptr<nn::YOLO26> det = try_load(size);
    Toast toast;
    if (!det && size == y26::ModelSize::S640)
    {
        size = y26::ModelSize::S320;
        det = try_load(size);
        toast.show(y26::tr(y26::Str::Fallback320, lang), image::COLOR_YELLOW, 2000);
    }

    camera::Camera cam(sw, sh, image::FMT_RGB888);
    y26::FpsMeter meter;
    y26::PressEdge edge;
    uint64_t last_us = time::ticks_us();

    while (!app::need_exit())
    {
        try
        {
            int tx = 0, ty = 0;
            bool pressed = false;
            ts.read(tx, ty, pressed);
            y26::Button btn = edge.update(pressed) ? y26::hit_test(layout, tx, ty) : y26::Button::None;
            bool want_snapshot = false;

            switch (btn)
            {
            case y26::Button::Exit:
                app::set_exit_flag(true);
                continue;
            case y26::Button::ConfDown:
                conf.down();
                app::set_app_config_kv(kCfg, "conf", conf.to_pct_string());
                break;
            case y26::Button::ConfUp:
                conf.up();
                app::set_app_config_kv(kCfg, "conf", conf.to_pct_string());
                break;
            case y26::Button::Model:
            {
                y26::ModelSize want = y26::other(size);
                show_message(disp, sw, sh, y26::tr(y26::Str::Loading, lang), image::COLOR_WHITE);
                det.reset();                                  // free NPU memory before loading the other
                det = try_load(want);
                if (det)
                    size = want;
                else
                {
                    toast.show(y26::tr(y26::Str::Fallback320, lang), image::COLOR_YELLOW, 2000);
                    size = y26::ModelSize::S320;
                    det = try_load(size);
                }
                app::set_app_config_kv(kCfg, "model", y26::to_string(size));
                last_us = time::ticks_us();                   // don't count the load in FPS
                break;
            }
            case y26::Button::Snapshot:
                want_snapshot = true;
                break;
            case y26::Button::None:
                break;
            }

            image::Image *raw = cam.read();
            if (!raw)
                continue;
            std::unique_ptr<image::Image> img(raw);

            double infer_ms = 0;
            if (det)
            {
                uint64_t t0 = time::ticks_us();
                std::unique_ptr<std::vector<nn::Object>> objs(det->detect(*img, conf.value(), 0.45f));
                infer_ms = (time::ticks_us() - t0) / 1000.0;
                for (const auto &o : *objs)
                {
                    img->draw_rect(o.x, o.y, o.w, o.h, image::COLOR_RED, 3);
                    char msg[64];
                    snprintf(msg, sizeof msg, "%s %.2f", det->labels[o.class_id].c_str(), o.score);
                    img->draw_string(o.x, std::max(0, o.y - 24), msg, image::COLOR_RED);
                }
            }
            else
            {
                img->draw_string(16, sh / 2, std::string(y26::tr(y26::Str::NoModel, lang)) + " " + mud_for(size),
                                 image::COLOR_RED);
            }

            // Snapshot = detections only, before the UI chrome is drawn.
            if (want_snapshot)
            {
                bool ok = save_snapshot(*img);
                toast.show(y26::tr(ok ? y26::Str::Saved : y26::Str::SaveFailed, lang),
                           ok ? image::COLOR_GREEN : image::COLOR_RED, 1500);
            }

            char confs[32];
            snprintf(confs, sizeof confs, "%.2f", conf.value());
            draw_button(*img, layout.exit, y26::tr(y26::Str::Exit, lang));
            draw_button(*img, layout.model, y26::to_string(size));
            draw_button(*img, layout.conf_down, "-");
            draw_button(*img, layout.conf_up, "+");
            img->draw_string(layout.conf_value.x + 6, layout.conf_value.y + 4, y26::tr(y26::Str::Conf, lang), image::COLOR_WHITE);
            img->draw_string(layout.conf_value.x + 6, layout.conf_value.y + layout.conf_value.h / 2, confs, image::COLOR_WHITE);
            draw_button(*img, layout.snapshot, y26::tr(y26::Str::Snapshot, lang));
            img->draw_string(layout.status_x, layout.status_y,
                             y26::format_status(meter.fps(), meter.latency_ms(), y26::pixels(size)), image::COLOR_GREEN);
            if (time::ticks_ms() < toast.until_ms)
                img->draw_string(layout.status_x, layout.status_y + 32, toast.text, toast.color);

            disp.show(*img);

            uint64_t now = time::ticks_us();
            meter.add((now - last_us) / 1000.0, infer_ms);
            last_us = now;
        }
        catch (const std::exception &e)
        {
            log::error("frame skipped: %s", e.what());
        }
    }
    return 0;
}
```
Decision to note in the README: the snapshot is the frame **with detections but without the buttons/status overlay** — the spec says "annotated frame", which this satisfies; the UI chrome is not part of the picture.

- [ ] **Step 2: Build** — `just build` — Expected: success. Fix signature mismatches against the SDK headers (see Task 8 Step 7). `image::COLOR_YELLOW` — if it doesn't exist, use `image::Color::from_rgb(255, 200, 0)`.

- [ ] **Step 3: Host tests still pass** — `just test` — Expected: all `0 failures`.

- [ ] **Step 4: `README.md`** with these sections (fill with real content, matching this plan):
  - What it is (one paragraph; links to MaixHub listing once published and to app 123 as predecessor).
  - Controls table (from spec §4.2).
  - Building: `just test`, `just build`, `just deploy` (needs Docker, Git LFS, ssh alias `maixcam`).
  - Rebuilding models: `just models 320|640`, then the mAP check and board compare commands from Task 6 Steps 2 and 6; update `model/MANIFEST.md`.
  - **Manual test checklist** (run after `just deploy`, tick each):
    1. Launch from launcher; status line shows `-` then numbers within 2 s.
    2. Point at a person: red box labelled `person`.
    3. `+` to 0.95: fewer boxes; `-` to 0.45 and it stops there.
    4. Exit and relaunch: threshold and model size are remembered.
    5. Tap `320` → "Loading..." → `640`; FPS drops, status shows `640`. Hold the finger on the button through the load: it switches once only.
    6. Rename `/maixapp/apps/yolo26_detect/assets/yolo26n_640_int8.cvimodel`, tap `320`: yellow fallback message, stays on 320. Restore the file.
    7. Snapshot twice: "Saved" twice; `ssh maixcam ls /root/snapshots` shows `yolo26_1.jpg` and `yolo26_2.jpg`; open one — boxes present, no buttons.
    8. Settings → Language → 中文, relaunch: Chinese labels render (no boxes/tofu).
    9. Exit returns to the launcher.
  - Release checklist (Task 11).

- [ ] **Step 5: Deploy and run the manual checklist** — `just deploy`, then go through items 1–9. Report each item's result to the user; any failure is fixed before committing.

- [ ] **Step 6: Commit**
```bash
git add main/src/main.cpp README.md
git commit -m "Touch controls, model switch with fallback, persistence, snapshots, i18n" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: CI workflow

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Write the workflow**
```yaml
name: ci
on:
  push:
  pull_request:
jobs:
  host-tests:
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v4
      - uses: extractions/setup-just@v2
      - run: just test
  model-tests:
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: cd model && uv venv --python 3.12 && uv pip install pytest numpy opencv-python-headless pycocotools && .venv/bin/pytest -q tests/test_select_images.py tests/test_decode.py tests/test_eval_map.py tests/test_compare_board.py
  maixcam-build:
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v4
        with:
          lfs: true
      - uses: actions/cache@v4
        with:
          path: ~/.cache/maixcam-yolo26
          key: maixcdk-db6b17821b900c97eaa899be548ddc1cba0dd635
      - run: scripts/build.sh build
      - run: file build/yolo26_detect | grep -q RISC-V
```

- [ ] **Step 2: Lint locally** — `docker run --rm -v "$PWD":/repo -w /repo rhysd/actionlint:latest` — Expected: no output.

- [ ] **Step 3: Commit**
```bash
git add .github/workflows/ci.yml
git commit -m "CI: host tests, model tests, MaixCAM build" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
(The workflow first runs when the repo is pushed in Task 12.)

---

### Task 11: Release workflow, version gate, CHANGELOG

**Files:**
- Create: `scripts/check_release.sh`, `test/test_check_release.sh`, `.github/workflows/release.yml`, `CHANGELOG.md`
- Modify: `justfile` (add `test-scripts` to `test`), `README.md` (release checklist)

**Interfaces:**
- Produces: `scripts/check_release.sh <tag>` — exit 0 iff `<tag>` == `v` + `app.yaml` version == top `## [X.Y.Z]` in `CHANGELOG.md`; prints the mismatch otherwise. Reads files from the current directory.

- [ ] **Step 1: Write the failing test** — `test/test_check_release.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
S="$(cd "$(dirname "$0")/.." && pwd)/scripts/check_release.sh"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
cd "$T"
printf 'id: yolo26_detect\nversion: 1.2.0\n' > app.yaml
printf '# Changelog\n\n## [1.2.0] - 2026-10-08\n- x\n\n## [1.1.0] - 2026-10-01\n' > CHANGELOG.md
"$S" v1.2.0 >/dev/null || { echo "FAIL: matching tag rejected"; exit 1; }
! "$S" v1.1.0 >/dev/null 2>&1 || { echo "FAIL: wrong tag accepted"; exit 1; }
! "$S" 1.2.0 >/dev/null 2>&1 || { echo "FAIL: tag without v accepted"; exit 1; }
printf '# Changelog\n\n## [1.1.0] - 2026-10-01\n' > CHANGELOG.md
! "$S" v1.2.0 >/dev/null 2>&1 || { echo "FAIL: stale changelog accepted"; exit 1; }
echo "check_release: ok"
```
Add to the `test` recipe in `justfile` as a final line: `    bash test/test_check_release.sh`
Run: `just test` — Expected: fails with `No such file or directory` for `check_release.sh`.

- [ ] **Step 2: Implement** — `scripts/check_release.sh`:
```bash
#!/usr/bin/env bash
# Release gate: tag, app.yaml version and top CHANGELOG entry must agree.
set -euo pipefail
TAG=${1:?usage: check_release.sh vX.Y.Z}
[[ "$TAG" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "tag '$TAG' is not vX.Y.Z" >&2; exit 1; }
V=${TAG#v}
APP=$(sed -n 's/^version:[[:space:]]*//p' app.yaml | head -1)
LOG=$(sed -n 's/^## \[\([0-9.]*\)\].*/\1/p' CHANGELOG.md | head -1)
[ "$APP" = "$V" ] || { echo "app.yaml version $APP != tag $V" >&2; exit 1; }
[ "$LOG" = "$V" ] || { echo "CHANGELOG top entry $LOG != tag $V" >&2; exit 1; }
echo "release $V: tag, app.yaml and CHANGELOG agree"
```
Run: `chmod +x scripts/check_release.sh && just test` — Expected: `check_release: ok`.

- [ ] **Step 3: `CHANGELOG.md`**
```markdown
# Changelog

## [1.0.0] - YYYY-MM-DD
- First release: YOLO26n COCO detection at 320 and 640 (switchable), confidence control,
  snapshots, English/Chinese UI. Models built and verified by `model/` (see MANIFEST).
- Successor to MaixHub app 123 (YOLOv11n 320, MaixPy).
```
Set the date to the actual release day in Task 12.

- [ ] **Step 4: `.github/workflows/release.yml`**
```yaml
name: release
on:
  push:
    tags: ["v*.*.*"]
permissions:
  contents: write
jobs:
  release:
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v4
        with:
          lfs: true
      - uses: extractions/setup-just@v2
      - run: scripts/check_release.sh "${GITHUB_REF_NAME}"
      - run: just test
      - uses: actions/cache@v4
        with:
          path: ~/.cache/maixcam-yolo26
          key: maixcdk-db6b17821b900c97eaa899be548ddc1cba0dd635
      - run: scripts/build.sh release
      - name: Collect package
        run: |
          shopt -s nullglob
          zips=(dist/*.zip)
          [ ${#zips[@]} -eq 1 ] || { echo "expected 1 zip, got: ${zips[*]}"; exit 1; }
          out="yolo26_detect_${GITHUB_REF_NAME}.zip"
          cp "${zips[0]}" "$out"
          unzip -l "$out" | grep -q 'assets/yolo26n_320_int8.cvimodel'
          unzip -l "$out" | grep -q 'yolo26_detect$'
          sha256sum "$out" > "$out.sha256"
      - run: gh release create "${GITHUB_REF_NAME}" yolo26_detect_${GITHUB_REF_NAME}.zip yolo26_detect_${GITHUB_REF_NAME}.zip.sha256 --title "${GITHUB_REF_NAME}" --notes "See CHANGELOG.md"
        env:
          GH_TOKEN: ${{ github.token }}
```
The `unzip -l | grep` checks guard against an LFS pointer being packaged instead of the real model: also add `- run: test $(stat -c%s assets/yolo26n_320_int8.cvimodel) -gt 100000` right after checkout.

- [ ] **Step 5: Local dry run of packaging** — `just release-local && unzip -l dist/*.zip` — Expected: lists `app.yaml`, `assets/…cvimodel` (MB-sized, not ~130 bytes), `assets/*.mud`, `assets/app.png`, the `yolo26_detect` binary.

- [ ] **Step 6: README release checklist** — append:
  1. `just test`; `just deploy`; full manual checklist passes; board vs sim compare passes for both sizes.
  2. Bump `app.yaml` `version`, add the `CHANGELOG.md` entry with today's date, commit.
  3. `git tag vX.Y.Z && git push origin main vX.Y.Z`; wait for the `release` workflow to go green.
  4. Download `yolo26_detect_vX.Y.Z.zip` from the GitHub Release; verify its sha256.
  5. Upload it at maixhub.com/app (new version of `yolo26_detect`), paste the CHANGELOG entry as notes.
  6. Install from MaixHub on the board via the app store and run checklist items 1, 2, 9.

- [ ] **Step 7: Commit**
```bash
git add scripts/check_release.sh test/test_check_release.sh .github/workflows/release.yml CHANGELOG.md justfile README.md
git commit -m "Release workflow with tag/app.yaml/CHANGELOG gate" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 12: Publish (outward-facing — confirm each step with the user first)

- [ ] **Step 1: Ask the user** to confirm creating the public repo `stephencox/maixcam-yolo26` and pushing. Then:
```bash
gh repo create stephencox/maixcam-yolo26 --public --description "MaixCAM app: YOLO26n object detection (MaixCDK, 320/640)" --source . --push
```
Expected: CI runs; `gh run watch` shows `host-tests`, `model-tests`, `maixcam-build` green. Fix any red job before continuing.

- [ ] **Step 2: Ask the user** to confirm tagging 1.0.0. Set the CHANGELOG date, commit, then `git tag v1.0.0 && git push origin main v1.0.0`; `gh run watch` the `release` workflow; confirm the Release has the zip + sha256.

- [ ] **Step 3: User uploads to MaixHub** — give the user the zip path/URL and the CHANGELOG text. MaixHub has no upload API; the user does this in the browser. Ask the user for the new app's URL.

- [ ] **Step 4: Post-publish** — add the MaixHub URL to `README.md`; commit and push. Remind the user to add "Superseded by <url>" to app 123's description (manual, on MaixHub) and optionally to the old repo's README (ask before editing that repo).
