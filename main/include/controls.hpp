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
        Rect status;   // pill: top row between Exit and Model, or a second row when too narrow
    };

    // Top row: Exit (left), Model (right). Bottom row: conf -, value, conf + (left),
    // Snapshot (right). Status pill centred in the gap between Exit and Model
    // (button height); if that pill would be under 100 px wide it moves to a
    // 32 px second row under the top row. Buttons never drop below 48 px tall.
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
        const int gap_l = l.exit.x + l.exit.w, gap_r = l.model.x;
        const int pill_w = gap_r - gap_l - 2 * m;
        if (pill_w >= 100)
            l.status = {gap_l + m, top, pill_w, bh};
        else
            l.status = {m, top + bh + m, std::min(sw - 2 * m, 160), 32};
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
