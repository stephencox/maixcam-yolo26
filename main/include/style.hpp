// UI style ("style C"): colours, per-class palette, label text. Pure C++,
// host-tested; main.cpp converts Rgb to the SDK colour type.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "controls.hpp"

namespace y26
{
    struct Rgb
    {
        uint8_t r, g, b;
    };

    inline constexpr Rgb kAccent{255, 90, 78};   // #FF5A4E coral
    inline constexpr Rgb kPanel{29, 33, 39};     // #1D2127
    inline constexpr Rgb kText{255, 255, 255};
    inline constexpr Rgb kDarkText{16, 20, 24};  // #101418

    inline constexpr std::array<Rgb, 8> kClassPalette{{
        kAccent,
        {46, 196, 182},    // teal   #2EC4B6
        {255, 193, 69},    // amber  #FFC145
        {155, 93, 229},    // violet #9B5DE5
        {76, 201, 240},    // sky    #4CC9F0
        {167, 201, 87},    // lime   #A7C957
        {241, 91, 181},    // pink   #F15BB5
        {255, 140, 66},    // orange #FF8C42
    }};

    inline Rgb class_color(int class_id)
    {
        if (class_id < 0) return kClassPalette[0];
        return kClassPalette[static_cast<std::size_t>(class_id) % kClassPalette.size()];
    }

    // Relative luminance (sRGB, Rec.709 weights); dark text on light fills.
    inline Rgb label_text_color(Rgb bg)
    {
        auto lin = [](uint8_t v) {
            double c = v / 255.0;
            return c <= 0.04045 ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4);
        };
        double y = 0.2126 * lin(bg.r) + 0.7152 * lin(bg.g) + 0.0722 * lin(bg.b);
        return y > 0.5 ? kDarkText : kText;
    }

    inline std::string format_label(const std::string &name, float score)
    {
        long p = std::lround(static_cast<double>(score) * 100.0);
        if (p < 0) p = 0;
        if (p > 100) p = 100;
        char buf[16];
        std::snprintf(buf, sizeof buf, " %ld%%", p);
        return name + buf;
    }

    // Label tab placement: above the box by default; inside the box top when
    // that would leave the screen or overlap an already-placed tab (an inside
    // tab is kept even if it collides, so a label is never hidden). Clamped to
    // the screen width. scripts/render_art.py mirrors this rule for the cover.
    inline Rect place_label_tab(const Rect &box, int tab_w, int tab_h, const std::vector<Rect> &placed,
                                int screen_w, int /*screen_h*/)
    {
        int x = std::max(0, std::min(box.x, screen_w - tab_w));
        Rect above{x, box.y - tab_h, tab_w, tab_h};
        bool hit = above.y < 0;
        for (const Rect &p : placed)
            if (above.x < p.x + p.w && p.x < above.x + above.w && above.y < p.y + p.h && p.y < above.y + above.h)
                hit = true;
        return hit ? Rect{x, box.y, tab_w, tab_h} : above;
    }
}
