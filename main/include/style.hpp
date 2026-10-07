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
    inline constexpr Rgb kMuted{170, 176, 184};  // secondary text

    inline constexpr std::array<Rgb, 8> kClassPalette{{
        {229, 57, 44},     // coral  #E5392C
        {22, 137, 127},    // teal   #16897F
        {183, 121, 31},    // amber  #B7791F
        {123, 63, 196},    // violet #7B3FC4
        {31, 127, 174},    // sky    #1F7FAE
        {94, 127, 37},     // lime   #5E7F25
        {194, 51, 127},    // pink   #C2337F
        {196, 90, 18},     // orange #C45A12
    }};

    inline Rgb class_color(int class_id)
    {
        if (class_id < 0) return kClassPalette[0];
        return kClassPalette[static_cast<std::size_t>(class_id) % kClassPalette.size()];
    }

    // Label text is always kText (white); every palette colour keeps >= 3:1
    // contrast against it (bold labels count as large text), checked in tests.

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
                                int screen_w)
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
