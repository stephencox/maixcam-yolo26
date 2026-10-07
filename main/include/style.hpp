// UI style ("style C"): colours, per-class palette, label text. Pure C++,
// host-tested; main.cpp converts Rgb to the SDK colour type.
#pragma once
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>

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
}
