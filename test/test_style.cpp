#include "check.hpp"
#include "style.hpp"
#include <algorithm>
#include <cmath>

using namespace y26;

static bool same(Rgb a, Rgb b) { return a.r == b.r && a.g == b.g && a.b == b.b; }

TEST(constants)
{
    CHECK(same(kAccent, Rgb{255, 90, 78}));
    CHECK(same(kPanel, Rgb{29, 33, 39}));
    CHECK(same(kText, Rgb{255, 255, 255}));
}

TEST(palette_entry_zero_is_coral)   // person; deeper than the UI accent
{
    CHECK(same(class_color(0), Rgb{229, 57, 44}));
    CHECK(same(kClassPalette[0], Rgb{229, 57, 44}));
}

TEST(class_color_wraps)
{
    for (int i = 0; i < 8; ++i)
    {
        CHECK(same(class_color(i), kClassPalette[i]));
        CHECK(same(class_color(i + 8), kClassPalette[i]));
        CHECK(same(class_color(i + 72), kClassPalette[i]));
    }
}

TEST(class_color_negative_is_entry_zero)
{
    CHECK(same(class_color(-1), kClassPalette[0]));
    CHECK(same(class_color(-100), kClassPalette[0]));
}

TEST(palette_entries_distinct)
{
    for (int i = 0; i < 8; ++i)
        for (int j = i + 1; j < 8; ++j) CHECK(!same(kClassPalette[i], kClassPalette[j]));
}

static double lin(uint8_t v)
{
    double c = v / 255.0;
    return c <= 0.04045 ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4);
}

static double contrast_vs_white(Rgb c)
{
    double y = 0.2126 * lin(c.r) + 0.7152 * lin(c.g) + 0.0722 * lin(c.b);
    return 1.05 / (y + 0.05);
}

TEST(palette_white_text_contrast_at_least_3)   // bold labels = large text
{
    double lowest = 100;
    for (const Rgb &c : kClassPalette)
    {
        CHECK(contrast_vs_white(c) >= 3.0);
        lowest = std::min(lowest, contrast_vs_white(c));
    }
    CHECK(contrast_vs_white(kClassPalette[2]) == lowest);   // amber is the weakest
    CHECK_NEAR(lowest, 3.6, 0.2);
}

TEST(muted_and_accent_unchanged)
{
    CHECK(same(kMuted, Rgb{170, 176, 184}));
    CHECK(same(kAccent, Rgb{255, 90, 78}));
}

TEST(format_label_rounds_and_clamps)
{
    CHECK_EQ(format_label("person", 0.91f), std::string("person 91%"));
    CHECK_EQ(format_label("cat", 0.906f), std::string("cat 91%"));
    CHECK_EQ(format_label("cat", 0.904f), std::string("cat 90%"));
    CHECK_EQ(format_label("dog", 1.7f), std::string("dog 100%"));
    CHECK_EQ(format_label("dog", -0.3f), std::string("dog 0%"));
    CHECK_EQ(format_label("traffic light", 0.45f), std::string("traffic light 45%"));
}

static bool eq(Rect a, Rect b) { return a.x == b.x && a.y == b.y && a.w == b.w && a.h == b.h; }

TEST(tab_above_by_default)
{
    CHECK(eq(place_label_tab({100, 100, 50, 50}, 60, 24, {}, 552), Rect{100, 76, 60, 24}));
}

TEST(tab_inside_when_above_is_off_screen)
{
    CHECK(eq(place_label_tab({100, 10, 50, 50}, 60, 24, {}, 552), Rect{100, 10, 60, 24}));
}

TEST(tab_inside_on_collision)
{
    std::vector<Rect> placed{{110, 70, 60, 24}};
    CHECK(eq(place_label_tab({100, 100, 50, 50}, 60, 24, placed, 552), Rect{100, 100, 60, 24}));
}

TEST(tab_inside_kept_even_if_colliding)
{
    std::vector<Rect> placed{{110, 70, 60, 24}, {100, 100, 60, 24}};
    CHECK(eq(place_label_tab({100, 100, 50, 50}, 60, 24, placed, 552), Rect{100, 100, 60, 24}));
}

TEST(tab_touching_is_not_collision)
{
    std::vector<Rect> placed{{100, 52, 60, 24}};   // ends at y=76, tab starts at 76
    CHECK(eq(place_label_tab({100, 100, 50, 50}, 60, 24, placed, 552), Rect{100, 76, 60, 24}));
}

TEST(tab_right_edge_clamp)
{
    CHECK(eq(place_label_tab({540, 100, 10, 50}, 60, 24, {}, 552), Rect{492, 76, 60, 24}));
    CHECK(eq(place_label_tab({-5, 100, 10, 50}, 60, 24, {}, 552), Rect{0, 76, 60, 24}));
}

TEST(tab_exactly_fits_above)
{
    CHECK(eq(place_label_tab({100, 24, 50, 50}, 60, 24, {}, 552), Rect{100, 0, 60, 24}));    // box.y == tab_h
    CHECK(eq(place_label_tab({100, 23, 50, 50}, 60, 24, {}, 552), Rect{100, 23, 60, 24}));   // one short: inside
}

TEST(tab_wider_than_screen_clamps_to_zero)
{
    CHECK(eq(place_label_tab({100, 100, 50, 50}, 600, 24, {}, 552), Rect{0, 76, 600, 24}));
}

int main() { return check::run_all(); }
