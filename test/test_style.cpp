#include "check.hpp"
#include "style.hpp"

using namespace y26;

static bool same(Rgb a, Rgb b) { return a.r == b.r && a.g == b.g && a.b == b.b; }

TEST(constants)
{
    CHECK(same(kAccent, Rgb{255, 90, 78}));
    CHECK(same(kPanel, Rgb{29, 33, 39}));
    CHECK(same(kText, Rgb{255, 255, 255}));
}

TEST(palette_entry_zero_is_accent)
{
    CHECK(same(class_color(0), kAccent));
    CHECK(same(kClassPalette[0], kAccent));
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
    CHECK(same(class_color(-1), kAccent));
    CHECK(same(class_color(-100), kAccent));
}

TEST(palette_entries_distinct)
{
    for (int i = 0; i < 8; ++i)
        for (int j = i + 1; j < 8; ++j) CHECK(!same(kClassPalette[i], kClassPalette[j]));
}

TEST(label_text_color_contrast)
{
    const Rgb dark{16, 20, 24};
    CHECK(same(label_text_color(Rgb{255, 193, 69}), dark));    // amber
    CHECK(same(label_text_color(Rgb{167, 201, 87}), dark));    // lime
    CHECK(same(label_text_color(Rgb{155, 93, 229}), kText));   // violet
    CHECK(same(label_text_color(kPanel), kText));
    CHECK(same(label_text_color(Rgb{255, 255, 255}), dark));
    CHECK(same(label_text_color(Rgb{0, 0, 0}), kText));
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
    CHECK(eq(place_label_tab({100, 100, 50, 50}, 60, 24, {}, 552, 368), Rect{100, 76, 60, 24}));
}

TEST(tab_inside_when_above_is_off_screen)
{
    CHECK(eq(place_label_tab({100, 10, 50, 50}, 60, 24, {}, 552, 368), Rect{100, 10, 60, 24}));
}

TEST(tab_inside_on_collision)
{
    std::vector<Rect> placed{{110, 70, 60, 24}};
    CHECK(eq(place_label_tab({100, 100, 50, 50}, 60, 24, placed, 552, 368), Rect{100, 100, 60, 24}));
}

TEST(tab_inside_kept_even_if_colliding)
{
    std::vector<Rect> placed{{110, 70, 60, 24}, {100, 100, 60, 24}};
    CHECK(eq(place_label_tab({100, 100, 50, 50}, 60, 24, placed, 552, 368), Rect{100, 100, 60, 24}));
}

TEST(tab_touching_is_not_collision)
{
    std::vector<Rect> placed{{100, 52, 60, 24}};   // ends at y=76, tab starts at 76
    CHECK(eq(place_label_tab({100, 100, 50, 50}, 60, 24, placed, 552, 368), Rect{100, 76, 60, 24}));
}

TEST(tab_right_edge_clamp)
{
    CHECK(eq(place_label_tab({540, 100, 10, 50}, 60, 24, {}, 552, 368), Rect{492, 76, 60, 24}));
    CHECK(eq(place_label_tab({-5, 100, 10, 50}, 60, 24, {}, 552, 368), Rect{0, 76, 60, 24}));
}

int main() { return check::run_all(); }
