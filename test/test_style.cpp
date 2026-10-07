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

int main() { return check::run_all(); }
