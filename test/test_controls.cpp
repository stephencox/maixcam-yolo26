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
    // status pill: in bounds, not over any button, above the bottom row
    const Rect &st = l.status;
    CHECK(st.x >= 0 && st.y >= 0 && st.x + st.w <= sw && st.y + st.h <= sh);
    CHECK(st.w > 0 && st.h > 0);
    for (auto &b : r) CHECK(!overlap(st, b));
    CHECK(st.y + st.h <= l.conf_down.y);
}

TEST(layout_552x368) { check_layout(552, 368); }
TEST(layout_320x240) { check_layout(320, 240); }
TEST(layout_narrow_192x240) { check_layout(192, 240); }

TEST(status_pill_on_top_row_when_gap_wide)
{
    for (auto sz : {std::pair<int, int>{552, 368}, {320, 240}})
    {
        Layout l = make_layout(sz.first, sz.second);
        int gap = l.model.x - (l.exit.x + l.exit.w);
        CHECK(l.status.w >= 100);
        CHECK_EQ(l.status.h, l.exit.h);
        CHECK_EQ(l.status.y, l.exit.y);
        CHECK_EQ(l.status.w, gap - 8);
        CHECK_EQ(l.status.x, l.exit.x + l.exit.w + 4);   // centred in the gap
        CHECK(l.status.h >= 48);
    }
}

TEST(status_pill_second_row_when_gap_narrow)
{
    Layout l = make_layout(192, 240);
    CHECK_EQ(l.status.x, 4);
    CHECK_EQ(l.status.h, 32);
    CHECK(l.status.y >= l.exit.y + l.exit.h);
}

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

TEST(press_edge_fires_once_while_held)
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

TEST(threshold_bad_persisted_values_fall_back)
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
    CHECK(!parse_model_size("1024").has_value());
    CHECK(!parse_model_size("").has_value());
    CHECK_EQ(to_string(ModelSize::S640), std::string("640"));
}

TEST(status_row_split_boundary_99_vs_100)
{
    bool saw99 = false, saw100 = false;
    for (int sw = 150; sw <= 400; ++sw)
    {
        Layout l = make_layout(sw, 240);
        int pill_w = l.model.x - (l.exit.x + l.exit.w) - 8;
        if (pill_w == 99) { saw99 = true; CHECK_EQ(l.status.h, 32); CHECK(l.status.y > l.exit.y); }
        if (pill_w == 100) { saw100 = true; CHECK_EQ(l.status.y, l.exit.y); CHECK_EQ(l.status.w, 100); }
    }
    CHECK(saw99 && saw100);
}

int main() { return check::run_all(); }
