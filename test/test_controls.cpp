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

int main() { return check::run_all(); }
