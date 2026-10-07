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
