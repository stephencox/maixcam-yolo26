// Rolling frame-rate and inference-latency meter. Pure C++, host-tested.
#pragma once
#include <array>
#include <cstddef>
#include <cstdio>
#include <optional>
#include <string>

namespace y26
{
    class FpsMeter
    {
    public:
        static constexpr std::size_t kWindow = 30;

        // Non-positive durations (first frame, clock steps) are dropped.
        void add(double frame_ms, double infer_ms)
        {
            if (frame_ms > 0) frames_.push(frame_ms);
            if (infer_ms > 0) infers_.push(infer_ms);
        }

        std::optional<double> fps() const
        {
            auto m = frames_.mean();
            if (!m) return std::nullopt;
            return 1000.0 / *m;
        }

        std::optional<double> latency_ms() const { return infers_.mean(); }

    private:
        struct Ring
        {
            std::array<double, kWindow> v{};
            std::size_t n = 0, i = 0;
            void push(double x) { v[i] = x; i = (i + 1) % kWindow; if (n < kWindow) ++n; }
            std::optional<double> mean() const
            {
                if (n == 0) return std::nullopt;
                double s = 0;
                for (std::size_t k = 0; k < n; ++k) s += v[k];
                return s / static_cast<double>(n);
            }
        };
        Ring frames_, infers_;
    };

    inline std::string format_status(std::optional<double> fps, std::optional<double> latency_ms, int model_px)
    {
        char f[16] = "-", l[16] = "-";
        if (fps) std::snprintf(f, sizeof f, "%.1f", *fps);
        if (latency_ms) std::snprintf(l, sizeof l, "%.0f", *latency_ms);
        char out[64];
        std::snprintf(out, sizeof out, "FPS %s | %s ms | %d", f, l, model_px);
        return out;
    }
}
