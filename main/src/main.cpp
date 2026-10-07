// YOLO26 detector app. The only file that touches MaixCDK; all decisions
// live in main/include/*.hpp and are host-tested.
#include "maix_basic.hpp"
#include "maix_camera.hpp"
#include "maix_display.hpp"
#include "maix_image.hpp"
#include "maix_nn_yolo26.hpp"
#include "maix_touchscreen.hpp"

#include "controls.hpp"
#include "fps_meter.hpp"
#include "strings.hpp"

#include <memory>

using namespace maix;

static const char *kFont = "/maixapp/share/font/SourceHanSansCN-Regular.otf";

static std::string asset(const std::string &name)
{
    std::string base = app::get_app_path();
    return (base.empty() ? std::string(".") : base) + "/assets/" + name;
}

static std::string mud_for(y26::ModelSize m) { return asset("yolo26n_" + y26::to_string(m) + ".mud"); }

static void draw_button(image::Image &img, const y26::Rect &r, const std::string &text)
{
    img.draw_rect(r.x, r.y, r.w, r.h, image::COLOR_WHITE, 2);
    image::Size ts = image::string_size(text);
    img.draw_string(r.x + (r.w - ts.width()) / 2, r.y + (r.h - ts.height()) / 2, text, image::COLOR_WHITE);
}

int main()
{
    y26::Lang lang = y26::lang_from_locale(i18n::get_locale());
    if (image::load_font("ui", kFont, 22) == err::ERR_NONE)
        image::set_default_font("ui");

    display::Display disp;
    touchscreen::TouchScreen ts;
    const int sw = disp.width(), sh = disp.height();
    const y26::Layout layout = y26::make_layout(sw, sh);

    y26::ModelSize size = y26::ModelSize::S320;
    std::unique_ptr<nn::YOLO26> det;
    try
    {
        det = std::make_unique<nn::YOLO26>(mud_for(size), false);
        log::info("YOLO26 loaded: %dx%d", det->input_width(), det->input_height());
    }
    catch (const std::exception &e)
    {
        log::error("model load failed: %s", e.what());
    }

    camera::Camera cam(sw, sh, image::FMT_RGB888);
    y26::FpsMeter meter;
    uint64_t last_us = time::ticks_us();

    while (!app::need_exit())
    {
        int tx = 0, ty = 0;
        bool pressed = false;
        ts.read(tx, ty, pressed);
        if (pressed && y26::hit_test(layout, tx, ty) == y26::Button::Exit)
            app::set_exit_flag(true);

        image::Image *raw = cam.read();
        if (!raw)
            continue;
        std::unique_ptr<image::Image> img(raw);

        double infer_ms = 0;
        if (det)
        {
            uint64_t t0 = time::ticks_us();
            std::unique_ptr<std::vector<nn::Object>> objs(det->detect(*img, 0.5f, 0.45f));
            infer_ms = (time::ticks_us() - t0) / 1000.0;
            for (const auto &o : *objs)
            {
                img->draw_rect(o.x, o.y, o.w, o.h, image::COLOR_RED, 3);
                char msg[64];
                snprintf(msg, sizeof msg, "%s %.2f", det->labels[o.class_id].c_str(), o.score);
                img->draw_string(o.x, std::max(0, o.y - 24), msg, image::COLOR_RED);
            }
        }
        else
        {
            img->draw_string(16, sh / 2, std::string(y26::tr(y26::Str::NoModel, lang)) + " " + mud_for(size), image::COLOR_RED);
        }

        draw_button(*img, layout.exit, y26::tr(y26::Str::Exit, lang));
        img->draw_string(layout.status_x, layout.status_y,
                         y26::format_status(meter.fps(), meter.latency_ms(), y26::pixels(size)), image::COLOR_GREEN);
        disp.show(*img);

        uint64_t now = time::ticks_us();
        meter.add((now - last_us) / 1000.0, infer_ms);
        last_us = now;
    }
    return 0;
}
