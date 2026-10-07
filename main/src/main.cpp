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
#include "snapshot.hpp"
#include "strings.hpp"

#include <dirent.h>
#include <sys/stat.h>

#include <memory>

using namespace maix;

static const char *kFont = "/maixapp/share/font/SourceHanSansCN-Regular.otf";
static const char *kSnapDir = "/root/snapshots";
static const char *kCfg = "yolo26_detect";

struct Toast
{
    std::string text;
    image::Color color = image::COLOR_WHITE;
    uint64_t until_ms = 0;
    void show(const std::string &t, image::Color c, uint64_t dur_ms)
    {
        text = t;
        color = c;
        until_ms = time::ticks_ms() + dur_ms;
    }
};

static std::string asset(const std::string &name)
{
    std::string base = app::get_app_path();
    return (base.empty() ? std::string(".") : base) + "/assets/" + name;
}

static std::string mud_for(y26::ModelSize m) { return asset("yolo26n_" + y26::to_string(m) + ".mud"); }

static std::unique_ptr<nn::YOLO26> try_load(y26::ModelSize m)
{
    try
    {
        auto d = std::make_unique<nn::YOLO26>(mud_for(m), false);
        log::info("YOLO26 loaded: %dx%d", d->input_width(), d->input_height());
        return d;
    }
    catch (const std::exception &e)
    {
        log::error("load %s failed: %s", mud_for(m).c_str(), e.what());
        return nullptr;
    }
}

static void draw_button(image::Image &img, const y26::Rect &r, const std::string &text)
{
    img.draw_rect(r.x, r.y, r.w, r.h, image::COLOR_WHITE, 2);
    image::Size ts = image::string_size(text);
    img.draw_string(r.x + (r.w - ts.width()) / 2, r.y + (r.h - ts.height()) / 2, text, image::COLOR_WHITE);
}

static void show_message(display::Display &disp, int sw, int sh, const std::string &text, image::Color c)
{
    image::Image screen(sw, sh, image::FMT_RGB888);
    screen.clear();
    screen.draw_string(16, sh / 2 - 12, text, c);
    disp.show(screen);
}

static bool save_snapshot(image::Image &img)
{
    mkdir(kSnapDir, 0755);
    std::vector<std::string> names;
    if (DIR *d = opendir(kSnapDir))
    {
        while (dirent *e = readdir(d))
            names.emplace_back(e->d_name);
        closedir(d);
    }
    std::string path = std::string(kSnapDir) + "/" + y26::snapshot_name(y26::next_snapshot_index(names));
    return img.save(path.c_str()) == err::ERR_NONE;
}

int main()
{
    y26::Lang lang = y26::lang_from_locale(i18n::get_locale());
    if (image::load_font("ui", kFont, 22) == err::ERR_NONE)
        image::set_default_font("ui");
    else
        log::warn("font %s not loaded; Chinese text may not render", kFont);

    display::Display disp;
    touchscreen::TouchScreen ts;
    const int sw = disp.width(), sh = disp.height();
    const y26::Layout layout = y26::make_layout(sw, sh);

    y26::ModelSize size = y26::parse_model_size(app::get_app_config_kv(kCfg, "model", "320")).value_or(y26::ModelSize::S320);
    y26::Threshold conf = y26::Threshold::from_pct_string(app::get_app_config_kv(kCfg, "conf", "50"));

    show_message(disp, sw, sh, y26::tr(y26::Str::Loading, lang), image::COLOR_WHITE);
    std::unique_ptr<nn::YOLO26> det = try_load(size);
    Toast toast;
    if (!det && size == y26::ModelSize::S640)
    {
        size = y26::ModelSize::S320;
        det = try_load(size);
        toast.show(y26::tr(y26::Str::Fallback320, lang), image::COLOR_YELLOW, 2000);
    }

    camera::Camera cam(sw, sh, image::FMT_RGB888);
    y26::FpsMeter meter;
    y26::PressEdge edge;
    uint64_t last_us = time::ticks_us();

    while (!app::need_exit())
    {
        try
        {
            int tx = 0, ty = 0;
            bool pressed = false;
            ts.read(tx, ty, pressed);
            y26::Button btn = edge.update(pressed) ? y26::hit_test(layout, tx, ty) : y26::Button::None;
            bool want_snapshot = false;

            switch (btn)
            {
            case y26::Button::Exit:
                app::set_exit_flag(true);
                continue;
            case y26::Button::ConfDown:
                conf.down();
                app::set_app_config_kv(kCfg, "conf", conf.to_pct_string());
                break;
            case y26::Button::ConfUp:
                conf.up();
                app::set_app_config_kv(kCfg, "conf", conf.to_pct_string());
                break;
            case y26::Button::Model:
            {
                y26::ModelSize want = y26::other(size);
                show_message(disp, sw, sh, y26::tr(y26::Str::Loading, lang), image::COLOR_WHITE);
                det.reset();                                  // free NPU memory before loading the other
                det = try_load(want);
                if (det)
                    size = want;
                else
                {
                    toast.show(y26::tr(y26::Str::Fallback320, lang), image::COLOR_YELLOW, 2000);
                    size = y26::ModelSize::S320;
                    det = try_load(size);
                }
                app::set_app_config_kv(kCfg, "model", y26::to_string(size));
                last_us = time::ticks_us();                   // don't count the load in FPS
                break;
            }
            case y26::Button::Snapshot:
                want_snapshot = true;
                break;
            case y26::Button::None:
                break;
            }

            image::Image *raw = cam.read();
            if (!raw)
                continue;
            std::unique_ptr<image::Image> img(raw);

            double infer_ms = 0;
            if (det)
            {
                uint64_t t0 = time::ticks_us();
                std::unique_ptr<std::vector<nn::Object>> objs(det->detect(*img, conf.value(), 0.45f));
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
                img->draw_string(16, sh / 2, std::string(y26::tr(y26::Str::NoModel, lang)) + " " + mud_for(size),
                                 image::COLOR_RED);
            }

            // Snapshot = detections only, before the UI chrome is drawn.
            if (want_snapshot)
            {
                bool ok = save_snapshot(*img);
                toast.show(y26::tr(ok ? y26::Str::Saved : y26::Str::SaveFailed, lang),
                           ok ? image::COLOR_GREEN : image::COLOR_RED, 1500);
            }

            char confs[32];
            snprintf(confs, sizeof confs, "%.2f", conf.value());
            draw_button(*img, layout.exit, y26::tr(y26::Str::Exit, lang));
            draw_button(*img, layout.model, y26::to_string(size));
            draw_button(*img, layout.conf_down, "-");
            draw_button(*img, layout.conf_up, "+");
            img->draw_string(layout.conf_value.x + 6, layout.conf_value.y + 4, y26::tr(y26::Str::Conf, lang), image::COLOR_WHITE);
            img->draw_string(layout.conf_value.x + 6, layout.conf_value.y + layout.conf_value.h / 2, confs, image::COLOR_WHITE);
            draw_button(*img, layout.snapshot, y26::tr(y26::Str::Snapshot, lang));
            img->draw_string(layout.status_x, layout.status_y,
                             y26::format_status(meter.fps(), meter.latency_ms(), y26::pixels(size)), image::COLOR_GREEN);
            if (time::ticks_ms() < toast.until_ms)
                img->draw_string(layout.status_x, layout.status_y + 32, toast.text, toast.color);

            disp.show(*img);

            uint64_t now = time::ticks_us();
            meter.add((now - last_us) / 1000.0, infer_ms);
            last_us = now;
        }
        catch (const std::exception &e)
        {
            log::error("frame skipped: %s", e.what());
        }
    }
    return 0;
}
