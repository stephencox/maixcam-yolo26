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
#include "style.hpp"
#include "strings.hpp"

#include <dirent.h>
#include <sys/stat.h>

#include <algorithm>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

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

static image::Color col(y26::Rgb c) { return image::Color::from_rgb(c.r, c.g, c.b); }

// Filled rounded rect: two overlapping rects plus four filled circles.
static void fill_round_rect(image::Image &img, const y26::Rect &r, int radius, y26::Rgb c)
{
    int rad = std::min(radius, std::min(r.w, r.h) / 2);
    image::Color k = col(c);
    img.draw_rect(r.x + rad, r.y, r.w - 2 * rad, r.h, k, -1);
    img.draw_rect(r.x, r.y + rad, r.w, r.h - 2 * rad, k, -1);
    img.draw_circle(r.x + rad, r.y + rad, rad, k, -1);
    img.draw_circle(r.x + r.w - 1 - rad, r.y + rad, rad, k, -1);
    img.draw_circle(r.x + rad, r.y + r.h - 1 - rad, rad, k, -1);
    img.draw_circle(r.x + r.w - 1 - rad, r.y + r.h - 1 - rad, rad, k, -1);
}

static y26::Rect inset(const y26::Rect &r, int d) { return {r.x + d, r.y + d, r.w - 2 * d, r.h - 2 * d}; }

static void draw_centered(image::Image &img, const y26::Rect &r, const std::string &text, y26::Rgb c, int dy = 0)
{
    image::Size ts = image::string_size(text);
    img.draw_string(r.x + (r.w - ts.width()) / 2, r.y + (r.h - ts.height()) / 2 + dy, text, col(c));
}

static void draw_button(image::Image &img, const y26::Rect &r, const std::string &text, y26::Rgb fill = y26::kPanel)
{
    fill_round_rect(img, inset(r, 1), 10, fill);
    draw_centered(img, r, text, y26::kText);
}

// Detection box with a filled label tab above it (inside the box top if it would leave the frame).
static void draw_detection(image::Image &img, const nn::Object &o, const std::string &name,
                           std::vector<y26::Rect> &placed, int sw, int sh)
{
    y26::Rgb c = y26::class_color(o.class_id);
    img.draw_rect(o.x, o.y, o.w, o.h, col(c), 3);
    std::string label = y26::format_label(name, o.score);
    image::Size ts = image::string_size(label);
    y26::Rect tab = y26::place_label_tab({o.x, o.y, o.w, o.h}, ts.width() + 12, ts.height() + 6, placed, sw, sh);
    placed.push_back(tab);
    img.draw_rect(tab.x, tab.y, tab.w, tab.h, col(c), -1);
    img.draw_string(tab.x + 6, tab.y + 3, label, col(y26::label_text_color(c)));
}

static void draw_pill(image::Image &img, const y26::Rect &r, const std::string &text, y26::Rgb text_color)
{
    fill_round_rect(img, r, r.h / 2, y26::kPanel);
    draw_centered(img, r, text, text_color);
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

static int run(display::Display &disp, touchscreen::TouchScreen &ts, y26::Lang lang)
{
    const int sw = disp.width(), sh = disp.height();
    const y26::Layout layout = y26::make_layout(sw, sh);

    y26::ModelSize size = y26::parse_model_size(app::get_app_config_kv(kCfg, "model", "320")).value_or(y26::ModelSize::S320);
    y26::Threshold conf = y26::Threshold::from_pct_string(app::get_app_config_kv(kCfg, "conf", "50"));

    show_message(disp, sw, sh, y26::tr(y26::Str::Loading, lang), image::COLOR_WHITE);
    // Camera first: if the model load below runs short of memory, the 320 fallback still has a camera.
    camera::Camera cam(sw, sh, image::FMT_RGB888);
    std::unique_ptr<nn::YOLO26> det = try_load(size);
    Toast toast;
    if (!det && size == y26::ModelSize::S640)
    {
        size = y26::ModelSize::S320;
        det = try_load(size);
        app::set_app_config_kv(kCfg, "model", y26::to_string(size));
        toast.show(y26::tr(y26::Str::Fallback320, lang), image::COLOR_YELLOW, 2000);
    }

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
                meter = y26::FpsMeter{};
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
                std::vector<y26::Rect> placed;
                for (const auto &o : *objs)
                    draw_detection(*img, o, det->labels[o.class_id], placed, sw, sh);
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
            fill_round_rect(*img, inset(layout.conf_value, 1), 10, y26::kPanel);
            {
                const y26::Rect &cv = layout.conf_value;
                std::string pct = std::to_string(conf.pct()) + "%";
                image::Size t1 = image::string_size(y26::tr(y26::Str::Conf, lang));
                image::Size t2 = image::string_size(pct);
                int top = cv.y + (cv.h - (t1.height() + t2.height())) / 2;
                img->draw_string(cv.x + (cv.w - t1.width()) / 2, top, y26::tr(y26::Str::Conf, lang),
                                 col(y26::Rgb{170, 176, 184}));
                img->draw_string(cv.x + (cv.w - t2.width()) / 2, top + t1.height(), pct, col(y26::kText));
            }
            draw_button(*img, layout.snapshot, y26::tr(y26::Str::Snapshot, lang), y26::kAccent);
            draw_pill(*img, layout.status, y26::format_status(meter.fps(), meter.latency_ms()), y26::kText);
            if (time::ticks_ms() < toast.until_ms)
            {
                image::Size tsz = image::string_size(toast.text);
                y26::Rect tr{layout.status.x, layout.status.y + layout.status.h + 4, tsz.width() + 24, tsz.height() + 10};
                fill_round_rect(*img, tr, tr.h / 2, y26::kPanel);
                img->draw_string(tr.x + 12, tr.y + 5, toast.text, toast.color);
            }

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

int main()
{
    std::unique_ptr<display::Display> disp;
    std::unique_ptr<touchscreen::TouchScreen> ts;
    try
    {
        y26::Lang lang = y26::lang_from_locale(i18n::get_locale());
        if (image::load_font("ui", kFont, 22) == err::ERR_NONE)
            image::set_default_font("ui");
        else
            log::warn("font %s not loaded; Chinese text may not render", kFont);
        disp = std::make_unique<display::Display>();
        ts = std::make_unique<touchscreen::TouchScreen>();
        return run(*disp, *ts, lang);
    }
    catch (const std::exception &e)
    {
        log::error("fatal: %s", e.what());
        // Show the error and wait for Exit / the launcher; no crash loop.
        try
        {
            if (disp)
            {
                const y26::Layout layout = y26::make_layout(disp->width(), disp->height());
                while (!app::need_exit())
                {
                    image::Image screen(disp->width(), disp->height(), image::FMT_RGB888);
                    screen.clear();
                    screen.draw_string(16, disp->height() / 2 - 12, std::string("Error: ") + e.what(), image::COLOR_RED);
                    draw_button(screen, layout.exit, "Exit");
                    disp->show(screen);
                    int tx = 0, ty = 0;
                    bool pressed = false;
                    if (ts)
                        ts->read(tx, ty, pressed);
                    if (pressed && y26::hit_test(layout, tx, ty) == y26::Button::Exit)
                        app::set_exit_flag(true);
                    time::sleep_ms(50);
                }
            }
            else
                while (!app::need_exit())
                    time::sleep_ms(100);
        }
        catch (const std::exception &e2)
        {
            log::error("error screen failed: %s", e2.what());
        }
        return 1;
    }
}
