// On-screen text in English and Chinese. Pure C++, host-tested.
#pragma once
#include <array>
#include <string>

namespace y26
{
    enum class Lang { En, Zh };

    inline Lang lang_from_locale(const std::string &loc)
    {
        return loc.rfind("zh", 0) == 0 ? Lang::Zh : Lang::En;
    }

    enum class Str { Exit, Snapshot, Conf, Saved, SaveFailed, Loading, Fallback320, NoModel, Count_ };

    inline const char *tr(Str s, Lang l)
    {
        static const std::array<std::array<const char *, 2>, static_cast<size_t>(Str::Count_)> t = {{
            {"< Exit", "< 退出"},
            {"Snapshot", "拍照"},
            {"conf", "置信度"},
            {"Saved", "已保存"},
            {"Save failed", "保存失败"},
            {"Loading...", "加载中..."},
            {"640 model failed, using 320", "640 模型加载失败，改用 320"},
            {"Model not found:", "找不到模型："},
        }};
        return t[static_cast<size_t>(s)][l == Lang::Zh ? 1 : 0];
    }
}
