#include "check.hpp"
#include "strings.hpp"
#include <cstring>

using namespace y26;

TEST(every_key_has_both_languages)
{
    for (int i = 0; i < static_cast<int>(Str::Count_); ++i)
    {
        auto k = static_cast<Str>(i);
        CHECK(tr(k, Lang::En) != nullptr && std::strlen(tr(k, Lang::En)) > 0);
        CHECK(tr(k, Lang::Zh) != nullptr && std::strlen(tr(k, Lang::Zh)) > 0);
    }
}

TEST(locale_selection)
{
    CHECK(lang_from_locale("zh") == Lang::Zh);
    CHECK(lang_from_locale("zh_CN") == Lang::Zh);
    CHECK(lang_from_locale("zh-TW") == Lang::Zh);
    CHECK(lang_from_locale("en") == Lang::En);
    CHECK(lang_from_locale("ja") == Lang::En);
    CHECK(lang_from_locale("") == Lang::En);
    CHECK(lang_from_locale("z") == Lang::En);
}

TEST(english_text)
{
    CHECK_EQ(std::string(tr(Str::Exit, Lang::En)), std::string("< Exit"));
    CHECK_EQ(std::string(tr(Str::Saved, Lang::En)), std::string("Saved"));
}

int main() { return check::run_all(); }
