#pragma once
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace check
{
    struct Case { const char *name; std::function<void()> fn; };
    inline std::vector<Case> &cases() { static std::vector<Case> c; return c; }
    inline int &failures() { static int f = 0; return f; }
    struct Reg { Reg(const char *n, std::function<void()> f) { cases().push_back({n, std::move(f)}); } };
    inline void fail(const char *file, int line, const std::string &msg)
    {
        std::fprintf(stderr, "%s:%d: FAIL %s\n", file, line, msg.c_str());
        ++failures();
    }
    inline int run_all()
    {
        for (auto &c : cases()) c.fn();
        std::printf("%zu tests, %d failures\n", cases().size(), failures());
        return failures() ? 1 : 0;
    }
}

#define CHECK_CAT2(a, b) a##b
#define CHECK_CAT(a, b) CHECK_CAT2(a, b)
#define TEST(name)                                                        \
    static void name();                                                   \
    static check::Reg CHECK_CAT(reg_, name)(#name, name);                 \
    static void name()
#define CHECK(c) do { if (!(c)) check::fail(__FILE__, __LINE__, #c); } while (0)
#define CHECK_EQ(a, b) do { if (!((a) == (b))) check::fail(__FILE__, __LINE__, #a " == " #b); } while (0)
#define CHECK_NEAR(a, b, eps) do { if (std::fabs((a) - (b)) > (eps)) check::fail(__FILE__, __LINE__, #a " ~= " #b); } while (0)
