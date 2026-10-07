// Snapshot file naming. Numbered, not timestamped: the RTC may be unset.
#pragma once
#include <string>
#include <vector>

namespace y26
{
    inline std::string snapshot_name(int index) { return "yolo26_" + std::to_string(index) + ".jpg"; }

    // Highest valid "yolo26_<digits>.jpg" + 1. Names with non-digits, no digits,
    // or more than 9 digits (would overflow int) are ignored.
    inline int next_snapshot_index(const std::vector<std::string> &names)
    {
        const std::string pre = "yolo26_", suf = ".jpg";
        int best = 0;
        for (const auto &n : names)
        {
            if (n.size() <= pre.size() + suf.size()) continue;
            if (n.compare(0, pre.size(), pre) != 0) continue;
            if (n.compare(n.size() - suf.size(), suf.size(), suf) != 0) continue;
            std::string digits = n.substr(pre.size(), n.size() - pre.size() - suf.size());
            if (digits.empty() || digits.size() > 9) continue;
            int v = 0;
            bool ok = true;
            for (char c : digits)
            {
                if (c < '0' || c > '9') { ok = false; break; }
                v = v * 10 + (c - '0');
            }
            if (ok && v > best) best = v;
        }
        return best + 1;
    }
}
