#include "check.hpp"
#include "snapshot.hpp"

using namespace y26;

TEST(empty_dir_starts_at_1) { CHECK_EQ(next_snapshot_index({}), 1); }

TEST(max_plus_one)
{
    CHECK_EQ(next_snapshot_index({"yolo26_1.jpg", "yolo26_7.jpg", "yolo26_3.jpg"}), 8);
    CHECK_EQ(next_snapshot_index({"yolo26_007.jpg"}), 8);
}

TEST(ignores_odd_names)   // Review Focus 3
{
    CHECK_EQ(next_snapshot_index({"yolo26_.jpg", "yolo26_12a.jpg", "yolo26_5.png", "notes.txt",
                                  "yolo26_-3.jpg", "Yolo26_9.jpg", "yolo26_99999999999999.jpg"}), 1);
}

TEST(name_format) { CHECK_EQ(snapshot_name(42), std::string("yolo26_42.jpg")); }

int main() { return check::run_all(); }
