set shell := ["bash", "-euo", "pipefail", "-c"]

cxx := env_var_or_default("CXX", "g++")
cxxflags := "-std=c++17 -O1 -g -Wall -Wextra -Werror -fsanitize=address,undefined -fno-omit-frame-pointer -Imain/include -Itest"

# Host tests for the pure headers
test:
    mkdir -p test/bin
    for t in test/test_*.cpp; do \
        n=$(basename "$t" .cpp); \
        {{cxx}} {{cxxflags}} "$t" -o "test/bin/$n" && "test/bin/$n"; \
    done
    bash test/test_check_release.sh

gpu := env_var_or_default("GPU", "192.168.12.30")

# Rebuild + verify one model size on the GPU box (export, INT8+BF16 head, simulator mAP gate)
models size="320":
    rsync -a --exclude .git --exclude .superpowers --exclude model/.venv --exclude model/work --exclude 'model/exp_*' ./ {{gpu}}:maixcam-yolo26/
    ssh {{gpu}} 'set -e; cd maixcam-yolo26/model; .venv/bin/pytest -q; \
      .venv/bin/python export.py --size {{size}}; ./convert.sh {{size}}; \
      for L in eval board; do docker run --rm -v "$PWD":/m -v ~/coco:/coco -w /m maixcam-yolo26/tpu_mlir:1.28.1 \
        python3 sim_detect.py --size {{size}} --list lists/$L.txt --coco /coco --out work/{{size}}/sim_$L.json; done; \
      .venv/bin/python check_map.py {{size}}'
    @echo "now run: just smoke {{size}}"

board := env_var_or_default("BOARD", "maixcam")

# Cross-compile the app for the MaixCAM (MaixCDK in docker)
build:
    scripts/build.sh build

release-local:
    scripts/build.sh release
    ls -l dist/*.zip

# Package, copy to the board, install via the app store CLI (not on PATH on the board)
deploy: release-local
    scp dist/yolo26_detect_v$(sed -n 's/^version:[[:space:]]*//p' app.yaml | head -1).zip {{board}}:/tmp/yolo26_detect.zip
    ssh {{board}} '/maixapp/apps/app_store/app_store install /tmp/yolo26_detect.zip && rm /tmp/yolo26_detect.zip'

# Board vs simulator compare for one model size. Run ONLY with the app closed
# (the launcher is fine: board_detect uses the NPU, not the camera); never while
# yolo26_detect is running. Needs the model/work/<size>/sim_board.json from `just models`.
smoke size="320":
    mkdir -p model/work/{{size}}
    [ -f model/work/{{size}}/sim_board.json ] || scp {{gpu}}:maixcam-yolo26/model/work/{{size}}/sim_board.json model/work/{{size}}/sim_board.json
    ssh {{board}} 'rm -rf /root/y26 && mkdir -p /root/y26/imgs'
    scp assets/yolo26n_{{size}}.mud assets/yolo26n_{{size}}_int8.cvimodel model/board_detect.py {{board}}:/root/y26/
    scp model/board_images/*.jpg {{board}}:/root/y26/imgs/
    ssh {{board}} 'cd /root/y26 && python3 board_detect.py yolo26n_{{size}}.mud imgs 0.45' | sed -n 's/^Y26JSON //p' > model/work/{{size}}/board.json
    python3 model/compare_board.py model/work/{{size}}/sim_board.json model/work/{{size}}/board.json
