LD_LIBRARY_PATH="$LIBTORCH/lib" RUST_BACKTRACE=1 LIBTORCH_STATIC=0 cargo r --features bevy/dynamic_linking,ml --release --bin pitchvis
