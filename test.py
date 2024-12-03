from delocate.fuse import fuse_wheels

fuse_wheels(
    "Pillow-9.4.0-2-cp39-cp39-macosx_10_10_x86_64.whl",
    "Pillow-9.4.0-cp39-cp39-macosx_11_0_arm64.whl",
    "Pillow-9.4.0-cp39-cp39-macosx_11_0_universal2.whl",
)
