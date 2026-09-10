# Third-party notices

This package builds against and redistributes (at build time) the
following third-party software:

## ncnn — BSD 3-Clause
- Source: https://github.com/Tencent/ncnn
- Release: 20260526 (official prebuilt archives, downloaded at build
  time by the platform build scripts; not committed to this repository)
- License: https://github.com/Tencent/ncnn/blob/master/LICENSE.txt

## MoltenVK — Apache License 2.0
- Source: https://github.com/KhronosGroup/MoltenVK
- Release: 1.4.2 (official tarball, downloaded by the Apple podspec
  prepare_command and vendored into the app bundle at build time)
- License: https://github.com/KhronosGroup/MoltenVK/blob/main/LICENSE

The prebuilt ncnn archives used per platform:
- Linux x86_64: ncnn-20260526-ubuntu-2204-shared.zip
- Windows x64: ncnn-20260526-windows-vs2022-shared.zip
- Android: ncnn-20260526-android-vulkan.zip (static, per-ABI)
- iOS: ncnn-20260526-ios-vulkan.zip (static frameworks)
- macOS: ncnn-20260526-apple-vulkan.zip (static frameworks)
