#
# ncnn — macOS: ncnn (Vulkan via MoltenVK discovery) static frameworks
# from the official ncnn apple-vulkan release.
#
# The prebuilt archive is fetched by the prepare_command into
# build/ncnn-apple-vulkan/ and vendored via vendored_frameworks.
#
Pod::Spec.new do |s|
  s.name             = "ncnn"
  s.version          = "0.1.0"
  s.summary          = "ncnn (Vulkan) inference bindings for Dart/Flutter"
  s.description      = "Thin C shim over ncnn::Net, exposed via dart:ffi."
  s.homepage         = "https://github.com/hhoao/ncnn"
  s.license          = "BSD-3-Clause"
  s.author           = { "hhoao" => "hhoao@users.noreply.github.com" }
  s.source           = { :path => "." }
  s.ios.deployment_target  = "12.0"
  s.osx.deployment_target  = "10.15"

  ncnn_version = "20260526"
  ncnn_dir = "build/ncnn-apple-vulkan"
  mvk_version = "1.4.2"
  mvk_dir = "build/moltenvk"

  s.prepare_command = <<-CMD
    mkdir -p build
    if [ ! -d "#{ncnn_dir}/ncnn_static.xcframework" ]; then
      curl -fL --retry 3 \
        "https://github.com/Tencent/ncnn/releases/download/#{ncnn_version}/ncnn-#{ncnn_version}-apple-vulkan.zip" \
        -o build/ncnn-apple-vulkan.zip
      unzip -qo build/ncnn-apple-vulkan.zip -d #{ncnn_dir}
      rm -f build/ncnn-apple-vulkan.zip
      # Rename the vendored bundle ncnn.framework -> ncnn_static.framework
      # (binary too; the ar symbol table is untouched by a rename).
      #
      # WHY: the pod is named "ncnn", so the pod target's own product is
      # ${PODS_CONFIGURATION_BUILD_DIR}/ncnn/ncnn.framework, while CocoaPods
      # stages vendored xcframework slices into
      # ${PODS_CONFIGURATION_BUILD_DIR}/XCFrameworkIntermediates/ncnn/. The
      # app links with `-framework ncnn`, and FRAMEWORK_SEARCH_PATHS lists
      # BOTH dirs — the pod-product dir sorts first, so ld resolves the flag
      # to the pod's own (shim-only) framework and every ncnn:: C++ symbol
      # comes out undefined (seen in hhoao/huji#2 macOS CI). Renaming the
      # vendored bundle disambiguates the link flag: `-framework ncnn` keeps
      # meaning the shim product, `-framework ncnn_static` the real library.
      # (Pre-migration this pod was called huji_ncnn — no collision.)
      for fw in #{ncnn_dir}/ncnn.xcframework/*/ncnn.framework; do
        d="$(dirname "$fw")"
        mv "$fw" "$d/ncnn_static.framework"
        mv "$d/ncnn_static.framework/Versions/A/ncnn" \
           "$d/ncnn_static.framework/Versions/A/ncnn_static"
        rm -f "$d/ncnn_static.framework/ncnn"
        ln -s Versions/Current/ncnn_static "$d/ncnn_static.framework/ncnn_static"
      done
      perl -pi -e 's{ncnn\.framework/Versions/A/ncnn}{ncnn_static.framework/Versions/A/ncnn_static}g; s{>ncnn\.framework<}{>ncnn_static.framework<}g' \
        #{ncnn_dir}/ncnn.xcframework/Info.plist
      mv #{ncnn_dir}/ncnn.xcframework #{ncnn_dir}/ncnn_static.xcframework
    fi
    # ncnn's apple-vulkan build links against the Vulkan loader; on macOS the
    # loader is MoltenVK (Xcode no longer bundles it). Stream-extract just the
    # dynamic xcframework from the release tarball.
    mkdir -p #{mvk_dir}
    if [ ! -d "#{mvk_dir}/MoltenVK.xcframework" ]; then
      curl -fL --retry 3 \
        "https://github.com/KhronosGroup/MoltenVK/releases/download/v#{mvk_version}/MoltenVK-macos.tar" \
        | tar -x -C #{mvk_dir} --strip-components=3 MoltenVK/MoltenVK/dynamic/MoltenVK.xcframework
    fi
  CMD

  s.vendored_frameworks = [
    "#{ncnn_dir}/ncnn_static.xcframework",
    "#{ncnn_dir}/glslang.xcframework",
    "#{ncnn_dir}/openmp.xcframework",
    "#{mvk_dir}/MoltenVK.xcframework",
  ]

  s.pod_target_xcconfig = {
    "DEFINES_MODULE" => "YES",
    "EXCLUDED_ARCHS[sdk=iphonesimulator*]" => "",
    "CLANG_CXX_LANGUAGE_STANDARD" => "c++17",
    # The shim includes "ncnn/net.h". Vendored xcframework headers are not
    # on the include path by default; point at the staged (platform-selected)
    # Headers dir. "Prepare xcframeworks" phases populate it before compile.
    # ncnn_static.framework: the vendored bundle is renamed in the
    # prepare_command (see WHY there — pod-product/vendored-framework name
    # collision on `ncnn`).
    "HEADER_SEARCH_PATHS" => "$(PODS_CONFIGURATION_BUILD_DIR)/XCFrameworkIntermediates/ncnn/ncnn_static.framework/Headers",
  }

  # NOTE: `src` contains committed real-file copies of the shared shim
  # sources at the package root — CocoaPods file patterns cannot
  # reference files outside the podspec directory (macos/), and `dart pub
  # publish` replaces symlinks with duplicate regular files anyway.
  # tool/sync_apple_sources.sh keeps them in sync (CI asserts freshness).
  #
  # ncnn_link_anchor.m keeps the FFI entry points alive at app link
  # time (pure-FFI pod, nothing references hn_* natively; see the file).
  s.source_files = "src/ncnn_api.{h,cpp}", "src/ncnn_link_anchor.m"
  s.public_header_files = "src/ncnn_api.h"
  s.requires_arc = false
end
