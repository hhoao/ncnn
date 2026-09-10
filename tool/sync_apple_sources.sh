#!/usr/bin/env bash
# Syncs the shared shim sources into ios/src and macos/src as real file
# copies. CocoaPods file patterns cannot reference files outside the
# podspec directory, and `dart pub publish` does NOT preserve symlinks
# (each link is uploaded as a duplicate regular file), so the Apple
# sides keep committed copies instead of links.
#
# CI asserts freshness with:
#   tool/sync_apple_sources.sh && git diff --exit-code ios/src macos/src
set -euo pipefail
cd "$(dirname "$0")/.."
for d in ios/src macos/src; do
  mkdir -p "$d"
  cp src/ncnn_api.h src/ncnn_api.cpp src/ncnn_link_anchor.m "$d/"
done
echo "synced src/ncnn_api.{h,cpp}, src/ncnn_link_anchor.m -> ios/src, macos/src"
