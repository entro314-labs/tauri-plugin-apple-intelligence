#!/usr/bin/env bash

# Build Swift dylib
set -euo pipefail

echo "🔨  Building Apple On-Device AI library components …"

# Resolve paths relative to this script, not the caller's CWD, so the build
# works no matter where it's invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC="${ROOT_DIR}/ailib/apple-ai.swift"
# Emit straight into prebuilt/, which is where build.rs and the linker consume it.
OUT_DIR="${ROOT_DIR}/prebuilt"

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ Error: This library can only be built on macOS"
    exit 1
fi

# Require macOS 26+ (FoundationModels)
MACOS_MAJOR=$(sw_vers -productVersion | cut -d. -f1)
if (( MACOS_MAJOR < 26 )); then
  echo "❌  Need macOS 26.0+ (FoundationModels). Current: $(sw_vers -productVersion)" >&2
  exit 1
fi

# Create output directory
mkdir -p "${OUT_DIR}"

# Build Swift dylib
echo "📦  Swift → ${OUT_DIR}/libappleai.dylib"
swiftc \
  -O -whole-module-optimization \
  -emit-library -emit-module -module-name AppleOnDeviceAI \
  -framework Foundation -framework FoundationModels \
  -target arm64-apple-macos26.0 \
  -Xlinker -install_name -Xlinker @rpath/libappleai.dylib \
  -Xlinker -rpath -Xlinker @loader_path \
  "${SRC}" \
  -o "${OUT_DIR}/libappleai.dylib"

echo "✅  Swift dylib built → ${OUT_DIR}/libappleai.dylib"
