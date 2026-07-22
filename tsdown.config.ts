import { defineConfig } from "tsdown";

export default defineConfig({
  entry: "guest-js/index.ts",
  format: ["esm", "cjs"],
  // Generate .d.ts via Oxc's isolated-declarations transformer (Rust) instead of the
  // TypeScript Compiler API, so the build needs no `typescript` "." export. Required
  // because the pinned tsgo 7.x preview ships no JS Compiler API.
  dts: { oxc: true },
  outDir: "dist-js",
  // The bindings run in the Tauri webview; keep the bundle free of node shims.
  platform: "neutral",
  fixedExtension: true,
});
