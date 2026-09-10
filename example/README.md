# ncnn example

Loads an ultralytics-exported ncnn model (`.param` + `.bin`) from disk
and runs a single classify pass on a zeros frame at the exported
`imgsz`. Point the two text fields at your `model.ncnn.param` /
`model.ncnn.bin` (keep `metadata.yaml` next to them for class names and
`imgsz`), then press Load and Run.

```sh
flutter pub get
flutter run -d linux   # or windows / macos / android / ios
```
