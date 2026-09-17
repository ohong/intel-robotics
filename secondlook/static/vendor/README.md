# Vendored browser modules

Unmodified copies, served locally because the demo laptop may be offline. Same versions Physical AI Studio ships.

| Package | Version | License | Files |
| --- | --- | --- | --- |
| three | 0.184.0 (npm) | MIT, `three/LICENSE` | `build/three.module.min.js`, `build/three.core.min.js`, `examples/jsm/controls/OrbitControls.js`, `examples/jsm/loaders/STLLoader.js` |
| urdf-loader | 0.13.1 (npm) | Apache-2.0, `urdf-loader/LICENSE` | `src/URDFLoader.js`, `src/URDFClasses.js` |

`urdf-loader/collada-unsupported.js` is ours: the import map points urdf-loader's ColladaLoader import at it, because SO101 uses STL only.

The SO101 twin assets are not vendored here. Run `uv run scripts/sync_twin_assets.py` to fetch them, pinned and hash-checked, into `static/twin/SO101/`.
