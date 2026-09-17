// Stand-in for three's ColladaLoader, which urdf-loader imports. The SO101 URDF
// uses STL meshes only, so a .dae mesh is an error rather than a silent skip.
export class ColladaLoader {
  constructor() {}
  load(url) { throw new Error(`Collada meshes are not vendored: ${url}`); }
}
