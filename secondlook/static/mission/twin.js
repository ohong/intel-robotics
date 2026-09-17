// SO101 digital twin. Scene constants, camera, and model transform copy Physical AI
// Studio's robot viewer (robot-viewer-scene.tsx, controller/robot-viewer.tsx) so the
// arm looks the same here as in Studio. Joint values are Studio's units: degrees for
// revolute joints, converted with degToRad exactly as mapJointToURDFJoint does.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import URDFLoader from 'urdf-loader';

const URDF_PATH = '/static/twin/SO101/so101_new_calib.urdf';
const COLORS = {
  background: '#242528', ambientLight: '#c8d6eb', primaryLight: '#e8f0ff', fillLight: '#88aadd',
  gridCell: '#3a3d4f', gridSection: '#545870', checkerboardEven: '#2d2f35', checkerboardOdd: '#313339',
};
const FLOOR_SIZE = 21;
const CHECKERBOARD_TILE_SIZE = 0.5;
const GHOST = new THREE.MeshStandardMaterial({
  color: '#6fa8ff', emissive: '#2b5fb8', emissiveIntensity: .6, transparent: true, opacity: .26, depthWrite: false,
});
const IDLE_ORBIT_AFTER_MS = 15000;

function checkerboardFloor() {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 512;
  const context = canvas.getContext('2d');
  const tiles = 24, tileSize = canvas.width / tiles;
  for (let x = 0; x < tiles; x += 1) {
    for (let y = 0; y < tiles; y += 1) {
      context.fillStyle = (x + y) % 2 === 0 ? COLORS.checkerboardEven : COLORS.checkerboardOdd;
      context.fillRect(x * tileSize, y * tileSize, tileSize, tileSize);
    }
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.setScalar(FLOOR_SIZE / (tiles * CHECKERBOARD_TILE_SIZE));
  texture.colorSpace = THREE.SRGBColorSpace;
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(FLOOR_SIZE, FLOOR_SIZE),
                               new THREE.MeshStandardMaterial({map: texture, roughness: .8, metalness: 0}));
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = -.005;
  floor.receiveShadow = true;
  return floor;
}

// Port of drei's <Grid infiniteGrid> shader with Studio's sizes and colours.
function infiniteGrid() {
  const material = new THREE.ShaderMaterial({
    side: THREE.DoubleSide, transparent: true,
    uniforms: {
      cellSize: {value: .25}, sectionSize: {value: .5}, cellThickness: {value: .5}, sectionThickness: {value: 1},
      cellColor: {value: new THREE.Color(COLORS.gridCell)}, sectionColor: {value: new THREE.Color(COLORS.gridSection)},
      fadeDistance: {value: FLOOR_SIZE - 1}, fadeStrength: {value: 1},
    },
    vertexShader: `
      uniform float fadeDistance;
      varying vec3 worldPosition;
      void main() {
        worldPosition = position.xzy * (1. + fadeDistance);
        worldPosition.xz += cameraPosition.xz;
        gl_Position = projectionMatrix * viewMatrix * vec4(worldPosition, 1.);
      }`,
    fragmentShader: `
      uniform float cellSize, sectionSize, cellThickness, sectionThickness, fadeDistance, fadeStrength;
      uniform vec3 cellColor, sectionColor;
      varying vec3 worldPosition;
      float getGrid(float size, float thickness) {
        vec2 r = worldPosition.xz / size;
        vec2 grid = abs(fract(r - .5) - .5) / fwidth(r);
        float line = min(grid.x, grid.y) + 1. - thickness;
        return 1. - min(line, 1.);
      }
      void main() {
        float g1 = getGrid(cellSize, cellThickness);
        float g2 = getGrid(sectionSize, sectionThickness);
        float d = 1. - min(distance(cameraPosition.xz, worldPosition.xz) / fadeDistance, 1.);
        vec3 color = mix(cellColor, sectionColor, min(1., sectionThickness * g2));
        float alpha = (g1 + g2) * pow(d, fadeStrength);
        alpha = mix(.75 * alpha, alpha, g2);
        if (alpha <= 0.) discard;
        gl_FragColor = vec4(color, alpha);
        #include <colorspace_fragment>
      }`,
  });
  const grid = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), material);
  grid.frustumCulled = false;
  return grid;
}

// Soft blob standing in for drei's <ContactShadows opacity={0.2} scale={2.5}>.
function contactShadow() {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 128;
  const context = canvas.getContext('2d');
  const gradient = context.createRadialGradient(64, 64, 0, 64, 64, 64);
  gradient.addColorStop(0, 'rgba(0,0,0,1)');
  gradient.addColorStop(1, 'rgba(0,0,0,0)');
  context.fillStyle = gradient;
  context.fillRect(0, 0, 128, 128);
  const shadow = new THREE.Mesh(new THREE.PlaneGeometry(1.2, 1.2), new THREE.MeshBasicMaterial({
    map: new THREE.CanvasTexture(canvas), transparent: true, opacity: .2, depthWrite: false,
  }));
  shadow.rotation.x = -Math.PI / 2;
  shadow.position.y = .001;
  return shadow;
}

function loadRobot() {
  return new Promise((resolve, reject) => {
    const manager = new THREE.LoadingManager();
    const loader = new URDFLoader(manager);
    let robot = null;
    manager.onLoad = () => robot ? resolve(robot) : reject(new Error('URDF loaded without a robot'));
    manager.onError = url => reject(new Error(`Failed to load ${url}. Run scripts/sync_twin_assets.py.`));
    loader.load(URDF_PATH, result => { robot = result; }, undefined, reject);
  });
}

// Studio: outer group rotated -45° about Y; inner group [-π/2, 0, -π/4], scale 3.
function placed(robot) {
  const inner = new THREE.Group();
  inner.rotation.set(-Math.PI / 2, 0, -Math.PI / 4);
  inner.scale.setScalar(3);
  inner.add(robot);
  const outer = new THREE.Group();
  outer.rotation.y = THREE.MathUtils.degToRad(-45);
  outer.add(inner);
  return outer;
}

function applyJoints(robot, jointNames, values) {
  jointNames.forEach((name, index) => {
    const joint = robot.joints[name];
    if (!joint) return;
    robot.setJointValue(name, joint.jointType === 'revolute' ? THREE.MathUtils.degToRad(values[index]) : values[index]);
  });
}

export async function createTwin(container) {
  const renderer = new THREE.WebGLRenderer({antialias: true});
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  container.replaceChildren(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(COLORS.background);
  scene.add(new THREE.AmbientLight(COLORS.ambientLight, .7));
  const key = new THREE.DirectionalLight(COLORS.primaryLight, 2.5);
  key.position.set(1.5, 3.5, 2);
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  Object.assign(key.shadow.camera, {left: -6, right: 6, top: 6, bottom: -6, near: .2, far: 40});
  key.shadow.camera.updateProjectionMatrix();
  key.shadow.bias = -.0001;
  const fill = new THREE.DirectionalLight(COLORS.fillLight, .4);
  fill.position.set(2, 2, -3);
  scene.add(key, fill, checkerboardFloor(), infiniteGrid(), contactShadow());

  const camera = new THREE.PerspectiveCamera(75, 1, .1, 1000);  // drei PerspectiveCamera defaults
  camera.position.set(2, 1, 1);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.autoRotateSpeed = .35;
  let lastInteraction = performance.now();
  controls.addEventListener('start', () => { lastInteraction = performance.now(); controls.autoRotate = false; });
  controls.addEventListener('end', () => { lastInteraction = performance.now(); });

  const robot = await loadRobot();
  robot.traverse(node => { if (node.isMesh) node.castShadow = node.receiveShadow = true; });
  const ghost = robot.clone();
  ghost.traverse(node => { if (node.isMesh) { node.material = GHOST; node.castShadow = false; node.renderOrder = 2; } });
  ghost.visible = false;
  scene.add(placed(robot), placed(ghost));

  const resize = () => {
    const {clientWidth: width, clientHeight: height} = container;
    if (!width || !height) return;
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  };
  new ResizeObserver(resize).observe(container);
  resize();

  renderer.setAnimationLoop(() => {
    if (!controls.autoRotate && performance.now() - lastInteraction > IDLE_ORBIT_AFTER_MS) controls.autoRotate = true;
    controls.update();
    renderer.render(scene, camera);
  });

  return {
    limitsDeg(name) {
      const limit = robot.joints[name]?.limit;
      return limit ? [THREE.MathUtils.radToDeg(limit.lower), THREE.MathUtils.radToDeg(limit.upper)] : null;
    },
    setPose(jointNames, values) { applyJoints(robot, jointNames, values); },
    setGhost(jointNames, values) {
      ghost.visible = Boolean(values);
      if (values) applyJoints(ghost, jointNames, values);
    },
  };
}
