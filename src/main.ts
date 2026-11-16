import WebGPUPreLoader from 'webGPU-framework/renderer/core/WebGPUPreLoader';
import LogGL from 'webGPU-framework/renderer/core/LogGL';
import KeyListener from 'webGPU-framework/utils/input/KeyListener';
import {KEY_P} from 'webGPU-framework/utils/input/Key';
import {saveCanvasToDisk} from 'webGPU-framework/utils/image/ImageUtils';
import UncappedFrames from 'webGPU-framework/utils/UncappedFrames';
import {createDescriptionOverlay} from './DescriptionOverlay';
import IrradianceCache from './irradianceCache/IrradianceCache';
import './main.css';

// Always installed, so the checkbox next to the fps counter can turn it on; ?uncapped starts with it on.
UncappedFrames.install(new URLSearchParams(window.location.search).has('uncapped'));

const containerDiv = document.getElementById('webgl-preloader-container');
const progressDiv = document.getElementById('webgl-preloader-progress');

function handlePreloadProgress(progress: number) {
  if (!containerDiv || !progressDiv) {
    throw new ReferenceError('Unable to find container divs in html');
  }

  containerDiv.style.display = progress < 1 ? 'block' : 'none';
  progressDiv.style.width = ((276 * progress) | 0) + 'px';
}

async function main() {
  handlePreloadProgress(0);

  const preloader = new WebGPUPreLoader();

  const canvasContainer = document.createElement('div');
  canvasContainer.classList.add('webgpu-container');
  document.body.appendChild(canvasContainer);

  LogGL.ENABLED = true;
  const irradianceCache = new IrradianceCache();

  createDescriptionOverlay(IrradianceCache, canvasContainer);

  await irradianceCache.start(canvasContainer, preloader, true);

  preloader.load(
    () => {
      handlePreloadProgress(1);
      irradianceCache.init();
    },
    (progress) => handlePreloadProgress(progress),
  );

  KeyListener.addKeyHitEvent(KEY_P, () => {
    saveCanvasToDisk(document.querySelector('canvas')!, 'virtuallightmap.png');
  });
}

main();
