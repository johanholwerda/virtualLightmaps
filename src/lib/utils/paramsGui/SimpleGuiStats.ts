// based on Stats by mrdoob https://github.com/mrdoob/stats.js/blob/master/src/Stats.js

import {createCanvas} from "../image/ImageUtils";
import {clamp01} from "../../renderer/utils/MathUtils";
import Profiler from "../../renderer/render/Profiler";
import UncappedFrames from "../UncappedFrames";

export function guiCreateStats(parent: HTMLElement, name = 'FPS'): HTMLDivElement {
  let min = Infinity, max = 0, round = Math.round;
  const PR = round(window.devicePixelRatio || 1);

  const HEIGHT = 26 * PR,
        TEXT_X = 0, TEXT_Y = 0,
        GRAPH_X = 0, GRAPH_Y = 12 * PR,
        GRAPH_HEIGHT = 14 * PR, COLUMN = PR * 2;

  // The canvas stretches to fill the row, so the backing store follows its laid out size to stay pixel crisp.
  let WIDTH = 260 * PR, GRAPH_WIDTH = WIDTH;

  const canvas = createCanvas(WIDTH, HEIGHT);
  canvas.className = 'gui-canvas';

  const context = canvas.getContext('2d') as CanvasRenderingContext2D;

  const setContextStyle = () => {
    context.font = `bold ${11 * PR}px sans-serif`;
    context.textBaseline = 'top';
  };
  setContextStyle();

  const updateContext = async (value: number, maxValue: number) => {
    min = Math.min(min, value);
    max = Math.max(max, value);
    context.clearRect(0, 0, WIDTH, GRAPH_Y);

    if ('memory' in window.performance) {
      context.fillStyle = 'lightgray';
      context.textAlign = 'right';
      type Memory = { jsHeapSizeLimit: number, totalJSHeapSize: number, usedJSHeapSize: number };
      let mem = window.performance.memory as Memory;
      let mb = 1048576;
      context.fillText(`${mem.usedJSHeapSize / mb | 0}/${mem.jsHeapSizeLimit / mb | 0} MB`, WIDTH, TEXT_Y);
      context.textAlign = 'left';
    }

    context.fillStyle = value < 58 ? 'pink' : 'lightgreen';
    context.fillText(`${round(value)} ${name} (${round(min)}-${round(max)})`, TEXT_X, TEXT_Y);

    createImageBitmap(canvas).then(ibm => {
      context.clearRect(GRAPH_X, GRAPH_Y, GRAPH_WIDTH, GRAPH_HEIGHT);
      context.drawImage(ibm, GRAPH_X + COLUMN, GRAPH_Y, GRAPH_WIDTH - COLUMN, GRAPH_HEIGHT, GRAPH_X, GRAPH_Y, GRAPH_WIDTH - COLUMN, GRAPH_HEIGHT);
      ibm.close();

      //up to the highest value seen, so a screen above maxValue hz doesn't fill every bar
      const height = round(clamp01(value / Math.max(maxValue, max)) * GRAPH_HEIGHT);
      context.fillRect(GRAPH_X + GRAPH_WIDTH - COLUMN, GRAPH_Y + GRAPH_HEIGHT - height, COLUMN, height);
    });
  };

  updateContext(0, 100);
  let beginTime = (performance || Date).now(), prevTime = beginTime, frames = 0;

  function end() {
    frames++;

    let time = (performance || Date).now();

    if (time >= prevTime + 1000) {
      updateContext((frames * 1000) / (time - prevTime), 100);

      prevTime = time;
      frames = 0;
    }

    return time;
  }

  const row = document.createElement('div');
  row.className = 'gui-stats-row';
  row.appendChild(canvas);
  parent.appendChild(row);

  const resizeCanvas = () => {
    const width = round(canvas.clientWidth * PR);
    if (width <= 0 || width === WIDTH) return;
    WIDTH = GRAPH_WIDTH = width;
    canvas.width = WIDTH; // clears the canvas and resets the 2d context state
    setContextStyle();
  };

  new ResizeObserver(resizeCanvas).observe(canvas);

  function update() {
    if (!canvas.isConnected) {
      return;
    }
    beginTime = end();

    requestAnimationFrame(update);
  }

  requestAnimationFrame(update);

  return row;
}

export function guiCreateProfilerToggle(parent: HTMLElement, name = 'profiler') {
  const checkbox = document.createElement('input');
  checkbox.id = 'input-profiler';
  checkbox.type = 'checkbox';
  checkbox.title = name;
  checkbox.checked = Profiler.isEnabled();
  checkbox.addEventListener('change', () => Profiler.setEnabled(checkbox.checked));

  parent.appendChild(checkbox);
}

//only shown when UncappedFrames is installed (the sandbox bootstrap does that)
export function guiCreateUncappedToggle(parent: HTMLElement, name = 'uncapped: frames not tied to vsync') {
  if (!UncappedFrames.isInstalled()) return;

  const label = document.createElement('span');
  label.className = 'gui-stats-toggle';
  label.title = name;
  label.textContent = 'uncapped';

  const checkbox = document.createElement('input');
  checkbox.id = 'input-uncapped';
  checkbox.type = 'checkbox';
  checkbox.title = name;
  checkbox.checked = UncappedFrames.isEnabled();
  checkbox.addEventListener('change', () => UncappedFrames.setEnabled(checkbox.checked));

  label.addEventListener('click', () => checkbox.click());
  parent.appendChild(label);
  parent.appendChild(checkbox);
}
