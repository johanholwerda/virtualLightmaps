import IDestructible from './IDestructible';
import LogGL from './LogGL';
import {createCanvas} from '../../utils/image/ImageUtils';

export default class CanvasManager implements IDestructible {
  public canvas!: HTMLCanvasElement;
  public pixelRatio: number = 1;
  //the size of the canvas in pixels instead of the size of the container, like for exporting. css still stretches it
  public fixedSize: {width: number, height: number} | null = null;
  protected readonly container: HTMLElement | null;
  private resizeObserver: ResizeObserver | null = null;
  //no pixel ratio was passed: use the one of the screen, also when it changes
  private followDevicePixelRatio: boolean = false;
  //the size of the container from the resize observer, -1 until its first callback
  private observedWidth: number = -1;
  private observedHeight: number = -1;

  public constructor(container: HTMLElement | null, pixelRatio: number = -1) {
    this.container = container;
    if (container) {
      if (pixelRatio > 0) {
        this.pixelRatio = pixelRatio;
      }else {
        this.followDevicePixelRatio = true;
        this.pixelRatio = window.devicePixelRatio;
      }

      //at the size update() gives it, so the first frame doesn't resize it (and every render target on screen)
      const rect = container.getBoundingClientRect();
      this.canvas = createCanvas((rect.width * this.pixelRatio) | 0, (rect.height * this.pixelRatio) | 0);
      Object.assign(this.canvas.style, {
        display: 'block',
        width: '100%',
        height: '100%',
        inset: '0'
      });

      if (rect.width * rect.height < 1) LogGL.error('canvasParent has no area', container);
      container.appendChild(this.canvas);

      //reading the size in update() with getBoundingClientRect every frame can force a layout, the observer gets it
      //when it changes. without ResizeObserver (or before its first callback) update() reads it itself
      if (typeof ResizeObserver !== 'undefined') {
        this.resizeObserver = new ResizeObserver((entries) => {
          const entry = entries[entries.length - 1];
          //the border box, like getBoundingClientRect
          const box = entry.borderBoxSize?.[0];
          this.observedWidth = box ? box.inlineSize : entry.contentRect.width;
          this.observedHeight = box ? box.blockSize : entry.contentRect.height;
        });
        this.resizeObserver.observe(container);
      }
    } else {
      LogGL.error('canvasParent is null');
    }
  }

  public update(): boolean {
    if (this.fixedSize) return this.setPixelSize(this.fixedSize.width, this.fixedSize.height);

    //browser zoom and moving the window to another screen change it
    if (this.followDevicePixelRatio) this.pixelRatio = window.devicePixelRatio;

    if (this.container) {
      let w = this.observedWidth;
      let h = this.observedHeight;
      if (w < 0) {
        const boundingRect = this.container.getBoundingClientRect();
        w = boundingRect.width;
        h = boundingRect.height;
      }
      if (w * h > 0) {
        return this.setSize(w, h);
      } else {
        LogGL.warn('canvasParent has no area', this.container);
      }
    }
    return false;
  }

  public setSize(w: number, h: number): boolean {
    return this.setPixelSize((w * this.pixelRatio) | 0, (h * this.pixelRatio) | 0);
  }

  private setPixelSize(width: number, height: number): boolean {
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;

      this.canvas.dispatchEvent(new Event('resize'));
      return true;
    } else {
      return false;
    }
  }

  destruct() {
    this.resizeObserver?.disconnect();
    this.container?.removeChild(this.canvas);
  }
}
