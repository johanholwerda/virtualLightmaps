import IPreloadable from './IPreloadable';
import IDestructible from './IDestructible';
import {clamp01} from '../utils/MathUtils';

const NOT_LOADED = -1;
const LOADING = 0;
const LOADED = 1;

type Callback = (() => void) | undefined;
type ProgressCallback = ((percentage: number) => void) | undefined;
type ResourceRequest = { resource: IPreloadable; progress: number; callbacks: Callback[] };

export default class WebGPUPreLoader implements IDestructible {
  private progress: number = 0;

  private requests: ResourceRequest[] = [];
  private onComplete: Callback;
  private onProgress: ProgressCallback;
  private loading: boolean = false;

  public add<T extends IPreloadable>(resource: T, callback?: Callback): T {
    const data = {
      resource,
      progress:    NOT_LOADED,
      callbacks: [callback],
    };

    this.requests.push(data);

    if (this.loading) {
      // add preloadable while preloader is loading, so directly start
      // this download
      this.loadResource(data);
    }
    return resource;
  }

  public load(onComplete?: Callback, onProgress?: ProgressCallback): void {
    if (this.loading) {
      console.error('preloader already started');
      return;
    }
    this.onComplete = onComplete;
    this.onProgress = onProgress;
    this.loading = true;

    this.requests.forEach((data) => {
      this.loadResource(data);
    });

    if (this.requests.length === 0) {
      if (this.onComplete) {
        this.onComplete();
        this.onComplete = undefined;
      }
    }
  }

  public addOnCompleteForResource(preloadable: IPreloadable, callback?: Callback) {
    this.requests.find((a) => a.resource === preloadable)?.callbacks.push(callback);
  }

  destruct() {
    this.onComplete = undefined;
    this.requests = [];
  }

  private loadResource(data: ResourceRequest) {
    if (data.progress === NOT_LOADED) {
      data.progress = LOADING;
      data.resource.load(() => {
        this.loadDoneCallback(data);
      }, (progress) => {
        data.progress = progress * 0.99;
        this.updateProgress();
      } );
    }
  }

  private loadDoneCallback(data: ResourceRequest) {
    if (data.progress === LOADED) {
      return;
    }

    data.progress = LOADED;
    data.callbacks.forEach((callback) => {
      if (callback) {
        callback();
      }
    });

    this.updateProgress();
  }

  private updateProgress() {
    const loadsCompleted = this.requests.reduce((a, b) => a + Math.max(0, b.progress), 0);

    // (c) Windows load progress
    // this.progress += (1 - this.progress) * Math.pow(1 / (1 + this.requests.length - loadsCompleted), 1.5);
    this.progress = clamp01(loadsCompleted / this.requests.length);

    if (this.onProgress) {
      if (this.requests.length) {
        this.onProgress(clamp01(this.progress));
      } else {
        this.onProgress(1);
      }
    }

    // check if all requested loads are loaded
    if (loadsCompleted >= this.requests.length) {
      if (this.onComplete) {
        this.onComplete();
        this.onComplete = undefined;
      }
    }
  }
}
