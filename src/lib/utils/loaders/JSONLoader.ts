import IPreloadable from '../../renderer/core/IPreloadable';
import WebGPUPreLoader from '../../renderer/core/WebGPUPreLoader';
import {loadJSON} from '../../renderer/utils/LoadUtils';

export default class JSONLoader implements IPreloadable {
  public data: any;

  private urlOrJson: string | Object;

  constructor(preloader: WebGPUPreLoader | null, urlOrJson: string | Object | Promise<any>) {
    this.urlOrJson = urlOrJson;
    if (preloader !== null) {
      preloader.add(this);
    }
  }

  public load(callback: () => void, onProgress?: (progress: number) => void): void {
    loadJSON(this.urlOrJson, (data) => {
      this.data = data;
      callback();
    }, onProgress);
  }
}
