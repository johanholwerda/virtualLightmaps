import LogGL from '../core/LogGL';
import {TextureOptions} from '../texture/Texture';

type CacheRegisterCallback = (data: any) => void;

let clientCache: { [path: string]: boolean } = {};
let clientCacheCallbacks: { [path: string]: Array<CacheRegisterCallback> | null } = {};
//the images loadImage is loading, returned to a second request of the same path
const loadingImages: Map<string, HTMLImageElement> = new Map();


function cacheIsPathRequested(path: string): any {
  return clientCache[path];
}

function cacheRegisterRequested(path: string): void {
  clientCache[path] = true;
}

function cacheRegisterCallback(path: string, callback: CacheRegisterCallback) {
  if (!clientCacheCallbacks[path]) {
    clientCacheCallbacks[path] = [];
  }
  (<Array<CacheRegisterCallback>>clientCacheCallbacks[path]).push(callback);
}

//resets the path before calling the callbacks, so a callback that loads the same path again starts a new request.
//a callback that throws is reported and doesn't keep the others from being called
function cacheRequestDone(path: string, data: any) {
  const cbs = clientCacheCallbacks[path];
  clientCacheCallbacks[path] = [];
  clientCache[path] = false;
  if (cbs) {
    for (const cb of cbs) {
      try {
        cb(data);
      } catch (error) {
        console.error(error);
      }
    }
  }
}

//the callbacks get null
function cacheRequestError(path: string, error: unknown) {
  LogGL.error('could not load', path, error);
  cacheRequestDone(path, null);
}


export function downloadFile(url: string, onProgress?: (progress: number) => void): Promise<ArrayBuffer> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('GET', url, true);
    xhr.responseType = 'arraybuffer';
    let progressCount = 0;

    xhr.onprogress = function (event: ProgressEvent) {
      if (event.lengthComputable && onProgress) {
        const progress = event.loaded / event.total;
        onProgress(progress);
      } else if (onProgress) {
        progressCount++;
        const progress = 1 - Math.pow(0.95, progressCount);
        onProgress(progress);
      }
    };

    xhr.onload = function () {
      if (xhr.status === 200) {
        onProgress && onProgress(1);
        resolve(xhr.response);
      } else {
        reject(new Error(`Error ${xhr.status}: ${xhr.statusText}`));
      }
    };

    xhr.onerror = function () {
      reject(new Error('Request failed'));
    };

    xhr.send();
  });
}

function loadFetch(path: string, callback: (callback: ArrayBuffer) => void, onProgress?: (progress: number) => void) {
  cacheRegisterCallback(path, callback);

  //an earlier request of the same path calls this callback too
  if (cacheIsPathRequested(path)) {
    return;
  }
  cacheRegisterRequested(path);

  //not .then().catch(): that would handle an error thrown by a callback as a failed request
  downloadFile(path, onProgress).then(
    (response) => cacheRequestDone(path, response),
    (error) => cacheRequestError(path, error),
  );
}

//the callback isn't called when the request failed (the error is logged): there is no text to decode
function loadFetchText(path: string, callback: (text: string) => void, onProgress?: (progress: number) => void) {
  loadFetch(path, (response: ArrayBuffer | null) => {
    if (response) callback(new TextDecoder().decode(response));
  }, onProgress);
}

export function loadText(urlOrText: string | Object | Promise<any>, callback: (text: string) => void, onProgress?: (progress: number) => void) {
  if (typeof urlOrText === 'string') {
    loadFetchText(urlOrText, callback, onProgress);
  } else if (typeof (<Promise<any>>urlOrText).then === 'function') {
    (<Promise<any>>urlOrText).then((data) => {
      callback(data);
    });
  } else {
    LogGL.warn('loadText: type not handled', typeof urlOrText, urlOrText);
    callback('error in loadText');
  }
}

export function loadJSON(urlOrJson: string | Object | Promise<any>, callback: (data: any) => void, onProgress?: (progress: number) => void) {
  if (typeof urlOrJson === 'string') {
    loadFetchText(urlOrJson, (text) => callback(JSON.parse(text)), onProgress);
  } else if (typeof (<Promise<any>>urlOrJson).then === 'function') {
    (<Promise<any>>urlOrJson).then((data) => {
      if (data?.default) {
        // if data.default is url, fetch
        if (typeof data.default === 'string') { // assume it is an url
          loadFetchText(data.default, (text) => callback(JSON.parse(text)), onProgress);
        } else {
          callback(data.default);
        }
      } else {
        callback(data);
      }
    });
  } else {
    callback(urlOrJson);
  }
}

export function loadArrayBuffer(path: string, callback: (data: ArrayBuffer) => void, onProgress?: (progress: number) => void) {
  loadFetch(path, callback, onProgress);
}


export function loadImage(path: string | {
  default: string
}, callback: (image: HTMLImageElement) => void): HTMLImageElement {
  if (typeof path !== 'string') {
    path = <string>path.default;
  }
  //apart from loadFetch's requests of the same path, which would give their callbacks this image
  const key = 'image:' + path;

  cacheRegisterCallback(key, callback);

  //the image of the request that is already loading, its load calls this callback too
  const loadingImage = cacheIsPathRequested(key) ? loadingImages.get(key) : undefined;
  if (loadingImage) {
    return loadingImage;
  }
  const image = new Image();

  cacheRegisterRequested(key);
  loadingImages.set(key, image);

  image.crossOrigin = 'Anonymous';

  image.onload = () => {
    loadingImages.delete(key);
    cacheRequestDone(key, image);
  };
  image.onerror = () => {
    LogGL.log('TextureLoader: could not load image', path);
    loadingImages.delete(key);
    cacheRequestDone(key, image);
  };

  image.src = path;

  return image;
}

export function loadImageBitmap(path: string | {
  default: string
}, callback: (image: ImageBitmap) => void, options: Partial<TextureOptions>): void {
  if (typeof createImageBitmap !== 'function') {
    LogGL.log('loadImageBitmap: createImageBitmap not available');
    return;
  }
  if (typeof path !== 'string') {
    path = <string>path.default;
  }
  fetch(<string>path)
    .then((response) => {
      return response.blob();
    })
    .then((blob) => {
      return createImageBitmap(blob, {
        premultiplyAlpha: 'none',
        colorSpaceConversion: 'none',
        imageOrientation: 'none',
      });
    })
    .then((imageBitmap) => {
      callback(imageBitmap);
    })
    .catch(() => {
      // @ts-ignore
      callback(null);
    });
}

export function loadMultipleImages(paths: string[], callback: (images: (HTMLImageElement | ImageBitmap)[]) => void, options: Partial<TextureOptions> = {}) {
  let imagesToLoad = paths.length;
  const images: (HTMLImageElement | ImageBitmap)[] = [];

  const receive = (i: number) => (image: HTMLImageElement | ImageBitmap) => {
    imagesToLoad--;
    images[i] = image;
    if (imagesToLoad === 0) {
      callback(images);
      if (loadSettings.loadImageBitmap) {
        for (let i = 0; i < images.length; ++i) {
          if (images[i]) {
            (<ImageBitmap>images[i]).close();
          }
        }
      }
    }
  };

  if (loadSettings.loadImageBitmap) {
    paths.forEach((path, i) => {
      loadImageBitmap(path, receive(i), options);
    });
  } else {
    paths.forEach((path, i) => {
      loadImage(path, receive(i));
    });
  }
}

export const loadSettings = {
  loadImageBitmap: false,
};
