import type Renderer from '../../renderer/render/Renderer';
import Vector2 from '../../renderer/math/Vector2';

type DrawableImage = HTMLVideoElement | HTMLImageElement | HTMLCanvasElement | ImageBitmap;

export function createCanvas(width: number, height: number): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

//a new canvas of the size of the image, with the image drawn into it, optionally mirrored
function drawToCanvas(image: DrawableImage, flipX: boolean = false, flipY: boolean = false): HTMLCanvasElement {
  const canvas = createCanvas(image.width, image.height);
  const context = <CanvasRenderingContext2D>canvas.getContext('2d');
  context.transform(flipX ? -1 : 1, 0, 0, flipY ? -1 : 1, flipX ? canvas.width : 0, flipY ? canvas.height : 0);
  context.drawImage(image, 0, 0);
  return canvas;
}

export function canvasToCanvas(source: HTMLCanvasElement | HTMLImageElement, destination: HTMLCanvasElement): HTMLCanvasElement {
  const ctx = <CanvasRenderingContext2D>destination.getContext('2d');
  ctx.drawImage(source, 0, 0);
  return destination;
}

export function imageToCanvas(source: HTMLCanvasElement | HTMLImageElement): HTMLCanvasElement {
  return drawToCanvas(source);
}

export function flipImageX(image: HTMLVideoElement | HTMLImageElement | HTMLCanvasElement): HTMLCanvasElement {
  return drawToCanvas(image, true, false);
}

export function flipImageY(image: DrawableImage): HTMLCanvasElement {
  return drawToCanvas(image, false, true);
}

export function getImageData(source: HTMLCanvasElement | HTMLImageElement, flipped: boolean = false): ImageData {
  const canvas = drawToCanvas(source, false, flipped);
  return (<CanvasRenderingContext2D>canvas.getContext('2d')).getImageData(0, 0, source.width, source.height);
}

//the intrinsic size of an image source: .width/.height are layout attributes and not reliable for video and image elements
export function getImageSize(image: GPUCopyExternalImageSource): Vector2 {
  if (typeof HTMLVideoElement != 'undefined' && image instanceof HTMLVideoElement) return new Vector2(image.videoWidth, image.videoHeight);
  if (typeof HTMLImageElement != 'undefined' && image instanceof HTMLImageElement) return new Vector2(image.naturalWidth, image.naturalHeight);
  if (typeof VideoFrame != 'undefined' && image instanceof VideoFrame) return new Vector2(image.displayWidth, image.displayHeight);
  const sized = <ImageBitmap | ImageData | HTMLCanvasElement | OffscreenCanvas>image;
  return new Vector2(sized.width, sized.height);
}

export function resizeImage(image: DrawableImage, width: number, height: number): DrawableImage {
  const size = getImageSize(image);
  if (width === size.x && height === size.y) {
    return image;
  }
  const canvas = createCanvas(width, height);
  const context = <CanvasRenderingContext2D>canvas.getContext('2d');
  context.drawImage(image, 0, 0, size.x, size.y, 0, 0, width, height);
  return canvas;
}

export function createDummyImage(width: number, height: number, text: string = ''): HTMLCanvasElement {
  const canvas = createCanvas(width, height);
  const ctx = <CanvasRenderingContext2D>canvas.getContext('2d');
  ctx.fillStyle = '#eb7d24';
  ctx.fillRect(0, 0, width, height);

  ctx.beginPath();
  for (let x = 0; x < width; x += 64) {
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
  }
  for (let y = 0; y < height; y += 64) {
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
  }
  ctx.strokeStyle = '#ffffff';
  ctx.stroke();

  ctx.font = '30px Arial';
  ctx.fillStyle = '#ffffff';
  ctx.fillText(width + 'x' + height, 8, 32);
  if (text.length) {
    ctx.fillText(text, Math.max(0, width / 2 - 8 * text.length), height / 2 + 16);
  }
  return canvas;
}

export function dataToCanvas(data: Uint8Array, width: number, height: number, flipY: boolean = false): HTMLCanvasElement {
  const canvas = createCanvas(width, height);
  const ctx = <CanvasRenderingContext2D>canvas.getContext('2d');

  const imgData: ImageData = ctx.createImageData(width, height);

  for (let y = 0; y < height; y++) {
    let imgOffset = y * width * 4;
    let dataOffset = flipY ? y * width * 4 : (height - y - 1) * width * 4;
    for (let x = 0; x < width; x++) {
      imgData.data[imgOffset++] = data[dataOffset++];
      imgData.data[imgOffset++] = data[dataOffset++];
      imgData.data[imgOffset++] = data[dataOffset++];
      imgData.data[imgOffset++] = data[dataOffset++];
    }
  }

  ctx.putImageData(imgData, 0, 0);
  return canvas;
}

export function saveDataToDisk(
  data: Uint8Array,
  width: number,
  height: number,
  name: string     = 'image',
  fileType: string = 'png',
  quality: number  = 1,
) {
  let canvas = dataToCanvas(data, width, height);
  saveCanvasToDisk(canvas, name, fileType, quality);
}

export async function saveTextureToDisk(renderer: Renderer, texture: GPUTexture , name: string = 'image', fileType: string = 'png', quality: number = 1) {
  const data = await getFromTextureData(renderer, texture);
  let canvas = dataToCanvas(<Uint8Array>data, texture.width, texture.height, true);
  saveCanvasToDisk(canvas, name, fileType, quality);
}

//the pixels of an rgba8unorm texture without row padding. finishes the current command encoder, the texture needs
//GPUTextureUsage.COPY_SRC
export async function getFromTextureData(renderer:Renderer, texture:GPUTexture): Promise<Uint8Array> {
  if(texture.format != 'rgba8unorm'){
    console.warn('getFromTextureData: only supports rgba8unorm');
    return new Uint8Array(texture.width * texture.height * 4);
  }

  // Round up bytesPerRow to nearest 256
  const bytesPerRow = Math.ceil((texture.width * 4) / 256) * 256;
  const bufferSize = bytesPerRow * texture.height;

  const outputBuffer = renderer.createBuffer(bufferSize, GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);

  renderer.endPasses();
  renderer.commandEncoder.copyTextureToBuffer(
    {
      texture: texture,
    },
    {
      buffer: outputBuffer,
      bytesPerRow: bytesPerRow,
    },
    {
      width: texture.width,
      height: texture.height,
    }
  );
  //readBuffer copies it in the same command encoder and resolves once that is submitted
  const read = renderer.readBuffer(outputBuffer, Uint8Array);
  renderer.finish();
  const paddedData = await read;
  outputBuffer.destroy();

  // Remove padding
  const actualData = new Uint8Array(texture.width * texture.height * 4);
  for (let y = 0; y < texture.height; y++) {
    const paddedOffset = y * bytesPerRow;
    const actualOffset = y * texture.width * 4;
    actualData.set(
      paddedData.subarray(paddedOffset, paddedOffset + texture.width * 4),
      actualOffset
    );
  }

  return actualData;
}

export function saveCanvasToDisk(canvas: HTMLCanvasElement, name: string = 'image', fileType: string = 'png', quality: number = 1) {
  canvas.toBlob(
    (blob) => {
      if (blob) {
        let URLObj = window.URL;
        let a = document.createElement('a');
        a.href = URLObj.createObjectURL(blob);
        a.download = name;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
      }
    },
    'image/' + fileType,
    quality,
  );
}
