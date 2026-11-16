/*
the pixels of an 8 bits RGB or RGBA png (not interlaced) as RGBA, 4 bytes per pixel, exactly as stored. For data in the
channels of a png: a canvas premultiplies the alpha, which loses the RGB where it is low, and createImageBitmap can convert
the color space. Without imports, so it runs in node too (DecompressionStream is global there)
 */
export default async function decodePngRgba8(png: ArrayBuffer): Promise<{width: number, height: number, data: Uint8Array}> {
  const bytes = new Uint8Array(png);
  const view = new DataView(png);
  const signature = [137, 80, 78, 71, 13, 10, 26, 10];
  if (signature.some((b, i) => bytes[i] != b)) throw new Error('decodePngRgba8: not a png');

  let width = 0, height = 0, colorType = 0;
  const idat: Uint8Array[] = [];
  for (let offset = 8; offset < bytes.length;) {
    const length = view.getUint32(offset);
    const type = String.fromCharCode(...bytes.subarray(offset + 4, offset + 8));
    const start = offset + 8;
    if (type == 'IHDR') {
      width = view.getUint32(start);
      height = view.getUint32(start + 4);
      const bitDepth = bytes[start + 8];
      colorType = bytes[start + 9];
      const interlace = bytes[start + 12];
      if (bitDepth != 8 || (colorType != 2 && colorType != 6) || interlace != 0) {
        throw new Error(`decodePngRgba8: only 8 bits RGB or RGBA without interlace, not depth ${bitDepth} type ${colorType} interlace ${interlace}`);
      }
    } else if (type == 'IDAT') {
      idat.push(bytes.subarray(start, start + length));
    } else if (type == 'IEND') {
      break;
    }
    offset = start + length + 4;
  }

  //the IDAT data is a zlib stream, which is 'deflate' for DecompressionStream
  const stream = new Blob(idat as BlobPart[]).stream().pipeThrough(new DecompressionStream('deflate'));
  const raw = new Uint8Array(await new Response(stream).arrayBuffer());

  const channels = colorType == 6 ? 4 : 3;
  const stride = width * channels;
  const pixels = new Uint8Array(stride * height);
  for (let y = 0; y < height; y++) {
    const filter = raw[y * (stride + 1)];
    const src = y * (stride + 1) + 1;
    const row = y * stride;
    const prior = row - stride;
    for (let x = 0; x < stride; x++) {
      const a = x >= channels ? pixels[row + x - channels] : 0;
      const b = y > 0 ? pixels[prior + x] : 0;
      const c = x >= channels && y > 0 ? pixels[prior + x - channels] : 0;
      let predictor = 0;
      switch (filter) {
        case 0: predictor = 0; break;
        case 1: predictor = a; break;
        case 2: predictor = b; break;
        case 3: predictor = (a + b) >> 1; break;
        case 4: {
          const p = a + b - c;
          const pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
          predictor = pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
          break;
        }
        default: throw new Error(`decodePngRgba8: unknown filter ${filter}`);
      }
      pixels[row + x] = (raw[src + x] + predictor) & 0xff;
    }
  }

  if (channels == 4) return {width, height, data: pixels};
  const data = new Uint8Array(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    data[i * 4] = pixels[i * 3];
    data[i * 4 + 1] = pixels[i * 3 + 1];
    data[i * 4 + 2] = pixels[i * 3 + 2];
    data[i * 4 + 3] = 255;
  }
  return {width, height, data};
}
