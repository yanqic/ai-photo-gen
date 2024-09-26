const sharp = require("sharp");
const ort = require("onnxruntime-web");

// Helper function to get scale factor
function getScaleFactor(im_h, im_w, ref_size) {
  let im_rh, im_rw;
  const aspectRatio = im_w / im_h;

  if (Math.max(im_h, im_w) < ref_size) {
    im_rh = ref_size;
    im_rw = Math.floor(ref_size * aspectRatio);
  } else if (Math.min(im_h, im_w) > ref_size) {
    im_rw = ref_size;
    im_rh = Math.floor(ref_size / aspectRatio);
  } else {
    im_rh = im_h;
    im_rw = im_w;
  }

  im_rw -= im_rw % 32;
  im_rh -= im_rh % 32;

  return { x_scale_factor: im_rw / im_w, y_scale_factor: im_rh / im_h };
}

// Main Inference part
async function mainInference(imagePath, modelPath, outputPath, ref_size) {
  const image = await sharp(imagePath).raw().toBuffer({ resolveWithObject: true });
  const { info: { width: im_w, height: im_h } } = image;

  const { x_scale_factor, y_scale_factor } = getScaleFactor(im_h, im_w, ref_size);
  const resizedWidth = Math.floor(im_w * x_scale_factor);
  const resizedHeight = Math.floor(im_h * y_scale_factor);

  const resizedImage = await sharp(image.data, {
    raw: { width: im_w, height: im_h, channels: 3 }
  })
    .resize(resizedWidth, resizedHeight, {
      kernel: sharp.kernel.lanczos2,
      fit: "fill",
    })
    .raw()
    .toBuffer();

  const inputTensor = new Float32Array(1 * 3 * resizedHeight * resizedWidth);
  let offset = 0;
  for (let i = 0; i < resizedImage.length; i += 3) {
    inputTensor[offset] = (resizedImage[i] - 127.5) / 127.5;
    inputTensor[offset + resizedHeight * resizedWidth] = (resizedImage[i + 1] - 127.5) / 127.5;
    inputTensor[offset + 2 * resizedHeight * resizedWidth] = (resizedImage[i + 2] - 127.5) / 127.5;
    offset++;
  }

  const session = await ort.InferenceSession.create(modelPath);
  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];

  const feeds = {};
  feeds[inputName] = new ort.Tensor("float32", inputTensor, [1, 3, resizedHeight, resizedWidth]);

  const results = await session.run(feeds);
  const matte = Float32Array.from(results[outputName].data, v => v * 255);

  const mask = await sharp(Buffer.from(matte), {
    raw: { width: resizedWidth, height: resizedHeight, channels: 1 }
  })
    .resize(im_w, im_h)
    .extractChannel(0)
    .toBuffer();

  await sharp(image.data, {
    raw: { width: im_w, height: im_h, channels: 3 }
  })
    .joinChannel(mask, { raw: { channels: 1, width: im_w, height: im_h } })
    .png()
    .toFile(outputPath);
}

// Example usage:
mainInference("./script/demo.jpg", "./public/models/modnet.onnx", "./script/output.png", 512)
  .then(() => console.log("Inference completed and output saved."))
  .catch(err => console.error("Error during inference:", err));
