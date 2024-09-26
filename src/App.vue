<template>
    <div class="min-h-screen bg-gray-100 p-6">
        <div class="max-w-6xl mx-auto">
            <h1 class="text-3xl text-center font-bold mb-6">Photo Generator</h1>
            <div class="flex bg-white">
                <!-- 左侧布局 -->
                <div class="w-1/2 p-4">
                    <h5 class="text-2xl font-bold mb-4">Upload Image</h5>
                    <!-- 上传图片部分 -->
                    <div
                        v-if="!imageUrl"
                        class="relative w-full h-[480px] border-2 border-dashed border-gray-300 rounded-xl overflow-hidden mb-4">
                        <label
                            id="upload-button"
                            for="upload"
                            class="absolute inset-0 flex flex-col items-center justify-center cursor-pointer">
                            <svg
                                width="25"
                                height="25"
                                viewBox="0 0 25 25"
                                fill="none"
                                xmlns="http://www.w3.org/2000/svg">
                                <path
                                    fill="#000"
                                    d="M3.5 24.3a3 3 0 0 1-1.9-.8c-.5-.5-.8-1.2-.8-1.9V2.9c0-.7.3-1.3.8-1.9.6-.5 1.2-.7 2-.7h18.6c.7 0 1.3.2 1.9.7.5.6.7 1.2.7 2v18.6c0 .7-.2 1.4-.7 1.9a3 3 0 0 1-2 .8H3.6Zm0-2.7h18.7V2.9H3.5v18.7Zm2.7-2.7h13.3c.3 0 .5 0 .6-.3v-.7l-3.7-5a.6.6 0 0 0-.6-.2c-.2 0-.4 0-.5.3l-3.5 4.6-2.4-3.3a.6.6 0 0 0-.6-.3c-.2 0-.4.1-.5.3l-2.7 3.6c-.1.2-.2.4 0 .7.1.2.3.3.6.3Z"></path>
                            </svg>
                            <span class="mt-2">Click to upload image</span>
                        </label>
                    </div>
                    <!-- 图片预览部分 -->
                    <div
                        v-if="imageUrl"
                        class="relative w-full flex justify-center items-center h-[480px] border-2 border-gray-300 rounded-xl overflow-hidden mb-4">
                        <img :src="imageUrl" class="h-full object-contain" />
                        <button
                            @click="replaceImage"
                            class="absolute top-2 right-2 flex items-center justify-center w-6 h-6 bg-gray-200 rounded-full hover:bg-gray-300 focus:outline-none">
                            <svg
                                class="w-4 h-4 text-gray-600"
                                fill="none"
                                stroke="currentColor"
                                stroke-width="2"
                                viewBox="0 0 24 24"
                                xmlns="http://www.w3.org/2000/svg">
                                <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12"></path>
                            </svg>
                        </button>
                    </div>

                    <input id="upload" type="file" accept="image/*" class="hidden" @change="onFileChange" />
                    <!-- 参数配置表单 -->
                    <div class="mb-4">
                        <label class="block mb-2 font-semibold">Parameter Configuration:</label>
                        <input
                            type="text"
                            v-model="param1"
                            class="border border-gray-300 p-2 w-full mb-2"
                            placeholder="Parameter 1" />
                        <input
                            type="text"
                            v-model="param2"
                            class="border border-gray-300 p-2 w-full mb-2"
                            placeholder="Parameter 2" />
                        <button @click="onSubmit" class="bg-blue-500 text-white p-2 rounded mt-2">Submit</button>
                    </div>
                    <label id="status" class="text-gray-600">{{ status }}</label>
                </div>

                <!-- 右侧布局 -->
                <div class="w-1/2 p-4">
                    <h5 class="text-2xl font-bold mb-4">Processed Image Preview</h5>
                    <div
                        id="container"
                        class="relative flex justify-center items-center w-full h-[480px] border-2 border-dashed border-gray-300 rounded-xl overflow-hidden mb-4">
                        <canvas ref="canvasRef" v-if="imageUrl" class="inset-0 object-contain h-full"></canvas>
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>

<script setup>
import { ref, onMounted } from 'vue';
import ort from 'onnxruntime-web';

// Refs
const status = ref('Loading model...');
const ref_size = 512;
const canvasRef = ref(null);
const param1 = ref('');
const param2 = ref('');
const imageUrl = ref('');

// Model and processor
let session;

onMounted(async () => {
    const modelFilepath = '/models/modnet.onnx';
    const response = await fetch(modelFilepath);
    const modelFile = await response.arrayBuffer();
    session = await ort.InferenceSession.create(modelFile);

    status.value = 'Ready';
});

const onFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e2) => {
        imageUrl.value = e2.target.result;
    };
    reader.readAsDataURL(file);
};

const replaceImage = () => {
    imageUrl.value = ''; // 清空图片预览
    canvasRef.value.getContext('2d').clearRect(0, 0, canvasRef.value.width, canvasRef.value.height); // 清除 canvas
    document.getElementById('upload').value = ''; // 重置文件输入框
    status.value = 'Ready for new upload';
};

const onSubmit = () => {
    console.log('Parameter 1:', param1.value);
    console.log('Parameter 2:', param2.value);
    predict(imageUrl.value);
};

function getScaleFactor(im_h, im_w, ref_size) {
    let im_rh, im_rw;

    if (Math.max(im_h, im_w) < ref_size || Math.min(im_h, im_w) > ref_size) {
        if (im_w >= im_h) {
            im_rh = ref_size;
            im_rw = Math.floor((im_w / im_h) * ref_size);
        } else {
            im_rw = ref_size;
            im_rh = Math.floor((im_h / im_w) * ref_size);
        }
    } else {
        im_rh = im_h;
        im_rw = im_w;
    }

    im_rw = im_rw - (im_rw % 32);
    im_rh = im_rh - (im_rh % 32);

    const x_scale_factor = im_rw / im_w;
    const y_scale_factor = im_rh / im_h;

    return { x_scale_factor, y_scale_factor };
}

// Predict foreground of the given image
const predict = async (imageFile) => {
    const img = new Image();
    img.src = imageFile;
    await img.decode();

    const { width: im_w, height: im_h } = img;

    const ctx = canvasRef.value.getContext('2d');
    canvasRef.value.width = im_w;
    canvasRef.value.height = im_h;

    status.value = 'Analyzing';

    // Preprocessing image
    const { x_scale_factor, y_scale_factor } = getScaleFactor(im_h, im_w, ref_size);
    const resizedWidth = Math.floor(im_w * x_scale_factor);
    const resizedHeight = Math.floor(im_h * y_scale_factor);
    ctx.drawImage(img, 0, 0, resizedWidth, resizedHeight);
    const rawImageData = ctx.getImageData(0, 0, resizedWidth, resizedHeight);
    const resizedImage = new Float32Array((rawImageData.data.length / 4) * 3);

    for (let i = 0, j = 0; i < rawImageData.data.length; i += 4, j += 3) {
        resizedImage[j] = (rawImageData.data[i] - 127.5) / 127.5; // R
        resizedImage[j + 1] = (rawImageData.data[i + 1] - 127.5) / 127.5; // G
        resizedImage[j + 2] = (rawImageData.data[i + 2] - 127.5) / 127.5; // B
    }

    // Prepare input shape
    let channels = 3; // Assuming RGB (3 channels)
    let input = new Float32Array(1 * channels * resizedHeight * resizedWidth);

    for (let c = 0; c < channels; c++) {
        for (let h = 0; h < resizedHeight; h++) {
            for (let w = 0; w < resizedWidth; w++) {
                input[c * resizedHeight * resizedWidth + h * resizedWidth + w] =
                    resizedImage[(h * resizedWidth + w) * channels + c];
            }
        }
    }

    // Load ONNX model and make predictions
    const inputName = session.inputNames[0];
    const outputName = session.outputNames[0];

    const feeds = {};
    feeds[inputName] = new ort.Tensor('float32', input, [1, 3, resizedHeight, resizedWidth]);

    const results = await session.run(feeds);
    const matteData = results[outputName].data;
    console.log('info', im_w, im_h, resizedWidth, resizedHeight);

    // Refine the matte and resize back to original size
    const maskData = Float32Array.from(matteData, (v) => v * 255);

    // Draw original image
    ctx.drawImage(img, 0, 0, resizedWidth, resizedHeight);
    const combineData = new ImageData(resizedWidth, resizedHeight);

    console.log('rawImageData length:', rawImageData.data.length, 'mask:', maskData.length, 'combineData', combineData.data.length);

    for (let i = 0; i < im_h; i++) {
        for (let j = 0; j < im_w; j++) {
            const index = i * im_w + j;
            const alphaValue = maskData[index]; // 取单通道值

            combineData.data[4 * index] = rawImageData.data[4 * index]; // R
            combineData.data[4 * index + 1] = rawImageData.data[4 * index + 1]; // G
            combineData.data[4 * index + 2] = rawImageData.data[4 * index + 2]; // B
            combineData.data[4 * index + 3] = alphaValue; // alpha
        }
    }
    canvasRef.value.width = im_w;
    canvasRef.value.height = im_h;
    ctx.putImageData(combineData, 0, 0);
    status.value = 'Done!';
};
</script>
