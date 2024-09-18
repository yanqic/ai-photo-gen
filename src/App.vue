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
                        class="relative w-full h-[480px] border-2 border-gray-300 rounded-xl overflow-hidden mb-4">
                        <img :src="imageUrl" class="w-full h-full object-cover" />
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
                        class="relative w-full h-[480px] border-2 border-dashed border-gray-300 rounded-xl overflow-hidden mb-4">
                        <canvas ref="canvas" v-if="imageUrl" class="absolute inset-0 w-full h-full"></canvas>
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue';
import { AutoModel, AutoProcessor, env, RawImage } from '@xenova/transformers';

// Refs
const status = ref('Loading model...');
const canvas = ref(null);
const param1 = ref('');
const param2 = ref('');
const imageUrl = ref('');

// Model and processor
let model, processor;

onMounted(async () => {
    env.allowLocalModels = true;
    env.useBrowserCache = false;

    model = await AutoModel.from_pretrained('RMBG-1.4', {
        config: { model_type: 'custom' }
    });

    processor = await AutoProcessor.from_pretrained('RMBG-1.4', {
        config: {
            do_normalize: true,
            do_pad: false,
            do_rescale: true,
            do_resize: true,
            image_mean: [0.5, 0.5, 0.5],
            image_std: [1, 1, 1],
            resample: 2,
            rescale_factor: 0.00392156862745098,
            size: { width: 1024, height: 1024 }
        }
    });

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
    canvas.value.getContext('2d').clearRect(0, 0, canvas.value.width, canvas.value.height); // 清除 canvas
    document.getElementById('upload').value = ''; // 重置文件输入框
    status.value = 'Ready for new upload';
};

const onSubmit = () => {
    console.log('Parameter 1:', param1.value);
    console.log('Parameter 2:', param2.value);
    predict(imageUrl.value);
};

const predict = async (url) => {
    const image = await RawImage.fromURL(url);
    imageUrl.value = url;

    status.value = 'Analysing...';

    const { pixel_values } = await processor(image);
    const { output } = await model({ input: pixel_values });

    const mask = await RawImage.fromTensor(output[0].mul(255).to('uint8')).resize(image.width, image.height);

    const ctx = canvas.value.getContext('2d');
    canvas.value.width = image.width;
    canvas.value.height = image.height;
    ctx.drawImage(image.toCanvas(), 0, 0);

    const pixelData = ctx.getImageData(0, 0, image.width, image.height);
    for (let i = 0; i < mask.data.length; ++i) {
        pixelData.data[4 * i + 3] = mask.data[i];
    }
    ctx.putImageData(pixelData, 0, 0);

    status.value = 'Done!';
};
</script>
