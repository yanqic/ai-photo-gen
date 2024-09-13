<template>
    <div class="min-h-screen bg-gray-100 flex flex-col items-center justify-center p-6">
        <h1 class="text-3xl font-bold mb-2">
            Background Removal with
            <a href="http://github.com/xenova/transformers.js" target="_blank" class="text-blue-600 hover:underline"
                >🤗 Transformers.js</a
            >
        </h1>
        <h4 class="text-lg mb-6">
            Runs locally in your browser, powered by the
            <a href="https://huggingface.co/briaai/RMBG-1.4" target="_blank" class="text-blue-600 hover:underline"
                >RMBG V1.4 model</a
            >
            from <a href="https://bria.ai/" target="_blank" class="text-blue-600 hover:underline">BRIA AI</a>
        </h4>
        <div
            id="container"
            ref="imageContainer"
            class="relative w-[720px] h-[480px] max-w-full max-h-full border-2 border-dashed border-gray-300 rounded-xl overflow-hidden mb-4"
            :style="containerStyle">
            <label
                id="upload-button"
                for="upload"
                class="absolute inset-0 flex flex-col items-center justify-center cursor-pointer"
                v-if="!imageUrl">
                <svg width="25" height="25" viewBox="0 0 25 25" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path
                        fill="#000"
                        d="M3.5 24.3a3 3 0 0 1-1.9-.8c-.5-.5-.8-1.2-.8-1.9V2.9c0-.7.3-1.3.8-1.9.6-.5 1.2-.7 2-.7h18.6c.7 0 1.3.2 1.9.7.5.6.7 1.2.7 2v18.6c0 .7-.2 1.4-.7 1.9a3 3 0 0 1-2 .8H3.6Zm0-2.7h18.7V2.9H3.5v18.7Zm2.7-2.7h13.3c.3 0 .5 0 .6-.3v-.7l-3.7-5a.6.6 0 0 0-.6-.2c-.2 0-.4 0-.5.3l-3.5 4.6-2.4-3.3a.6.6 0 0 0-.6-.3c-.2 0-.4.1-.5.3l-2.7 3.6c-.1.2-.2.4 0 .7.1.2.3.3.6.3Z"></path>
                </svg>
                <span class="mt-2">Click to upload image</span>
                <span id="example" class="text-sm text-blue-600 underline cursor-pointer" @click.prevent="loadExample">
                    (or try example)
                </span>
            </label>
            <canvas ref="canvas" v-if="imageUrl" class="absolute inset-0 w-full h-full"></canvas>
        </div>
        <label id="status" class="text-gray-600">{{ status }}</label>
        <input id="upload" type="file" accept="image/*" class="hidden" @change="onFileChange" />
    </div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue';
import { AutoModel, AutoProcessor, env, RawImage } from '@xenova/transformers';

// Constants
const EXAMPLE_URL =
    'https://images.pexels.com/photos/5965592/pexels-photo-5965592.jpeg?auto=compress&cs=tinysrgb&w=1024';

// Refs
const status = ref('Loading model...');
const imageContainer = ref(null);
const canvas = ref(null);
const imageUrl = ref('');

// Computed
const containerStyle = computed(() => ({
    backgroundImage: imageUrl.value ? '' : `url(${imageUrl.value})`,
    backgroundSize: '100% 100%',
    backgroundPosition: 'center',
    backgroundRepeat: 'no-repeat'
}));

// Model and processor
let model, processor;

onMounted(async () => {
    // Since we will download the model from the Hugging Face Hub, we can skip the local model check
    env.allowLocalModels = true;
    env.remoteHost = 'https://hf-mirror.com';

    // Proxy the WASM backend to prevent the UI from freezing
    env.backends.onnx.wasm.proxy = true;
    env.useBrowserCache = false;

    // Load model and processor
    env.localModelPath = '/models/';
    env.backends.onnx.wasm.wasmPaths='/public/'


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
            feature_extractor_type: 'ImageFeatureExtractor',
            image_std: [1, 1, 1],
            resample: 2,
            rescale_factor: 0.00392156862745098,
            size: { width: 1024, height: 1024 }
        }
    });

    status.value = 'Ready';
});

const loadExample = () => {
    predict(EXAMPLE_URL);
};

const onFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e2) => predict(e2.target.result);
    reader.readAsDataURL(file);
};

// Predict foreground of the given image
const predict = async (url) => {
    // Read image
    const image = await RawImage.fromURL(url);

    // Update UI
    imageUrl.value = url;

    // Set container width and height depending on the image aspect ratio
    const ar = image.width / image.height;
    const [cw, ch] = ar > 720 / 480 ? [720, 720 / ar] : [480 * ar, 480];
    imageContainer.value.style.width = `${cw}px`;
    imageContainer.value.style.height = `${ch}px`;

    status.value = 'Analysing...';

    // Preprocess image
    const { pixel_values } = await processor(image);

    // Predict alpha matte
    const { output } = await model({ input: pixel_values });

    // Resize mask back to original size
    const mask = await RawImage.fromTensor(output[0].mul(255).to('uint8')).resize(image.width, image.height);

    // Create new canvas
    const ctx = canvas.value.getContext('2d');
    canvas.value.width = image.width;
    canvas.value.height = image.height;

    // Draw original image output to canvas
    ctx.drawImage(image.toCanvas(), 0, 0);

    // Update alpha channel
    const pixelData = ctx.getImageData(0, 0, image.width, image.height);
    for (let i = 0; i < mask.data.length; ++i) {
        pixelData.data[4 * i + 3] = mask.data[i];
    }
    ctx.putImageData(pixelData, 0, 0);

    // Update UI
    imageContainer.value.style.background = `url("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQBAMAAADt3eJSAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAGUExURb+/v////5nD/3QAAAAJcEhZcwAADsMAAA7DAcdvqGQAAAAUSURBVBjTYwABQSCglEENMxgYGAAynwRB8BEAgQAAAABJRU5ErkJggg==")`;
    status.value = 'Done!';
};
</script>

<style scoped>
* {
    box-sizing: border-box;
    padding: 0;
    margin: 0;
    font-family: sans-serif;
}

html,
body {
    height: 100%;
}

body {
    padding: 16px 32px;
}

body,
#container,
#upload-button {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
}

h1,
h4 {
    text-align: center;
}

h4 {
    margin-top: 0.5rem;
}

#container {
    position: relative;
    width: 720px;
    height: 480px;
    max-width: 100%;
    max-height: 100%;
    border: 2px dashed #d1d5db;
    border-radius: 0.75rem;
    overflow: hidden;
    margin-top: 1rem;
    background-size: 100% 100%;
    background-position: center;
    background-repeat: no-repeat;
}

#upload-button {
    gap: 0.4rem;
    font-size: 18px;
    cursor: pointer;
}

#upload {
    display: none;
}

svg {
    pointer-events: none;
}

#example {
    font-size: 14px;
    text-decoration: underline;
    cursor: pointer;
}

#example:hover {
    color: #2563eb;
}

canvas {
    position: absolute;
    width: 100%;
    height: 100%;
}

#status {
    min-height: 16px;
    margin: 8px 0;
}
</style>
