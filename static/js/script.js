const fileInput = document.querySelector("[data-file-input]");
const dropZone = document.querySelector("[data-drop-zone]");
const preview = document.querySelector("[data-preview]");
const uploadForm = document.querySelector("[data-upload-form]");
const loader = document.querySelector("[data-loader]");
const submitButton = document.querySelector("[data-submit]");

if (dropZone && fileInput) {
    ["dragenter", "dragover"].forEach((eventName) => {
        dropZone.addEventListener(eventName, (event) => {
            event.preventDefault();
            dropZone.classList.add("dragover");
        });
    });

    ["dragleave", "drop"].forEach((eventName) => {
        dropZone.addEventListener(eventName, (event) => {
            event.preventDefault();
            dropZone.classList.remove("dragover");
        });
    });

    dropZone.addEventListener("drop", (event) => {
        const file = event.dataTransfer.files[0];
        if (!file) return;
        fileInput.files = event.dataTransfer.files;
        renderPreview(file);
    });

    fileInput.addEventListener("change", () => {
        const file = fileInput.files[0];
        if (file) renderPreview(file);
    });
}

if (uploadForm) {
    uploadForm.addEventListener("submit", () => {
        if (loader) loader.hidden = false;
        if (submitButton) {
            submitButton.disabled = true;
            submitButton.textContent = "Analyzing...";
        }
    });
}

function renderPreview(file) {
    if (!preview || !file.type.startsWith("image/")) return;
    preview.src = URL.createObjectURL(file);
    preview.hidden = false;
}

const chart = document.getElementById("featureChart");
if (chart) {
    const features = JSON.parse(chart.dataset.features || "{}");
    drawFeatureChart(chart, features);
}

function drawFeatureChart(canvas, features) {
    const ctx = canvas.getContext("2d");
    const entries = Object.entries(features);
    const width = canvas.width = canvas.offsetWidth * window.devicePixelRatio;
    const height = canvas.height = 180 * window.devicePixelRatio;
    const scale = window.devicePixelRatio;
    const max = Math.max(...entries.map(([, value]) => Number(value)), 1);

    ctx.scale(scale, scale);
    ctx.clearRect(0, 0, width, height);
    ctx.font = "12px Segoe UI, Arial";

    entries.forEach(([key, value], index) => {
        const barWidth = (canvas.offsetWidth - 32) / entries.length;
        const x = 16 + index * barWidth;
        const barHeight = (Number(value) / max) * 105;
        const y = 130 - barHeight;
        const gradient = ctx.createLinearGradient(0, y, 0, 130);
        gradient.addColorStop(0, "#34d6f4");
        gradient.addColorStop(1, "#8f5cff");
        ctx.fillStyle = gradient;
        ctx.fillRect(x, y, Math.max(18, barWidth - 18), barHeight);
        ctx.fillStyle = "#edf7ff";
        ctx.fillText(String(value), x, Math.max(14, y - 8));
        ctx.fillStyle = "#9cb3c9";
        ctx.fillText(key.replace("_", " "), x, 158);
    });
}
