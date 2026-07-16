const frontInput = document.getElementById('frontInput');
const leftInput = document.getElementById('leftInput');
const rightInput = document.getElementById('rightInput');
const analyzeBtn = document.getElementById('analyzeBtn');

const uploadContent = document.getElementById('uploadContent');
const loading = document.getElementById('loading');
const results = document.getElementById('results');
const errorBox = document.getElementById('errorBox');

const files = { front: null, left: null, right: null };

function handleFileSelect(inputId, key, labelId, boxId) {
    document.getElementById(inputId).addEventListener('change', (e) => {
        if (e.target.files.length) {
            files[key] = e.target.files[0];
            document.getElementById(labelId).innerText = "Uploaded: " + files[key].name;
            document.getElementById(boxId).classList.add('has-file');
            checkReady();
        }
    });
}

handleFileSelect('frontInput', 'front', 'frontLabel', 'boxFront');
handleFileSelect('leftInput', 'left', 'leftLabel', 'boxLeft');
handleFileSelect('rightInput', 'right', 'rightLabel', 'boxRight');

function checkReady() {
    if (files.front && files.left && files.right) {
        analyzeBtn.classList.remove('hidden');
    }
}

analyzeBtn.addEventListener('click', async () => {
    uploadContent.classList.add('hidden');
    loading.classList.remove('hidden');
    errorBox.classList.add('hidden');
    results.classList.add('hidden');

    const formData = new FormData();
    formData.append('front', files.front);
    formData.append('left', files.left);
    formData.append('right', files.right);

    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Failed to analyze image');

        document.getElementById('cheekImage').src = data.cheek_image;
        document.getElementById('eyebrowFullImage').src = data.eyebrow_full_image;
        document.getElementById('rBrowImage').src = data.r_brow_image;
        document.getElementById('lBrowImage').src = data.l_brow_image;
        document.getElementById('lEarImage').src = data.l_ear_image || "";
        document.getElementById('rEarImage').src = data.r_ear_image || "";
        
        document.getElementById('cheekText').innerText = data.cheek_report;
        document.getElementById('eyebrowText').innerText = data.eyebrow_report;
        document.getElementById('earText').innerText = data.ear_report;

        document.getElementById('dropzone').classList.add('hidden');
        results.classList.remove('hidden');

    } catch (err) {
        showError(err.message);
        uploadContent.classList.remove('hidden');
        loading.classList.add('hidden');
    }
});

function showError(msg) {
    errorBox.innerText = msg;
    errorBox.classList.remove('hidden');
}
