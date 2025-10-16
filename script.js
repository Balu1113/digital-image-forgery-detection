const uploadBox = document.getElementById('uploadBox');
const fileInput = document.getElementById('fileInput');
const loading = document.getElementById('loading');
const results = document.getElementById('results');
const error = document.getElementById('error');
const uploadSection = document.querySelector('.upload-section');

uploadBox.addEventListener('click', () => fileInput.click());

uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.style.borderColor = '#764ba2';
    uploadBox.style.background = '#f0f1ff';
});

uploadBox.addEventListener('dragleave', () => {
    uploadBox.style.borderColor = '#667eea';
    uploadBox.style.background = '#f8f9ff';
});

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.style.borderColor = '#667eea';
    uploadBox.style.background = '#f8f9ff';
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

function handleFile(file) {
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
    
    if (!validTypes.includes(file.type)) {
        showError('Invalid file type. Please upload a JPEG or PNG image.');
        return;
    }
    
    if (file.size > 16 * 1024 * 1024) {
        showError('File too large. Maximum size is 16MB.');
        return;
    }
    
    uploadImage(file);
}

function uploadImage(file) {
    uploadSection.style.display = 'none';
    loading.classList.add('active');
    results.classList.remove('active');
    error.classList.remove('active');
    
    const formData = new FormData();
    formData.append('file', file);
    
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            showError(data.error);
        } else {
            showResults(data);
        }
    })
    .catch(err => {
        showError('An error occurred while processing the image. Please try again.');
        console.error(err);
    });
}

function showResults(data) {
    loading.classList.remove('active');
    results.classList.add('active');
    
    const predictionLabel = document.getElementById('predictionLabel');
    const predictionCard = document.getElementById('predictionCard');
    const confidenceFill = document.getElementById('confidenceFill');
    const confidenceText = document.getElementById('confidenceText');
    const originalImage = document.getElementById('originalImage');
    const elaImage = document.getElementById('elaImage');
    
    predictionLabel.textContent = data.prediction === 'Real' ? '✓ Authentic Image' : '✗ Forgery Detected';
    
    if (data.prediction === 'Real') {
        predictionCard.style.background = 'linear-gradient(135deg, #56ab2f 0%, #a8e063 100%)';
    } else {
        predictionCard.style.background = 'linear-gradient(135deg, #ee0979 0%, #ff6a00 100%)';
    }
    
    confidenceFill.style.width = data.confidence + '%';
    confidenceText.textContent = `Confidence: ${data.confidence.toFixed(2)}%`;
    
    originalImage.src = 'data:image/jpeg;base64,' + data.original_image;
    elaImage.src = 'data:image/jpeg;base64,' + data.ela_image;
}

function showError(message) {
    loading.classList.remove('active');
    results.classList.remove('active');
    error.classList.add('active');
    uploadSection.style.display = 'none';
    
    document.getElementById('errorMessage').textContent = message;
}

document.getElementById('analyzeAnother').addEventListener('click', () => {
    resetApp();
});

document.getElementById('tryAgain').addEventListener('click', () => {
    resetApp();
});

function resetApp() {
    uploadSection.style.display = 'block';
    loading.classList.remove('active');
    results.classList.remove('active');
    error.classList.remove('active');
    fileInput.value = '';
}
