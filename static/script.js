document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    
    const previewSection = document.getElementById('preview-section');
    const imagePreview = document.getElementById('image-preview');
    const predictBtn = document.getElementById('predict-btn');
    const resetBtn = document.getElementById('reset-btn');
    
    const resultSection = document.getElementById('result-section');
    const loader = document.getElementById('loader');
    const predictionData = document.getElementById('prediction-data');
    
    const speciesName = document.getElementById('species-name');
    const confidenceText = document.getElementById('confidence-text');
    const confidenceBar = document.getElementById('confidence-bar');
    const uploadRuleWarning = document.getElementById('upload-rule-warning');
    const fileMeta = document.getElementById('file-meta');
    
    let currentFile = null;
    const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024;

    // --- Drag & Drop Handlers ---
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.add('drag-over');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.remove('drag-over');
        }, false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    });

    dropZone.addEventListener('click', (event) => {
        if (event.target === browseBtn || browseBtn.contains(event.target)) {
            return;
        }
        fileInput.click();
    });

    // --- Click Handlers ---
    browseBtn.addEventListener('click', (event) => {
        event.preventDefault();
        event.stopPropagation();
        fileInput.click();
    });

    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });

    function handleFiles(files) {
        if (!files || files.length === 0) {
            return;
        }

        const file = files[0];
        const isSingleFile = files.length === 1;
        const isImageFile = file.type.startsWith('image/');

        if (file.size > MAX_FILE_SIZE_BYTES) {
            alert('Image is too large. Please upload a file up to 10MB.');
            return;
        }

        if (!isSingleFile || !isImageFile) {
            currentFile = null;
            if (uploadRuleWarning) {
                uploadRuleWarning.classList.remove('hidden');
            }
            alert('Please upload a single image file with one bird species.');
            return;
        }

        if (uploadRuleWarning) {
            uploadRuleWarning.classList.add('hidden');
        }

        currentFile = file;
        showPreview(currentFile);
    }

    function showPreview(file) {
        // Read file and show image
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onloadend = function() {
            imagePreview.src = reader.result;
            // UI state
            dropZone.classList.add('hidden');
            previewSection.classList.remove('hidden');
            resultSection.classList.add('hidden');
            // reset previous results
            predictionData.classList.add('hidden');
            confidenceBar.style.width = '0%';
            if (fileMeta) {
                fileMeta.textContent = `${file.name} • ${(file.size / (1024 * 1024)).toFixed(2)} MB`;
            }
        }
    }

    // --- Prediction Logic ---
    predictBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        // UI state: Show loader, hide preview buttons
        predictBtn.classList.add('hidden');
        resetBtn.classList.add('hidden');
        browseBtn.disabled = true;
        resultSection.classList.remove('hidden');
        loader.classList.remove('hidden');
        predictionData.classList.add('hidden');

        // Prepare form data
        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.success) {
                displayResult(data.class_name, data.confidence);
            } else {
                alert("Error during prediction: " + data.error);
                resetUI();
            }
        } catch (error) {
            console.error(error);
            alert("Network error occurred.");
            resetUI();
        } finally {
            browseBtn.disabled = false;
        }
    });

    function displayResult(name, confidence) {
        // UI state
        loader.classList.add('hidden');
        predictionData.classList.remove('hidden');
        resetBtn.classList.remove('hidden'); // allow to try again
        predictBtn.classList.remove('hidden');

        // Set data
        speciesName.textContent = name;
        confidenceText.textContent = confidence.toFixed(1) + '%';
        
        // Animate the bar
        setTimeout(() => {
            confidenceBar.style.width = confidence + '%';
        }, 100);
    }

    // --- Reset ---
    resetBtn.addEventListener('click', resetUI);

    function resetUI() {
        currentFile = null;
        fileInput.value = "";
        dropZone.classList.remove('hidden');
        previewSection.classList.add('hidden');
        resultSection.classList.add('hidden');
        predictBtn.classList.remove('hidden');
        resetBtn.classList.remove('hidden');
        browseBtn.disabled = false;
        predictionData.classList.add('hidden');
        confidenceBar.style.width = '0%';
        if (fileMeta) {
            fileMeta.textContent = 'No image selected';
        }
        if (uploadRuleWarning) {
            uploadRuleWarning.classList.remove('hidden');
        }
    }
});
