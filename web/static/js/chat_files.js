// chat_files.js
// Handles temporary file uploads for the chat session

let currentTempFilePath = null;
let currentFileName = null;
let currentFileDataUrl = null;
let currentFileType = null;

export function initChatFiles() {
    const uploadBtn = document.getElementById('uploadTempBtn');
    const fileInput = document.getElementById('tempFileInput');
    const previewContainer = document.getElementById('filePreviewContainer');
    const fileNameSpan = document.getElementById('fileName');
    const clearBtn = document.getElementById('clearFileBtn');

    if (!uploadBtn || !fileInput) return;

    uploadBtn.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        currentFileName = file.name;
        currentFileType = file.type;
        fileNameSpan.textContent = `📎 ${file.name}`;
        previewContainer.style.display = 'flex';

        if (file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = (ev) => {
                currentFileDataUrl = ev.target.result;
            };
            reader.readAsDataURL(file);
        } else {
            currentFileDataUrl = null;
        }

        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch('/api/chat/upload_temp', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            if (data.status === 'uploaded') {
                currentTempFilePath = data.file_path;
                console.log('File uploaded successfully:', currentTempFilePath);
            } else {
                alert('Upload failed: ' + (data.error || 'Unknown error'));
                clearFile();
            }
        } catch (error) {
            console.error('Error uploading file:', error);
            alert('Error uploading file');
            clearFile();
        }
    });

    clearBtn.addEventListener('click', () => {
        clearFile();
    });
}

export function getCurrentTempFile() {
    return currentTempFilePath;
}

export function getCurrentFileInfo() {
    return {
        path: currentTempFilePath,
        name: currentFileName,
        type: currentFileType,
        dataUrl: currentFileDataUrl
    };
}

export function clearFile() {
    const fileInput = document.getElementById('tempFileInput');
    const previewContainer = document.getElementById('filePreviewContainer');
    const fileNameSpan = document.getElementById('fileName');

    currentTempFilePath = null;
    currentFileName = null;
    currentFileDataUrl = null;
    currentFileType = null;
    if (fileInput) fileInput.value = '';
    if (fileNameSpan) fileNameSpan.textContent = '';
    if (previewContainer) previewContainer.style.display = 'none';
}
