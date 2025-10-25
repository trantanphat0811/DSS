// DOM Elements - sẽ được khởi tạo khi DOM load
let form, emailText, submitBtn, loading, result, resultTitle, resultText, confidence;

// Ví dụ email
const examples = {
    spam: "WIN FREE MONEY NOW! Click here to claim your $1000 prize! Limited time offer! Don't miss out! Act now!",
    ham: "Hi John, I hope you're doing well. I wanted to follow up on our meeting yesterday. Please let me know if you have any questions about the project proposal. Best regards, Sarah"
};

// Khởi tạo DOM elements và event listeners
function initializeElements() {
    form = document.getElementById('spamForm');
    emailText = document.getElementById('emailText');
    submitBtn = document.getElementById('submitBtn');
    loading = document.getElementById('loading');
    result = document.getElementById('result');
    resultTitle = document.getElementById('resultTitle');
    resultText = document.getElementById('resultText');
    confidence = document.getElementById('confidence');
    
    // Chỉ thêm event listener nếu form tồn tại
    if (form) {
        form.addEventListener('submit', handleFormSubmit);
    }
}

// Functions
async function handleFormSubmit(e) {
    e.preventDefault();
    
    // Kiểm tra xem các elements có tồn tại không
    if (!emailText || !submitBtn || !loading || !result) {
        console.error('DOM elements chưa được khởi tạo');
        return;
    }
    
    const text = emailText.value.trim();
    if (!text) {
        alert('Vui lòng nhập nội dung email!');
        return;
    }

    // Hiển thị loading
    submitBtn.disabled = true;
    loading.style.display = 'block';
    result.style.display = 'none';

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ email_text: text })
        });

        const data = await response.json();

        if (data.success) {
            showResult(data.result, data.confidence, data.email_text);
        } else {
            showError(data.error);
        }
    } catch (error) {
        console.error('Error in form submission:', error);
        showError('Lỗi kết nối đến server: ' + error.message);
    } finally {
        if (submitBtn) submitBtn.disabled = false;
        if (loading) loading.style.display = 'none';
    }
}

function showResult(resultValue, confidenceValue, emailText) {
    // Kiểm tra xem các elements có tồn tại không
    if (!resultTitle || !resultText || !confidence || !result) {
        console.error('DOM elements chưa được khởi tạo');
        return;
    }
    
    resultTitle.textContent = resultValue;
    resultText.textContent = `Email: "${emailText.substring(0, 100)}${emailText.length > 100 ? '...' : ''}"`;
    confidence.textContent = `Độ tin cậy: ${confidenceValue}%`;

    // Thiết lập màu sắc dựa trên kết quả
    result.className = 'result';
    if (resultValue === 'SPAM') {
        result.classList.add('danger');
    } else {
        result.classList.add('success');
    }

    result.style.display = 'block';
}

function showError(error) {
    // Kiểm tra xem các elements có tồn tại không
    if (!resultTitle || !resultText || !confidence || !result) {
        console.error('DOM elements chưa được khởi tạo');
        return;
    }
    
    resultTitle.textContent = 'Lỗi';
    resultText.textContent = error;
    confidence.textContent = '';
    
    result.className = 'result warning';
    result.style.display = 'block';
}

function loadExample(type) {
    if (emailText) {
        emailText.value = examples[type];
    }
}

function clearText() {
    if (emailText && result) {
        emailText.value = '';
        result.style.display = 'none';
    }
}

// Kiểm tra trạng thái server khi trang load
document.addEventListener('DOMContentLoaded', function() {
    // Khởi tạo DOM elements khi DOM đã sẵn sàng
    initializeElements();
});

window.addEventListener('load', async () => {
    try {
        const response = await fetch('/health');
        const data = await response.json();
        if (!data.model_loaded) {
            showError('Mô hình chưa được tải. Vui lòng thử lại sau.');
        }
    } catch (error) {
        console.log('Không thể kết nối đến server');
    }
});
