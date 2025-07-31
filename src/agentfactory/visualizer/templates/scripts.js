function toggleSystem(contentId, headerElement) {
    const content = document.getElementById(contentId || 'systemContent');
    const icon = headerElement ? headerElement.querySelector('.collapse-icon') : document.querySelector('.collapse-icon');
    
    if (content && icon) {
        content.classList.toggle('hidden');
        icon.classList.toggle('collapsed');
    }
}

function switchResult(resultIndex) {
    // Hide all result content
    const allResults = document.querySelectorAll('.result-content');
    allResults.forEach(result => {
        result.style.display = 'none';
    });
    
    // Show selected result
    const selectedResult = document.getElementById('result-' + resultIndex);
    if (selectedResult) {
        selectedResult.style.display = 'block';
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Fade out animation
    const style = document.createElement('style');
    style.textContent = `
        @keyframes fadeOut {
            from { opacity: 1; transform: scale(1); }
            to { opacity: 0; transform: scale(0); }
        }
    `;
    document.head.appendChild(style);
});