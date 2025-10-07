// ==================== ENHANCED UTILITIES WITH STATE MANAGEMENT ====================

const Utils = {
    // ==================== DOM MANIPULATION ====================
    
    getEl: (id) => document.getElementById(id),
    qs: (selector) => document.querySelector(selector),
    qsa: (selector) => document.querySelectorAll(selector),
    
    createElement: (tag, classes = [], attrs = {}) => {
        const el = document.createElement(tag);
        if (classes.length) el.className = classes.join(' ');
        Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, value));
        return el;
    },

    // ==================== LOADING & OVERLAY ====================
    
    showLoading: (text = 'Processing...') => {
        const overlay = Utils.getEl('loadingOverlay');
        if (!overlay) {
            console.warn('Loading overlay not found');
            return;
        }
        const loadingText = overlay.querySelector('.loading-text');
        if (loadingText) loadingText.textContent = text;
        overlay.classList.add('active');
        overlay.style.display = 'flex';
        
        // Auto-hide protection
        Utils._loadingTimer = setTimeout(() => {
            console.warn('Loading timeout - auto hiding');
            Utils.forceHideLoading();
        }, 30000); // 30 seconds max
    },
    
    hideLoading: () => {
        if (Utils._loadingTimer) {
            clearTimeout(Utils._loadingTimer);
            Utils._loadingTimer = null;
        }
        const overlay = Utils.getEl('loadingOverlay');
        if (overlay) {
            overlay.classList.remove('active');
            // Delay display:none to allow animation
            setTimeout(() => {
                overlay.style.display = 'none';
            }, 300);
        }
    },

    forceHideLoading: () => {
        if (Utils._loadingTimer) {
            clearTimeout(Utils._loadingTimer);
            Utils._loadingTimer = null;
        }
        const overlay = Utils.getEl('loadingOverlay');
        if (overlay) {
            overlay.classList.remove('active');
            overlay.style.display = 'none';
        }
    },

    // ==================== TOAST NOTIFICATIONS ====================
    
    showToast: (message, type = 'info', duration = 3000) => {
        const container = Utils.getEl('toastContainer');
        if (!container) {
            console.log(`Toast: [${type}] ${message}`);
            return;
        }
        
        const icons = {
            success: '✓',
            error: '✗',
            warning: '⚠',
            info: 'ℹ'
        };
        
        const toast = Utils.createElement('div', ['toast', type]);
        toast.innerHTML = `
            <div class="toast-icon">${icons[type] || icons.info}</div>
            <div class="toast-message">${message}</div>
            <button class="toast-close" onclick="this.parentElement.remove()">×</button>
        `;
        
        container.appendChild(toast);
        
        setTimeout(() => {
            toast.style.animation = 'fadeOut 0.3s ease';
            setTimeout(() => toast.remove(), 300);
        }, duration);
    },

    // ==================== DATA FORMATTING ====================
    
    formatNumber: (num, decimals = 2) => {
        if (num === null || num === undefined || isNaN(num)) return 'N/A';
        return Number(num).toFixed(decimals);
    },
    
    formatLargeNumber: (num) => {
        if (num === null || num === undefined || isNaN(num)) return 'N/A';
        if (num >= 1e9) return (num / 1e9).toFixed(1) + 'B';
        if (num >= 1e6) return (num / 1e6).toFixed(1) + 'M';
        if (num >= 1e3) return (num / 1e3).toFixed(1) + 'K';
        return num.toString();
    },
    
    formatPercent: (num, decimals = 2) => {
        if (num === null || num === undefined || isNaN(num)) return 'N/A';
        return Number(num).toFixed(decimals) + '%';
    },
    
    formatDate: (dateString) => {
        if (!dateString) return 'N/A';
        try {
            const date = new Date(dateString);
            return date.toLocaleDateString('en-US', {
                year: 'numeric',
                month: 'short',
                day: 'numeric'
            });
        } catch (e) {
            return dateString;
        }
    },

    formatFileSize: (bytes) => {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
    },

    // ==================== VALIDATION ====================
    
    validateFile: (file) => {
        const errors = [];
        
        if (!file) {
            errors.push('No file selected');
            return errors;
        }
        
        const ext = '.' + file.name.split('.').pop().toLowerCase();
        if (!CONFIG.UPLOAD.ALLOWED_EXTENSIONS.includes(ext)) {
            errors.push(`Invalid file type. Allowed: ${CONFIG.UPLOAD.ALLOWED_EXTENSIONS.join(', ')}`);
        }
        
        if (file.size > CONFIG.UPLOAD.MAX_FILE_SIZE) {
            errors.push(`File too large. Max size: ${Utils.formatFileSize(CONFIG.UPLOAD.MAX_FILE_SIZE)}`);
        }
        
        if (file.size === 0) {
            errors.push('File is empty');
        }
        
        return errors;
    },

    // ==================== CSV PARSING ====================
    
    parseCSV: (text) => {
        const lines = text.split('\n').filter(line => line.trim());
        if (lines.length === 0) return { headers: [], rows: [] };
        
        const headers = lines[0].split(',').map(h => h.trim());
        const rows = lines.slice(1).map(line => {
            return line.split(',').map(cell => cell.trim());
        });
        
        return { headers, rows };
    },
    
    detectDateColumn: (headers, firstRow) => {
        const dateKeywords = ['date', 'time', 'timestamp', 'datetime', 'period', 'tanggal'];
        for (let i = 0; i < headers.length; i++) {
            const header = headers[i].toLowerCase();
            if (dateKeywords.some(kw => header.includes(kw))) {
                return i;
            }
        }
        
        for (let i = 0; i < firstRow.length; i++) {
            const value = firstRow[i];
            if (/^\d{4}-\d{2}-\d{2}/.test(value) || /^\d{1,2}\/\d{1,2}\/\d{2,4}/.test(value)) {
                return i;
            }
        }
        
        return 0;
    },
    
    detectValueColumn: (headers, firstRow, dateColumnIndex) => {
        for (let i = 0; i < firstRow.length; i++) {
            if (i !== dateColumnIndex && !isNaN(parseFloat(firstRow[i]))) {
                return i;
            }
        }
        return dateColumnIndex === 0 ? 1 : 0;
    },

    // ==================== ARRAY UTILITIES ====================
    
    unique: (arr) => [...new Set(arr)],
    
    mean: (arr) => {
        const nums = arr.filter(x => !isNaN(x));
        return nums.length ? nums.reduce((a, b) => a + b, 0) / nums.length : 0;
    },
    
    std: (arr) => {
        const avg = Utils.mean(arr);
        const squareDiffs = arr.map(x => Math.pow(x - avg, 2));
        return Math.sqrt(Utils.mean(squareDiffs));
    },

    // ==================== DEBOUNCE & THROTTLE ====================
    
    debounce: (func, wait = 300) => {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },

    // ==================== STATE MANAGEMENT (IN-MEMORY) ====================
    
    memoryStorage: {},
    _storageWarningShown: false,
    _loadingTimer: null,
    
    saveToMemory: (key, value) => {
        try {
            if (!key || typeof key !== 'string') {
                console.error('Invalid key for memory storage');
                return false;
            }
            
            const serialized = JSON.stringify(value);
            Utils.memoryStorage[key] = serialized;
            
            console.log(`✓ Saved to memory: ${key} (${(serialized.length / 1024).toFixed(2)} KB)`);
            
            if (!Utils._storageWarningShown) {
                console.warn('⚠️ Using in-memory storage. Data will be lost on page refresh.');
                Utils._storageWarningShown = true;
            }
            
            return true;
        } catch (error) {
            console.error(`Failed to save to memory: ${key}`, error);
            return false;
        }
    },
    
    getFromMemory: (key) => {
        try {
            if (!key || typeof key !== 'string') {
                console.error('Invalid key for memory retrieval');
                return null;
            }
            
            const value = Utils.memoryStorage[key];
            if (!value) return null;
            
            return JSON.parse(value);
        } catch (error) {
            console.error(`Failed to retrieve from memory: ${key}`, error);
            return null;
        }
    },
    
    removeFromMemory: (key) => {
        try {
            if (Utils.memoryStorage[key]) {
                delete Utils.memoryStorage[key];
                console.log(`✓ Removed from memory: ${key}`);
                return true;
            }
            return false;
        } catch (error) {
            console.error(`Failed to remove from memory: ${key}`, error);
            return false;
        }
    },
    
    clearAllMemory: () => {
        Utils.memoryStorage = {};
        console.log('✓ All memory cleared');
    },
    
    getMemoryStats: () => {
        const keys = Object.keys(Utils.memoryStorage);
        let totalSize = 0;
        
        keys.forEach(key => {
            totalSize += Utils.memoryStorage[key].length;
        });
        
        return {
            keys: keys,
            count: keys.length,
            size: totalSize,
            sizeFormatted: Utils.formatFileSize(totalSize)
        };
    },

    // ==================== DOWNLOAD ====================
    
    downloadCSV: (data, filename = 'data.csv') => {
        const blob = new Blob([data], { type: 'text/csv;charset=utf-8;' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        window.URL.revokeObjectURL(url);
        Utils.showToast('File downloaded', 'success');
    },
    
    downloadJSON: (data, filename = 'data.json') => {
        const json = JSON.stringify(data, null, 2);
        const blob = new Blob([json], { type: 'application/json' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        window.URL.revokeObjectURL(url);
        Utils.showToast('File downloaded', 'success');
    },

    // ==================== COLOR UTILITIES ====================
    
    generateColorPalette: (count) => {
        const colors = [
            '#2196F3', '#64B5F6', '#90CAF9',
            '#4CAF50', '#81C784', '#A5D6A7',
            '#ff9800', '#FFB74D', '#FFCC80',
            '#f44336', '#E57373', '#EF9A9A'
        ];
        
        if (count <= colors.length) {
            return colors.slice(0, count);
        }
        
        const result = [...colors];
        while (result.length < count) {
            result.push(`hsl(${Math.random() * 360}, 70%, 60%)`);
        }
        
        return result;
    },

    // ==================== ANIMATION HELPERS ====================
    
    scrollToElement: (elementId, offset = 100) => {
        const element = Utils.getEl(elementId);
        if (element) {
            const y = element.getBoundingClientRect().top + window.pageYOffset - offset;
            window.scrollTo({ top: y, behavior: 'smooth' });
        }
    },
    
    animateNumber: (element, start, end, duration = 1000) => {
        if (!element) return;
        
        const range = end - start;
        const increment = range / (duration / 16);
        let current = start;
        
        const timer = setInterval(() => {
            current += increment;
            if ((increment > 0 && current >= end) || (increment < 0 && current <= end)) {
                current = end;
                clearInterval(timer);
            }
            element.textContent = Math.round(current);
        }, 16);
    },

    // ==================== SAFE VALUE ACCESS ====================
    
    getNestedValue: (obj, path, defaultValue = null) => {
        if (!obj || typeof obj !== 'object') return defaultValue;
        
        const parts = path.split('.');
        let current = obj;
        
        for (const part of parts) {
            if (!current || typeof current !== 'object' || !(part in current)) {
                return defaultValue;
            }
            current = current[part];
        }
        
        return current !== undefined && current !== null ? current : defaultValue;
    },
    
    isValidValue: (value) => {
        if (value === null || value === undefined) return false;
        if (typeof value === 'number' && isNaN(value)) return false;
        if (typeof value === 'string' && value.trim() === '') return false;
        return true;
    },
    
    safeArrayAccess: (arr, index, defaultValue = null) => {
        if (!Array.isArray(arr)) return defaultValue;
        if (index < 0 || index >= arr.length) return defaultValue;
        return arr[index];
    }
};

window.Utils = Utils;
console.log('✓ Utils loaded');