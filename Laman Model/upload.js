// ==================== UPLOAD PAGE ====================

const UploadPage = {
    selectedFile: null,
    previewData: null,
    maxPreviewRows: 10,
    
    /**
     * Render upload page
     */
    render() {
        const container = Utils.getEl('mainContainer');
        
        container.innerHTML = `
            <div class="page active" id="uploadPage">
                <h1 style="margin-bottom: 2rem; animation: fadeIn 0.5s ease;">
                    Unggah Data Deret Waktu Anda
                </h1>

                <!-- Upload Card -->
                <div class="card" style="animation: slideInUp 0.6s ease;">
                    <h2 class="card-title">Unggah Data</h2>
                    <p class="card-subtitle">
                        Unggah file CSV yang berisi data deret waktu Anda
                    </p>

                    <!-- Upload Zone -->
                    <div class="upload-zone" id="uploadZone">
                        <div class="upload-icon">☁️</div>
                        <h3 style="color: var(--light-blue); margin-bottom: 0.5rem;">
                            Taruh dan Masukkan File CSV Anda di Sini
                        </h3>
                        <p style="color: var(--text-secondary); margin-bottom: 1.5rem;">
                            atau klik tombol di bawah untuk menjelajahi file
                        </p>
                        <input type="file" id="fileInput" accept=".csv" style="display: none;">
                        <button class="btn btn-primary" onclick="document.getElementById('fileInput').click()">
                            Jelajahi File
                        </button>
                        <p style="color: var(--text-secondary); font-size: 0.9rem; margin-top: 1rem;">
                            Supported format: CSV (Max ${Utils.formatFileSize(CONFIG.UPLOAD.MAX_FILE_SIZE)})
                        </p>
                        <p style="color: var(--warning); font-size: 0.85rem; margin-top: 0.5rem;">
                            ⚠️ Minimum ${CONFIG.VALIDATION.MIN_DATA_POINTS} data points required
                        </p>
                    </div>

                    <!-- File Info -->
                    <div id="fileInfo" style="display: none; margin-top: 2rem;">
                        <div style="background: rgba(33, 150, 243, 0.1); border-left: 4px solid var(--bright-blue); padding:
                        <div style="background: rgba(33, 150, 243, 0.1); border-left: 4px solid var(--bright-blue); padding: 1.5rem; border-radius: 8px;">
                            <h4 style="color: var(--light-blue); margin-bottom: 1rem;">Selected File</h4>
                            <div class="info-grid">
                                <div class="info-item">
                                    <span class="info-item-label">Nama File:</span>
                                    <span class="info-item-value" id="fileName">-</span>
                                </div>
                                <div class="info-item">
                                    <span class="info-item-label">Ukuran:</span>
                                    <span class="info-item-value" id="fileSize">-</span>
                                </div>
                                <div class="info-item">
                                    <span class="info-item-label">Baris</span>
                                    <span class="info-item-value" id="fileRows">-</span>
                                </div>
                                <div class="info-item">
                                    <span class="info-item-label">Kolom:</span>
                                    <span class="info-item-value" id="fileCols">-</span>
                                </div>
                            </div>
                            <div id="fileValidation" style="margin-top: 1rem;"></div>
                        </div>
                    </div>
                </div>

                <!-- Configuration Card -->
                <div class="card" id="configCard" style="display: none; animation: slideInUp 0.7s ease;">
                    <h2 class="card-title">Configuration</h2>
                    <p class="card-subtitle">
                        Konfigurasi parameter untuk analisis deret waktu
                    </p>

                    <div class="grid-2">
                        <div class="form-group">
                            <label class="form-label">Date Column *</label>
                            <select class="form-control" id="dateColumn" required>
                                <option value="">Select date column</option>
                            </select>
                            <small style="color: var(--text-secondary); font-size: 0.85rem; display: block; margin-top: 0.5rem;">
                                Kolom berisi tanggal/waktu
                            </small>
                        </div>

                        <div class="form-group">
                            <label class="form-label">Value Column *</label>
                            <select class="form-control" id="valueColumn" required>
                                <option value="">Select value column</option>
                            </select>
                            <small style="color: var(--text-secondary); font-size: 0.85rem; display: block; margin-top: 0.5rem;">
                                Kolom berisi nilai numerik yang akan diprediksi
                            </small>
                        </div>

                        <div class="form-group">
                            <label class="form-label">Frequency *</label>
                            <select class="form-control" id="frequency" required>
                                ${CONFIG.UPLOAD.FREQUENCIES.map(f => 
                                    `<option value="${f.value}">${f.label}</option>`
                                ).join('')}
                            </select>
                            <small style="color: var(--text-secondary); font-size: 0.85rem; display: block; margin-top: 0.5rem;">
                                Interval waktu antara titik data (e.g., Harian, Bulanan)
                            </small>
                        </div>

                        <div class="form-group">
                            <label class="form-label">Pipeline Name *</label>
                            <input type="text" class="form-control" id="pipelineName" 
                                   placeholder="e.g., Sales Forecast Q4" required>
                            <small style="color: var(--text-secondary); font-size: 0.85rem; display: block; margin-top: 0.5rem;">
                                Nama untuk pipeline Anda
                            </small>
                        </div>
                    </div>

                    <!-- Exogenous Variables -->
                    <div class="form-group" style="margin-top: 2rem;">
                        <label class="form-label">
                            Exogenous Variables (Optional)
                            <span style="font-weight: normal; color: var(--text-secondary); margin-left: 0.5rem;">
                                - Diperlukan untuk model ARIMAX/SARIMAX
                            </span>
                        </label>
                        <p style="color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 1rem;">
                            Pilih Kolom tambahan yang dapat membantu prediksi (Eksogen)
                        </p>
                        <div id="exogenousVariables" style="display: flex; flex-wrap: wrap; gap: 1rem;">
                            <!-- Will be populated dynamically -->
                        </div>
                    </div>

                    <!-- Data Preview -->
                    <div style="margin-top: 2rem; padding-top: 2rem; border-top: 1px solid var(--border-color);">
                        <h4 style="color: var(--light-blue); margin-bottom: 1rem;">
                            Seputar Data
                            <span style="font-weight: normal; font-size: 0.9rem; color: var(--text-secondary);">
                                (First ${this.maxPreviewRows} rows)
                            </span>
                        </h4>
                        <div style="overflow-x: auto;">
                            <table id="previewTable" style="min-width: 100%;">
                                <thead id="previewHead"></thead>
                                <tbody id="previewBody"></tbody>
                            </table>
                        </div>
                    </div>

                    <!-- Action Buttons -->
                    <div style="display: flex; gap: 1rem; margin-top: 2rem; justify-content: flex-end;">
                        <button class="btn btn-secondary" onclick="UploadPage.reset()">
                            Ulangi
                        </button>
                        <button class="btn btn-primary btn-lg" id="uploadBtn" disabled>
                            Unggah dan Proses Data
                        </button>
                    </div>
                </div>

                <!-- Help Section -->
                <div class="card" style="background: rgba(33, 150, 243, 0.05);">
                    <h3 style="color: var(--light-blue); margin-bottom: 1rem;">Butuh Bantuan?</h3>
                    <div class="grid-2">
                        <div>
                            <h4 style="color: var(--light-blue); font-size: 1rem; margin-bottom: 0.5rem;">Wajib Format CSV</h4>
                            <ul style="color: var(--text-secondary); line-height: 1.8;">
                                <li>Minimal satu kolom tanggal/waktu</li>
                                <li>Minimal satu kolom nilai numerik</li>
                                <li>Minimal ${CONFIG.VALIDATION.MIN_DATA_POINTS} baris data</li>
                                <li>Tidak ada karakter khusus dalam nama kolom</li>
                            </ul>
                        </div>
                        <div>
                            <h4 style="color: var(--light-blue); font-size: 1rem; margin-bottom: 0.5rem;">Example Data Structure</h4>
                            <pre style="background: rgba(10, 22, 40, 0.6); padding: 1rem; border-radius: 8px; font-size: 0.85rem; overflow-x: auto;">
Date,Sales,Temperature
2024-01-01,1500,20
2024-01-02,1650,22
2024-01-03,1450,19
...</pre>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Setup event listeners
        this.setupEventListeners();
    },

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        const uploadZone = Utils.getEl('uploadZone');
        const fileInput = Utils.getEl('fileInput');
        const uploadBtn = Utils.getEl('uploadBtn');

        // Click to upload
        uploadZone.addEventListener('click', (e) => {
            // Don't trigger if clicking the button
            if (e.target.tagName !== 'BUTTON') {
                fileInput.click();
            }
        });

        // File input change
        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) this.handleFile(file);
        });

        // Drag and drop
        uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.stopPropagation();
            uploadZone.classList.add('dragover');
        });

        uploadZone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            e.stopPropagation();
            uploadZone.classList.remove('dragover');
        });

        uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            e.stopPropagation();
            uploadZone.classList.remove('dragover');
            
            const file = e.dataTransfer.files[0];
            if (file) this.handleFile(file);
        });

        // Upload button
        uploadBtn.addEventListener('click', () => this.uploadData());

        // Column selection changes - update preview
        Utils.getEl('dateColumn').addEventListener('change', () => {
            this.updatePreview();
            this.validateForm();
        });
        
        Utils.getEl('valueColumn').addEventListener('change', () => {
            this.updatePreview();
            this.validateForm();
        });

        // Pipeline name input
        Utils.getEl('pipelineName').addEventListener('input', () => {
            this.validateForm();
        });
    },

    /**
     * Handle file selection with validation
     */
    async handleFile(file) {
        // Validate file
        const errors = Utils.validateFile(file);
        if (errors.length > 0) {
            Utils.showToast(errors[0], 'error');
            return;
        }

        this.selectedFile = file;

        // Read file
        Utils.showLoading('Reading file...');
        
        try {
            const text = await this.readFile(file);
            this.previewData = Utils.parseCSV(text);
            
            // Validate data
            const dataValidation = this.validateData(this.previewData);
            
            if (!dataValidation.valid) {
                Utils.hideLoading();
                Utils.showToast(dataValidation.message, 'error');
                this.reset();
                return;
            }
            
            // Show file info
            this.displayFileInfo(file, dataValidation);
            
            // Populate column selectors
            this.populateColumnSelectors();
            
            // Auto-detect columns
            this.autoDetectColumns();
            
            // Show preview
            this.updatePreview();
            
            // Validate form
            this.validateForm();
            
            Utils.hideLoading();
            Utils.showToast('File loaded successfully', 'success');
            
            // Scroll to config
            Utils.scrollToElement('configCard');
            
        } catch (error) {
            Utils.hideLoading();
            Utils.showToast('Error reading file: ' + error.message, 'error');
            console.error('File read error:', error);
        }
    },

    /**
     * Read file as text
     */
    readFile(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => resolve(e.target.result);
            reader.onerror = (e) => reject(new Error('Failed to read file'));
            reader.readAsText(file);
        });
    },

    /**
     * Validate parsed data
     */
    validateData(data) {
        if (!data || !data.headers || !data.rows) {
            return { valid: false, message: 'Invalid CSV format' };
        }

        if (data.headers.length === 0) {
            return { valid: false, message: 'CSV has no columns' };
        }

        if (data.rows.length === 0) {
            return { valid: false, message: 'CSV has no data rows' };
        }

        if (data.rows.length < CONFIG.VALIDATION.MIN_DATA_POINTS) {
            return { 
                valid: false, 
                message: `CSV must have at least ${CONFIG.VALIDATION.MIN_DATA_POINTS} rows. Found: ${data.rows.length}` 
            };
        }

        // Check for empty column names
        const emptyHeaders = data.headers.filter(h => !h || h.trim() === '');
        if (emptyHeaders.length > 0) {
            return { valid: false, message: 'CSV contains empty column names' };
        }

        return { 
            valid: true, 
            rowCount: data.rows.length,
            columnCount: data.headers.length 
        };
    },

    /**
     * Display file information with validation status
     */
    displayFileInfo(file, validation) {
        Utils.getEl('fileName').textContent = file.name;
        Utils.getEl('fileSize').textContent = Utils.formatFileSize(file.size);
        Utils.getEl('fileRows').textContent = this.previewData.rows.length;
        Utils.getEl('fileCols').textContent = this.previewData.headers.length;
        
        // Show validation status
        const validationEl = Utils.getEl('fileValidation');
        if (validation.valid) {
            validationEl.innerHTML = `
                <div style="color: var(--success); padding: 0.5rem; background: rgba(76, 175, 80, 0.1); border-radius: 6px;">
                    ✓ File meets all requirements
                </div>
            `;
        }
        
        Utils.getEl('fileInfo').style.display = 'block';
        Utils.getEl('configCard').style.display = 'block';
    },

    /**
     * Populate column selectors
     */
    populateColumnSelectors() {
        const dateSelect = Utils.getEl('dateColumn');
        const valueSelect = Utils.getEl('valueColumn');
        const exoContainer = Utils.getEl('exogenousVariables');

        // Clear existing options (except first)
        dateSelect.innerHTML = '<option value="">Select date column</option>';
        valueSelect.innerHTML = '<option value="">Select value column</option>';
        exoContainer.innerHTML = '';

        // Add options
        this.previewData.headers.forEach((header, index) => {
            const option1 = new Option(header, index);
            const option2 = new Option(header, index);
            dateSelect.add(option1);
            valueSelect.add(option2);

            // Add exogenous checkbox
            const checkbox = `
                <label style="display: flex; align-items: center; gap: 0.5rem; padding: 0.75rem 1rem; background: rgba(30, 90, 150, 0.1); border: 1px solid var(--border-color); border-radius: 8px; cursor: pointer; transition: all 0.3s;">
                    <input type="checkbox" value="${index}" class="exo-checkbox" style="width: 18px; height: 18px; cursor: pointer;">
                    <span style="color: var(--text-secondary);">${header}</span>
                </label>
            `;
            exoContainer.innerHTML += checkbox;
        });

        // Add change listeners to exogenous checkboxes
        Utils.qsa('.exo-checkbox').forEach(cb => {
            cb.addEventListener('change', () => this.validateForm());
        });
    },

    /**
     * Auto-detect date and value columns
     */
    autoDetectColumns() {
        const firstRow = this.previewData.rows[0] || [];
        const dateColIndex = Utils.detectDateColumn(this.previewData.headers, firstRow);
        const valueColIndex = Utils.detectValueColumn(this.previewData.headers, firstRow, dateColIndex);

        Utils.getEl('dateColumn').value = dateColIndex;
        Utils.getEl('valueColumn').value = valueColIndex;

        // Set default pipeline name
        const timestamp = new Date().toISOString().split('T')[0];
        Utils.getEl('pipelineName').value = `Pipeline_${timestamp}`;

        console.log('Auto-detected columns:', { dateColIndex, valueColIndex });
    },

    /**
     * Update data preview table
     */
    updatePreview() {
        const dateCol = parseInt(Utils.getEl('dateColumn').value);
        const valueCol = parseInt(Utils.getEl('valueColumn').value);

        if (isNaN(dateCol) || isNaN(valueCol)) return;

        const thead = Utils.getEl('previewHead');
        const tbody = Utils.getEl('previewBody');

        // Build header
        thead.innerHTML = `
            <tr>
                <th style="background: rgba(33, 150, 243, 0.2);">${this.previewData.headers[dateCol]}</th>
                <th style="background: rgba(76, 175, 80, 0.2);">${this.previewData.headers[valueCol]}</th>
            </tr>
        `;

        // Build body (first N rows)
        const previewRows = this.previewData.rows.slice(0, this.maxPreviewRows);
        tbody.innerHTML = previewRows.map((row, idx) => `
            <tr>
                <td>${row[dateCol] || '-'}</td>
                <td>${row[valueCol] || '-'}</td>
            </tr>
        `).join('');

        if (this.previewData.rows.length > this.maxPreviewRows) {
            tbody.innerHTML += `
                <tr style="background: rgba(33, 150, 243, 0.05);">
                    <td colspan="2" style="text-align: center; font-style: italic; color: var(--text-secondary);">
                        ... and ${this.previewData.rows.length - this.maxPreviewRows} more rows
                    </td>
                </tr>
            `;
        }
    },

    /**
     * Validate form inputs
     */
    validateForm() {
        const dateCol = Utils.getEl('dateColumn').value;
        const valueCol = Utils.getEl('valueColumn').value;
        const pipelineName = Utils.getEl('pipelineName').value.trim();
        const uploadBtn = Utils.getEl('uploadBtn');

        const isValid = dateCol && valueCol && pipelineName && dateCol !== valueCol;
        
        uploadBtn.disabled = !isValid;

        if (dateCol === valueCol && dateCol) {
            Utils.showToast('Date and Value columns must be different', 'warning');
        }

        return isValid;
    },

    /**
 * Upload and process data with comprehensive validation
 */
    async uploadData() {
    // Final validation
    if (!this.validateForm()) {
        Utils.showToast('Please fill all required fields', 'warning');
        return;
    }

    const dateCol = parseInt(Utils.getEl('dateColumn').value);
    const valueCol = parseInt(Utils.getEl('valueColumn').value);
    const frequency = Utils.getEl('frequency').value;
    const pipelineName = Utils.getEl('pipelineName').value.trim();

    // Get selected exogenous variables
    const exogenous = Array.from(Utils.qsa('.exo-checkbox:checked'))
        .map(cb => parseInt(cb.value))
        .filter(val => val !== dateCol && val !== valueCol);

    // Prepare config
    const config = {
        date_column: this.previewData.headers[dateCol],
        value_column: this.previewData.headers[valueCol],
        frequency: frequency,
        pipeline_name: pipelineName,
        exogenous_columns: exogenous.map(idx => this.previewData.headers[idx])
    };

    console.log('Upload config:', config);

    // Upload
    Utils.showLoading('Uploading and validating data...');
    
    try {
        const result = await API.uploadFile(this.selectedFile, config);
        
        console.log('Upload result:', result);
        
        Utils.hideLoading();

        if (result.success && result.data) {
            // Comprehensive response validation
            const requiredFields = ['pipeline_id', 'date_column', 'value_column'];
            const missingFields = requiredFields.filter(field => !result.data[field]);
            
            if (missingFields.length > 0) {
                Utils.showToast(`Invalid API response: missing ${missingFields.join(', ')}`, 'error');
                console.error('Missing fields in response:', missingFields, result.data);
                return;
            }

            // Construct validated pipeline data with defaults
            const pipelineData = {
                pipeline_id: result.data.pipeline_id,
                pipeline_name: result.data.pipeline_name || pipelineName || 'Unnamed Pipeline',
                date_column: result.data.date_column || config.date_column,
                value_column: result.data.value_column || config.value_column,
                frequency: result.data.frequency || config.frequency,
                exogenous_columns: result.data.exogenous_columns || config.exogenous_columns || [],
                data_shape: result.data.data_shape || { 
                    rows: this.previewData.rows.length, 
                    columns: this.previewData.headers.length 
                },
                columns: result.data.columns || this.previewData.headers,
                created_at: new Date().toISOString()
            };
            
            // Save to memory
            Utils.saveToMemory('current_pipeline', pipelineData);
            
            Utils.showToast('Data uploaded successfully!', 'success');
            
            console.log('Pipeline saved to memory:', pipelineData);
            
            // Navigate to exploration after delay
            setTimeout(() => {
                Navigation.navigateTo('explore');
            }, 1500);
        } else {
            const errorMsg = result.error || 'Unknown error occurred';
            Utils.showToast('Upload failed: ' + errorMsg, 'error');
            console.error('Upload failed:', result);
        }
    } catch (error) {
        Utils.hideLoading();
        Utils.showToast('Upload failed: ' + error.message, 'error');
        console.error('Upload error:', error);
    }
},
    /**
     * Reset form to initial state
     */
    reset() {
        this.selectedFile = null;
        this.previewData = null;
        
        Utils.getEl('fileInfo').style.display = 'none';
        Utils.getEl('configCard').style.display = 'none';
        Utils.getEl('fileInput').value = '';
        Utils.getEl('uploadBtn').disabled = true;
        
        // Scroll to top
        window.scrollTo({ top: 0, behavior: 'smooth' });
        
        Utils.showToast('Form reset', 'info');
    },

    /**
     * Refresh/reload page
     */
    refresh() {
        this.render();
    }
};

// Make UploadPage available globally
window.UploadPage = UploadPage;