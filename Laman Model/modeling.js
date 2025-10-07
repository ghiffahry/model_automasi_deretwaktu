// ==================== MODELING PAGE (FIXED) ====================

const ModelingPage = {
    pipelineData: null,
    selectedModels: [],
    trainedModels: {},
    forecastResults: {},
    residualAnalysis: {},
    
    normalizeModelId(modelId) {
        return modelId.toUpperCase();
    },
    
    render() {
        const container = Utils.getEl('mainContainer');
        
        this.pipelineData = Utils.getFromMemory('current_pipeline');
        
        if (!this.pipelineData) {
            container.innerHTML = `
                <div class="page active">
                    <div class="card" style="text-align: center; padding: 4rem 2rem;">
                        <div style="font-size: 5rem; margin-bottom: 2rem;">🤖</div>
                        <h2 style="color: var(--light-blue); margin-bottom: 1rem;">No Pipeline Found</h2>
                        <p style="color: var(--text-secondary); margin-bottom: 2rem;">
                            Please upload and explore data first
                        </p>
                        <button class="btn btn-primary" onclick="Navigation.navigateTo('upload')">
                            Start New Pipeline
                        </button>
                    </div>
                </div>
            `;
            return;
        }

        container.innerHTML = `
            <div class="page active" id="modelingPage">
                <h1 style="margin-bottom: 2rem; animation: fadeIn 0.5s ease;">
                    🤖 Model Training & Forecasting
                </h1>

                <!-- Model Selection -->
                <div class="card" style="animation: slideInUp 0.6s ease;">
                    <h2 class="card-title">Select Models</h2>
                    <p class="card-subtitle">
                        Choose one or more models to train on your data. Each model has different strengths.
                    </p>
                    <div class="model-grid" id="modelGrid"></div>
                    
                    <!-- Training Options -->
                    <div style="margin-top: 2rem; padding-top: 2rem; border-top: 1px solid var(--border-color);">
                        <h4 style="color: var(--light-blue); margin-bottom: 1rem;">Training Options</h4>
                        <div class="grid-2">
                            <label style="display: flex; align-items: center; gap: 0.5rem; cursor: pointer;">
                                <input type="checkbox" id="arimaGridSearch" checked style="width: 18px; height: 18px;">
                                <span style="color: var(--text-secondary);">Enable ARIMA Grid Search</span>
                            </label>
                            <label style="display: flex; align-items: center; gap: 0.5rem; cursor: pointer;">
                                <input type="checkbox" id="sarimaGridSearch" checked style="width: 18px; height: 18px;">
                                <span style="color: var(--text-secondary);">Enable SARIMA Grid Search</span>
                            </label>
                        </div>
                    </div>

                    <div style="margin-top: 2rem; text-align: right;">
                        <button class="btn btn-primary btn-lg" id="trainModelsBtn" disabled>
                            🚀 Train Selected Models
                        </button>
                    </div>
                </div>

                <!-- Training Progress -->
                <div class="card" id="trainingProgress" style="display: none;">
                    <h2 class="card-title">Training Progress</h2>
                    <div class="progress-container">
                        <div class="progress-bar">
                            <div class="progress-fill" id="progressFill" style="width: 0%"></div>
                        </div>
                        <div class="progress-text" id="progressText">Initializing...</div>
                    </div>
                    <div id="progressDetails" style="margin-top: 1rem; color: var(--text-secondary); font-size: 0.9rem;"></div>
                </div>

                <!-- Model Results -->
                <div id="modelResults" style="display: none;">
                    <div class="card">
                        <h2 class="card-title">Model Performance</h2>
                        <div style="overflow-x: auto;">
                            <table id="performanceTable">
                                <thead>
                                    <tr>
                                        <th>Model</th>
                                        <th>RMSE</th>
                                        <th>MAE</th>
                                        <th>MAPE</th>
                                        <th>R²</th>
                                        <th>Training Time</th>
                                        <th>Actions</th>
                                    </tr>
                                </thead>
                                <tbody id="performanceBody"></tbody>
                            </table>
                        </div>
                    </div>

                    <!-- Model Comparison Chart -->
                    <div class="card">
                        <h2 class="card-title">Performance Comparison</h2>
                        <div class="chart-wrapper" id="performanceComparisonChart"></div>
                    </div>

                    <!-- Residual Analysis Section -->
                    <div class="card" id="residualAnalysisSection">
                        <h2 class="card-title">Residual Diagnostics</h2>
                        <p class="card-subtitle">
                            Analyze residuals to check model assumptions (normality, homoscedasticity, no autocorrelation)
                        </p>
                        <div style="margin-bottom: 1.5rem;">
                            <label class="form-label">Select Model for Analysis:</label>
                            <select class="form-control" id="residualModelSelect">
                                <option value="">Choose a model...</option>
                            </select>
                        </div>
                        
                        <div id="residualContent" style="display: none;">
                            <!-- Loading Indicator -->
                            <div id="residualLoading" style="text-align: center; padding: 2rem; display: none;">
                                <div class="spinner"></div>
                                <p style="color: var(--text-secondary); margin-top: 1rem;">Loading residual analysis...</p>
                            </div>

                            <!-- Assumption Tests -->
                            <div id="assumptionTestsContainer" style="margin-bottom: 2rem; display: none;">
                                <h3 style="color: var(--light-blue); margin-bottom: 1rem;">Statistical Tests</h3>
                                <div class="grid-3" id="assumptionTests"></div>
                            </div>

                            <!-- Fitted vs Actual -->
                            <div id="fittedVsActualContainer" style="display: none; margin-bottom: 2rem;">
                                <h3 style="color: var(--light-blue); margin-bottom: 1rem;">Model Fit Visualization</h3>
                                <div class="chart-wrapper" id="fittedVsActualChart"></div>
                            </div>

                            <!-- Diagnostic Plots -->
                            <div id="diagnosticPlotsContainer" style="display: none; margin-bottom: 2rem;">
                                <h3 style="color: var(--light-blue); margin-bottom: 1rem;">Diagnostic Plots</h3>
                                
                                <!-- Residual Time Series -->
                                <div style="margin-bottom: 2rem;">
                                    <div class="chart-wrapper" id="residualTimeChart"></div>
                                </div>
                                
                                <!-- Histogram and ACF side by side -->
                                <div class="grid-2" style="gap: 1.5rem;">
                                    <div class="chart-wrapper" id="residualHistChart"></div>
                                    <div class="chart-wrapper" id="residualACFChart"></div>
                                </div>
                            </div>

                            <!-- Error Message -->
                            <div id="residualError" style="display: none; text-align: center; padding: 2rem; background: rgba(244, 67, 54, 0.1); border-radius: 8px;">
                                <p style="color: var(--error); font-size: 1.1rem; margin: 0;" id="residualErrorText">
                                    ⚠️ Error loading residual analysis
                                </p>
                            </div>

                            <!-- No Data Message -->
                            <div id="residualNoData" style="display: none; text-align: center; padding: 2rem; background: rgba(255, 152, 0, 0.1); border-radius: 8px;">
                                <p style="color: var(--warning); font-size: 1.1rem; margin: 0;">
                                    ℹ️ Residual analysis not available for this model
                                </p>
                                <p style="color: var(--text-secondary); margin-top: 0.5rem; font-size: 0.9rem;">
                                    Only ARIMA/ARIMAX/SARIMA/SARIMAX models support residual diagnostics
                                </p>
                            </div>
                        </div>
                    </div>

                    <!-- Forecasting Section -->
                    <div class="card">
                        <h2 class="card-title">Generate Forecast</h2>
                        <div class="grid-2" style="align-items: end;">
                            <div class="form-group">
                                <label class="form-label">Select Model</label>
                                <select class="form-control" id="forecastModelSelect">
                                    <option value="">Choose a model...</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label class="form-label">Forecast Steps</label>
                                <input type="number" class="form-control" id="forecastSteps" 
                                       value="30" min="1" max="365">
                            </div>
                        </div>
                        <div style="text-align: right; margin-top: 1rem;">
                            <button class="btn btn-primary" id="generateForecastBtn">
                                📊 Generate Forecast
                            </button>
                        </div>
                    </div>

                    <!-- Forecast Visualization -->
                    <div class="card" id="forecastCard" style="display: none;">
                        <h2 class="card-title">Forecast Results</h2>
                        <div class="chart-wrapper" id="forecastChart"></div>
                        
                        <!-- Forecast Statistics -->
                        <div style="margin-top: 2rem;">
                            <h3 style="color: var(--light-blue); margin-bottom: 1rem;">Forecast Statistics</h3>
                            <div class="grid-4" id="forecastStats"></div>
                        </div>

                        <div style="margin-top: 2rem; display: flex; gap: 1rem; justify-content: flex-end;">
                            <button class="btn btn-secondary" onclick="ModelingPage.downloadForecast('csv')">
                                Download CSV
                            </button>
                            <button class="btn btn-secondary" onclick="ModelingPage.downloadForecast('json')">
                                Download JSON
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;

        this.renderModelGrid();
        this.setupEventListeners();
        
        // Restore trained models if any
        const savedModels = Utils.getFromMemory('trained_models');
        if (savedModels) {
            this.trainedModels = savedModels;
            this.displayModelResults();
        }
    },

    renderModelGrid() {
        const grid = Utils.getEl('modelGrid');
        const models = [
            { id: 'arima', name: 'ARIMA', icon: '📈', description: 'AutoRegressive Integrated Moving Average' },
            { id: 'sarima', name: 'SARIMA', icon: '🔄', description: 'Seasonal ARIMA for periodic patterns' },
            { id: 'transformer', name: 'Transformer', icon: '🤖', description: 'Deep learning time series model' }
        ];
        
        grid.innerHTML = models.map(model => `
            <div class="model-option" data-model="${model.id}">
                <div class="model-icon-large">${model.icon}</div>
                <div class="model-name">${model.name}</div>
                <div class="model-desc">${model.description}</div>
            </div>
        `).join('');

        Utils.qsa('.model-option').forEach(option => {
            option.addEventListener('click', () => this.toggleModel(option));
        });

        const arimaOption = Utils.qs('[data-model="arima"]');
        if (arimaOption) this.toggleModel(arimaOption);
    },

    toggleModel(element) {
        const modelId = element.dataset.model;
        
        if (element.classList.contains('selected')) {
            element.classList.remove('selected');
            this.selectedModels = this.selectedModels.filter(m => m !== modelId);
        } else {
            element.classList.add('selected');
            this.selectedModels.push(modelId);
        }

        Utils.getEl('trainModelsBtn').disabled = this.selectedModels.length === 0;
    },

    setupEventListeners() {
        const trainBtn = Utils.getEl('trainModelsBtn');
        if (trainBtn) {
            trainBtn.addEventListener('click', () => this.trainModels());
        }

        const forecastBtn = Utils.getEl('generateForecastBtn');
        if (forecastBtn) {
            forecastBtn.addEventListener('click', () => this.generateForecast());
        }
        
        const residualSelect = Utils.getEl('residualModelSelect');
        if (residualSelect) {
            residualSelect.addEventListener('change', (e) => {
                this.loadResidualAnalysis(e.target.value);
            });
        }
    },

    async trainModels() {
        if (this.selectedModels.length === 0) {
            Utils.showToast('Please select at least one model', 'warning');
            return;
        }

        const progressCard = Utils.getEl('trainingProgress');
        const progressFill = Utils.getEl('progressFill');
        const progressText = Utils.getEl('progressText');
        
        progressCard.style.display = 'block';
        progressFill.style.width = '0%';
        progressText.textContent = 'Starting training...';

        const options = {
            arimaGrid: Utils.getEl('arimaGridSearch').checked,
            sarimaGrid: Utils.getEl('sarimaGridSearch').checked
        };

        try {
            progressFill.style.width = '30%';
            progressText.textContent = 'Training models in parallel...';

            const normalizedModels = this.selectedModels.map(m => this.normalizeModelId(m));
            
            const result = await API.trainModels(
                this.pipelineData.pipeline_id,
                normalizedModels,
                options
            );

            progressFill.style.width = '100%';
            progressText.textContent = 'Training completed!';

            if (result.success && result.data && result.data.models) {
                Object.assign(this.trainedModels, result.data.models);
                
                Utils.saveToMemory('trained_models', this.trainedModels);

                setTimeout(() => {
                    progressCard.style.display = 'none';
                    this.displayModelResults();
                    Utils.showToast('Models trained successfully!', 'success');
                }, 1000);
            } else {
                throw new Error(result.error || 'Training failed');
            }
        } catch (error) {
            progressCard.style.display = 'none';
            Utils.showToast('Training failed: ' + error.message, 'error');
            console.error('Training error:', error);
        }
    },

    displayModelResults() {
        Utils.getEl('modelResults').style.display = 'block';
        
        const tbody = Utils.getEl('performanceBody');
        const forecastSelect = Utils.getEl('forecastModelSelect');
        const residualSelect = Utils.getEl('residualModelSelect');
        
        tbody.innerHTML = '';
        forecastSelect.innerHTML = '<option value="">Choose a model...</option>';
        residualSelect.innerHTML = '<option value="">Choose a model...</option>';

        Object.entries(this.trainedModels).forEach(([modelName, metrics]) => {
            tbody.innerHTML += `
                <tr>
                    <td><strong style="color: var(--light-blue);">${modelName}</strong></td>
                    <td>${Utils.formatNumber(metrics.rmse)}</td>
                    <td>${Utils.formatNumber(metrics.mae)}</td>
                    <td>${Utils.formatPercent(metrics.mape)}</td>
                    <td>${Utils.formatNumber(metrics.r2, 3)}</td>
                    <td>${Utils.formatNumber(metrics.training_time, 1)}s</td>
                    <td>
                        <button class="btn btn-sm btn-secondary" onclick="ModelingPage.showModelDetails('${modelName}')">
                            Details
                        </button>
                    </td>
                </tr>
            `;

            forecastSelect.innerHTML += `<option value="${modelName}">${modelName}</option>`;
            
            // Check if model supports residual analysis
            const modelType = modelName.replace(/X$/, '');
            if (modelType === 'ARIMA' || modelType === 'SARIMA') {
                residualSelect.innerHTML += `<option value="${modelName}">${modelName}</option>`;
            }
        });

        this.renderPerformanceComparison();
        Utils.scrollToElement('modelResults');
    },

    renderPerformanceComparison() {
        Charts.renderMetricsComparison('performanceComparisonChart', this.trainedModels);
    },

    async loadResidualAnalysis(modelName) {
        const content = Utils.getEl('residualContent');
        const loading = Utils.getEl('residualLoading');
        const assumptionContainer = Utils.getEl('assumptionTestsContainer');
        const fittedContainer = Utils.getEl('fittedVsActualContainer');
        const diagnosticContainer = Utils.getEl('diagnosticPlotsContainer');
        const noDataMsg = Utils.getEl('residualNoData');
        const errorMsg = Utils.getEl('residualError');
        const errorText = Utils.getEl('residualErrorText');
        
        // Reset all sections
        content.style.display = 'none';
        loading.style.display = 'none';
        assumptionContainer.style.display = 'none';
        fittedContainer.style.display = 'none';
        diagnosticContainer.style.display = 'none';
        noDataMsg.style.display = 'none';
        errorMsg.style.display = 'none';
        
        if (!modelName) {
            return;
        }

        // Check if model supports residual analysis
        const supportedModels = ['ARIMA', 'SARIMA'];
        const baseModelName = modelName.replace(/X$/, '');
        
        if (!supportedModels.includes(baseModelName)) {
            content.style.display = 'block';
            noDataMsg.style.display = 'block';
            return;
        }

        content.style.display = 'block';
        loading.style.display = 'block';
        
        try {
            console.log(`Loading residual analysis for ${modelName}...`);
            
            const result = await API.getResidualAnalysis(
                this.pipelineData.pipeline_id,
                modelName
            );
            
            loading.style.display = 'none';
            
            if (result.success && result.data) {
                console.log('Residual analysis data received:', result.data);
                
                let hasData = false;
                
                // Display assumption tests
                if (result.data.assumption_tests && Object.keys(result.data.assumption_tests).length > 0) {
                    this.displayAssumptionTests(result.data.assumption_tests);
                    assumptionContainer.style.display = 'block';
                    hasData = true;
                }
                
                // Display fitted vs actual
                if (result.data.fitted_data && result.data.fitted_data.dates && result.data.fitted_data.dates.length > 0) {
                    this.displayFittedVsActual(result.data.fitted_data);
                    fittedContainer.style.display = 'block';
                    hasData = true;
                }
                
                // Display diagnostic plots
                if (result.data.residual_plot || result.data.histogram || result.data.acf) {
                    this.displayDiagnosticPlots(result.data);
                    diagnosticContainer.style.display = 'block';
                    hasData = true;
                }
                
                if (!hasData) {
                    noDataMsg.style.display = 'block';
                    Utils.showToast('No residual data available for this model', 'warning');
                }
            } else {
                errorMsg.style.display = 'block';
                const errorMessage = result.error || 'Unknown error occurred';
                errorText.textContent = `⚠️ ${errorMessage}`;
                Utils.showToast('Failed to load residual analysis: ' + errorMessage, 'error');
                console.error('API Error:', result);
            }
        } catch (error) {
            loading.style.display = 'none';
            errorMsg.style.display = 'block';
            errorText.textContent = `⚠️ Error: ${error.message}`;
            Utils.showToast('Error loading residual analysis: ' + error.message, 'error');
            console.error('Residual analysis error:', error);
        }
    },

    displayAssumptionTests(tests) {
        const container = Utils.getEl('assumptionTests');
        if (!container || !tests) return;

        const testCards = [];

        // Normality test
        if (tests.normality && !tests.normality.error) {
            const normal = tests.normality;
            const isNormal = normal.is_normal || false;
            testCards.push(`
                <div class="stat-card" style="background: ${isNormal ? 'rgba(76, 175, 80, 0.1)' : 'rgba(244, 67, 54, 0.1)'};">
                    <div class="stat-icon">${isNormal ? '✅' : '❌'}</div>
                    <div class="stat-label">Normality</div>
                    <div class="stat-value" style="font-size: 0.9rem;">${normal.conclusion || 'N/A'}</div>
                    <div style="margin-top: 0.5rem; font-size: 0.85rem; color: var(--text-secondary);">
                        ${normal.test || 'Jarque-Bera Test'}<br>
                        p-value: ${Utils.formatNumber(normal.p_value, 4)}
                    </div>
                </div>
            `);
        }

        // Heteroscedasticity test
        if (tests.heteroscedasticity && !tests.heteroscedasticity.error) {
            const homo = tests.heteroscedasticity;
            const isHomo = homo.is_homoscedastic || false;
            testCards.push(`
                <div class="stat-card" style="background: ${isHomo ? 'rgba(76, 175, 80, 0.1)' : 'rgba(244, 67, 54, 0.1)'};">
                    <div class="stat-icon">${isHomo ? '✅' : '❌'}</div>
                    <div class="stat-label">Homoscedasticity</div>
                    <div class="stat-value" style="font-size: 0.9rem;">${homo.conclusion || 'N/A'}</div>
                    <div style="margin-top: 0.5rem; font-size: 0.85rem; color: var(--text-secondary);">
                        ${homo.test || 'ARCH LM Test'}<br>
                        p-value: ${Utils.formatNumber(homo.p_value, 4)}
                    </div>
                </div>
            `);
        }

        // White noise test
        if (tests.white_noise && !tests.white_noise.error) {
            const wn = tests.white_noise;
            const isWN = wn.is_white_noise || false;
            testCards.push(`
                <div class="stat-card" style="background: ${isWN ? 'rgba(76, 175, 80, 0.1)' : 'rgba(244, 67, 54, 0.1)'};">
                    <div class="stat-icon">${isWN ? '✅' : '❌'}</div>
                    <div class="stat-label">White Noise</div>
                    <div class="stat-value" style="font-size: 0.9rem;">${wn.conclusion || 'N/A'}</div>
                    <div style="margin-top: 0.5rem; font-size: 0.85rem; color: var(--text-secondary);">
                        ${wn.test || 'Ljung-Box Test'}<br>
                        min p-value: ${Utils.formatNumber(wn.min_p_value, 4)}
                    </div>
                </div>
            `);
        }

        if (testCards.length === 0) {
            testCards.push(`
                <div class="stat-card" style="background: rgba(244, 67, 54, 0.1);">
                    <div class="stat-icon">⚠️</div>
                    <div class="stat-label">No Test Data</div>
                    <div class="stat-value" style="font-size: 0.9rem;">N/A</div>
                </div>
            `);
        }

        container.innerHTML = testCards.join('');
    },

    displayFittedVsActual(data) {
        if (!data || !data.dates || !data.actual || !data.fitted) {
            console.warn('Incomplete fitted data:', data);
            return;
        }

        Charts.renderComparison('fittedVsActualChart', [
            {
                name: 'Actual',
                dates: data.dates,
                values: data.actual,
                color: '#2196F3'
            },
            {
                name: 'Fitted',
                dates: data.dates,
                values: data.fitted,
                color: '#4CAF50'
            }
        ]);
    },

    displayDiagnosticPlots(data) {
        // Residual time series plot
        if (data.residual_plot && data.residual_plot.dates && data.residual_plot.residuals) {
            Charts.renderResidualPlot('residualTimeChart', data.residual_plot);
        }
        
        // Histogram with normal distribution
        if (data.histogram && data.histogram.bins && data.histogram.counts) {
            Charts.renderResidualHistogram('residualHistChart', data.histogram);
        }
        
        // ACF plot
        if (data.acf && data.acf.lags && data.acf.acf) {
            Charts.renderACFPlot('residualACFChart', data.acf);
        }
    },

    async generateForecast() {
        const modelName = Utils.getEl('forecastModelSelect').value;
        const steps = parseInt(Utils.getEl('forecastSteps').value);

        if (!modelName) {
            Utils.showToast('Please select a model', 'warning');
            return;
        }

        if (!steps || steps < 1 || steps > 365) {
            Utils.showToast('Steps must be between 1 and 365', 'warning');
            return;
        }

        Utils.showLoading('Generating forecast...');
        
        try {
            const result = await API.generateForecast(
                this.pipelineData.pipeline_id,
                modelName,
                steps
            );

            Utils.hideLoading();

            if (result.success) {
                this.forecastResults = result.data;
                Utils.saveToMemory('forecast_results', result.data);
                
                this.displayForecast();
                Utils.showToast('Forecast generated successfully!', 'success');
            } else {
                throw new Error(result.error || 'Forecast generation failed');
            }
        } catch (error) {
            Utils.hideLoading();
            Utils.showToast('Forecast failed: ' + error.message, 'error');
            console.error('Forecast error:', error);
        }
    },

    displayForecast() {
        const forecastCard = Utils.getEl('forecastCard');
        forecastCard.style.display = 'block';

        Charts.renderForecast('forecastChart', this.forecastResults);
        this.displayForecastStats();
        Utils.scrollToElement('forecastCard');
    },

    displayForecastStats() {
        const container = Utils.getEl('forecastStats');
        if (!container) return;

        const values = this.forecastResults.forecast_values || [];
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const min = Math.min(...values);
        const max = Math.max(...values);
        const trend = values[values.length - 1] > values[0] ? 'Increasing' : 'Decreasing';

        container.innerHTML = `
            <div class="stat-card">
                <div class="stat-icon">📊</div>
                <div class="stat-value">${Utils.formatNumber(mean)}</div>
                <div class="stat-label">Mean Forecast</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">⬇️</div>
                <div class="stat-value">${Utils.formatNumber(min)}</div>
                <div class="stat-label">Minimum</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">⬆️</div>
                <div class="stat-value">${Utils.formatNumber(max)}</div>
                <div class="stat-label">Maximum</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">📈</div>
                <div class="stat-value">${trend}</div>
                <div class="stat-label">Trend</div>
            </div>
        `;
    },

    downloadForecast(format) {
        if (!this.forecastResults || !this.forecastResults.forecast_dates) {
            Utils.showToast('No forecast data available', 'warning');
            return;
        }

        if (format === 'csv') {
            const csv = this.convertToCSV();
            Utils.downloadCSV(csv, 'forecast_results.csv');
        } else if (format === 'json') {
            Utils.downloadJSON(this.forecastResults, 'forecast_results.json');
        }
    },

    convertToCSV() {
        const headers = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound'];
        const rows = this.forecastResults.forecast_dates.map((date, i) => [
            date,
            this.forecastResults.forecast_values[i],
            this.forecastResults.lower_bound[i],
            this.forecastResults.upper_bound[i]
        ]);

        return [
            headers.join(','),
            ...rows.map(row => row.join(','))
        ].join('\n');
    },

    showModelDetails(modelName) {
        const metrics = this.trainedModels[modelName];
        const details = `
Model: ${modelName}

Performance Metrics:
- RMSE: ${Utils.formatNumber(metrics.rmse)}
- MAE: ${Utils.formatNumber(metrics.mae)}
- MAPE: ${Utils.formatPercent(metrics.mape)}
- R²: ${Utils.formatNumber(metrics.r2, 3)}

Training Information:
- Training Time: ${Utils.formatNumber(metrics.training_time, 1)} seconds
- Sample Size: ${metrics.sample_size || 'N/A'}

${metrics.parameters ? `\nModel Parameters:\n${JSON.stringify(metrics.parameters, null, 2)}` : ''}
        `.trim();

        alert(details);
    }
};

window.ModelingPage = ModelingPage;
console.log('✓ ModelingPage loaded with enhanced error handling');