// ==================== ENHANCED EXPLORATION PAGE - COMPLETE VERSION ====================

const ExplorePage = {
    pipelineData: null,
    explorationResults: null,
    detailedResults: null,
    
    /**
     * Render exploration page with enhanced features
     */
    render() {
        const container = Utils.getEl('mainContainer');
        
        this.pipelineData = Utils.getFromMemory('current_pipeline');
        
        if (!this.pipelineData) {
            container.innerHTML = `
                <div class="page active">
                    <div class="card" style="text-align: center; padding: 4rem 2rem;">
                        <div style="font-size: 5rem; margin-bottom: 2rem;">📊</div>
                        <h2 style="color: var(--light-blue); margin-bottom: 1rem;">No Data Available</h2>
                        <p style="color: var(--text-secondary); margin-bottom: 2rem;">
                            Please upload data first to begin exploration
                        </p>
                        <button class="btn btn-primary" onclick="Navigation.navigateTo('upload')">
                            Upload Data
                        </button>
                    </div>
                </div>
            `;
            return;
        }

        container.innerHTML = `
            <div class="page active" id="explorePage">
                <h1 style="margin-bottom: 2rem; animation: fadeIn 0.5s ease;">
                    🔍 Exploratory Data Analysis
                </h1>

                <!-- Pipeline Info Card -->
                <div class="card" style="animation: slideInUp 0.6s ease;">
                    <h2 class="card-title">Pipeline Information</h2>
                    <div class="info-grid">
                        <div class="info-item">
                            <span class="info-item-label">Pipeline Name:</span>
                            <span class="info-item-value">${this.pipelineData.pipeline_name || 'N/A'}</span>
                        </div>
                        <div class="info-item">
                            <span class="info-item-label">Pipeline ID:</span>
                            <span class="info-item-value">${this.pipelineData.pipeline_id || 'N/A'}</span>
                        </div>
                        <div class="info-item">
                            <span class="info-item-label">Date Column:</span>
                            <span class="info-item-value">${this.pipelineData.date_column || 'N/A'}</span>
                        </div>
                        <div class="info-item">
                            <span class="info-item-label">Value Column:</span>
                            <span class="info-item-value">${this.pipelineData.value_column || 'N/A'}</span>
                        </div>
                        <div class="info-item">
                            <span class="info-item-label">Rows:</span>
                            <span class="info-item-value">${this.pipelineData.data_shape?.rows || 'N/A'}</span>
                        </div>
                        <div class="info-item">
                            <span class="info-item-label">Columns:</span>
                            <span class="info-item-value">${this.pipelineData.data_shape?.columns || 'N/A'}</span>
                        </div>
                    </div>
                    <div style="margin-top: 1.5rem; display: flex; gap: 1rem; justify-content: flex-end; flex-wrap: wrap;">
                        <button class="btn btn-secondary" id="runBasicExplorationBtn">
                            📈 Basic Analysis
                        </button>
                        <button class="btn btn-primary" id="runDetailedExplorationBtn">
                            🔬 Detailed Analysis
                        </button>
                        <button class="btn btn-primary" id="differencingBtn">
                            📊 Analyze Differencing & ACF/PACF
                        </button>
                    </div>
                </div>

                <!-- Basic Results Container -->
                <div id="basicResults" style="display: none;">
                    <!-- Basic Statistics -->
                    <div class="card" style="animation: slideInUp 0.7s ease;">
                        <h2 class="card-title">Basic Statistics</h2>
                        <div class="grid-4" id="basicStats"></div>
                    </div>

                    <!-- Time Series Plot -->
                    <div class="card" style="animation: slideInUp 0.8s ease;">
                        <h2 class="card-title">Time Series Visualization</h2>
                        <div class="chart-wrapper" id="timeSeriesChart"></div>
                    </div>

                    <!-- Decomposition -->
                    <div class="grid-2">
                        <div class="card" style="animation: slideInUp 0.9s ease;">
                            <h2 class="card-title">Trend Component</h2>
                            <div class="chart-wrapper" id="trendChart"></div>
                        </div>
                        <div class="card" style="animation: slideInUp 1s ease;">
                            <h2 class="card-title">Seasonal Component</h2>
                            <div class="chart-wrapper" id="seasonalityChart"></div>
                        </div>
                    </div>

                    <!-- Statistical Tests -->
                    <div class="card" style="animation: slideInUp 1.1s ease;">
                        <h2 class="card-title">Statistical Tests</h2>
                        <div id="statisticalTests"></div>
                    </div>

                    <!-- ACF & Distribution -->
                    <div class="grid-2">
                        <div class="card" style="animation: slideInUp 1.2s ease;">
                            <h2 class="card-title">Value Distribution</h2>
                            <div class="chart-wrapper" id="distributionChart"></div>
                        </div>
                        <div class="card" style="animation: slideInUp 1.3s ease;">
                            <h2 class="card-title">Autocorrelation Function</h2>
                            <div class="chart-wrapper" id="acfChart"></div>
                        </div>
                    </div>
                </div>

                <!-- Detailed Results Container -->
                <div id="detailedResults" style="display: none;">
                    <!-- Outlier Detection -->
                    <div class="card">
                        <h2 class="card-title">Outlier Detection</h2>
                        <div class="grid-2" id="outlierStats"></div>
                    </div>

                    <!-- Differencing Analysis -->
                    <div class="card">
                        <h2 class="card-title">Differencing Analysis</h2>
                        <div id="differencingInfo"></div>
                    </div>

                    <!-- Seasonality Strength -->
                    <div class="card">
                        <h2 class="card-title">Seasonality Analysis</h2>
                        <div id="seasonalityInfo"></div>
                    </div>

                    <!-- Comprehensive EDA Report -->
                    <div class="card">
                        <h2 class="card-title">Comprehensive Report</h2>
                        <div id="edaReport"></div>
                    </div>
                </div>

                <!-- Differencing & ACF/PACF Analysis Card -->
                <div class="card" id="differencingCard" style="display: none; animation: slideInUp 0.5s ease;">
                    <h2 class="card-title">📊 Differencing & ACF/PACF Analysis</h2>
                    <p class="card-subtitle" style="margin-bottom: 2rem;">
                        Analyze stationarity before and after differencing with Augmented Dickey-Fuller test, 
                        ACF and PACF plots to determine optimal ARIMA parameters (p, d, q)
                    </p>
                    
                    <!-- Stationarity Tests -->
                    <div style="margin-bottom: 2rem;">
                        <h3 style="color: var(--light-blue); margin-bottom: 1rem; font-size: 1.2rem;">
                            🔬 Stationarity Tests (Augmented Dickey-Fuller)
                        </h3>
                        <div class="grid-2" id="stationarityTests"></div>
                    </div>
                    
                    <!-- Time Series Comparison -->
                    <div style="margin-bottom: 2rem;">
                        <h3 style="color: var(--light-blue); margin-bottom: 1rem; font-size: 1.2rem;">
                            📈 Time Series Comparison
                        </h3>
                        <div class="chart-wrapper" id="differencingSeriesChart"></div>
                    </div>
                    
                    <!-- ACF and PACF for Original -->
                    <div style="margin-bottom: 2rem;">
                        <h3 style="color: var(--light-blue); margin-bottom: 1rem; font-size: 1.2rem;">
                            📉 Original Series - ACF & PACF
                        </h3>
                        <p style="color: var(--text-secondary); margin-bottom: 1rem; font-size: 0.9rem;">
                            Use these plots to identify patterns in the original time series
                        </p>
                        <div class="grid-2" style="gap: 1.5rem;">
                            <div class="chart-wrapper" id="acfOriginalChart"></div>
                            <div class="chart-wrapper" id="pacfOriginalChart"></div>
                        </div>
                    </div>
                    
                    <!-- ACF and PACF for Differenced -->
                    <div style="margin-bottom: 2rem;">
                        <h3 style="color: var(--light-blue); margin-bottom: 1rem; font-size: 1.2rem;">
                            📉 Differenced Series - ACF & PACF
                        </h3>
                        <p style="color: var(--text-secondary); margin-bottom: 1rem; font-size: 0.9rem;">
                            After differencing: ACF suggests MA(q) order, PACF suggests AR(p) order
                        </p>
                        <div class="grid-2" style="gap: 1.5rem;">
                            <div class="chart-wrapper" id="acfDiffChart"></div>
                            <div class="chart-wrapper" id="pacfDiffChart"></div>
                        </div>
                    </div>

                    <!-- Interpretation Guide -->
                    <div style="background: rgba(33, 150, 243, 0.1); padding: 1.5rem; border-radius: 8px; border-left: 4px solid var(--light-blue);">
                        <h4 style="color: var(--light-blue); margin-bottom: 1rem;">
                            📚 How to Interpret Results
                        </h4>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem;">
                            <div>
                                <strong style="color: var(--off-white);">Stationarity (ADF Test):</strong>
                                <p style="font-size: 0.9rem; color: var(--text-secondary); margin-top: 0.5rem;">
                                    • p-value < 0.05 → Stationary ✅<br>
                                    • p-value ≥ 0.05 → Non-stationary ❌<br>
                                    • If differenced is stationary, d=1 is good
                                </p>
                            </div>
                            <div>
                                <strong style="color: var(--off-white);">ACF (Autocorrelation):</strong>
                                <p style="font-size: 0.9rem; color: var(--text-secondary); margin-top: 0.5rem;">
                                    • Cuts off sharply after q lags → MA(q)<br>
                                    • Decays slowly → AR component needed<br>
                                    • Use for determining q parameter
                                </p>
                            </div>
                            <div>
                                <strong style="color: var(--off-white);">PACF (Partial Autocorrelation):</strong>
                                <p style="font-size: 0.9rem; color: var(--text-secondary); margin-top: 0.5rem;">
                                    • Cuts off sharply after p lags → AR(p)<br>
                                    • Decays slowly → MA component needed<br>
                                    • Use for determining p parameter
                                </p>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Next Steps -->
                <div class="card" id="nextStepsCard" style="text-align: center; display: none;">
                    <h3 style="color: var(--light-blue); margin-bottom: 2rem;">Ready to Build Models?</h3>
                    <button class="btn btn-primary btn-lg" onclick="Navigation.navigateTo('modeling')">
                        Proceed to Modeling →
                    </button>
                </div>
            </div>
        `;

        this.setupEventListeners();
        
        // Auto-run basic exploration if results exist
        const savedResults = Utils.getFromMemory('exploration_results');
        if (savedResults) {
            this.explorationResults = savedResults;
            this.displayBasicResults();
        }
        
        // Auto-run detailed exploration if results exist
        const savedDetailed = Utils.getFromMemory('detailed_exploration');
        if (savedDetailed) {
            this.detailedResults = savedDetailed;
            this.displayDetailedResults();
        }
    },

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        const basicBtn = Utils.getEl('runBasicExplorationBtn');
        if (basicBtn) {
            basicBtn.addEventListener('click', () => this.runBasicExploration());
        }

        const detailedBtn = Utils.getEl('runDetailedExplorationBtn');
        if (detailedBtn) {
            detailedBtn.addEventListener('click', () => this.runDetailedExploration());
        }

        const diffBtn = Utils.getEl('differencingBtn');
        if (diffBtn) {
            diffBtn.addEventListener('click', () => this.loadDifferencingAnalysis());
        }
    },

    /**
     * Load differencing analysis with ACF/PACF plots
     * Matches backend endpoint: /explore/differencing-plots
     */
    async loadDifferencingAnalysis() {
        if (!this.pipelineData || !this.pipelineData.pipeline_id) {
            Utils.showToast('No pipeline data available', 'warning');
            console.error('[Differencing] No pipeline data found');
            return;
        }

        console.log('[Differencing] Starting analysis for pipeline:', this.pipelineData.pipeline_id);
        Utils.showLoading('Loading differencing analysis with ACF/PACF plots...');

        try {
            // Direct API call to backend endpoint
            const apiUrl = `${CONFIG.API_BASE_URL}/explore/differencing-plots`;
            console.log('[Differencing] Calling API:', apiUrl);
            
            const response = await fetch(apiUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    pipeline_id: this.pipelineData.pipeline_id
                })
            });

            console.log('[Differencing] Response status:', response.status);
            console.log('[Differencing] Response ok:', response.ok);
            
            const result = await response.json();
            console.log('[Differencing] Response data:', result);
            
            Utils.hideLoading();

            // Validate response structure matches backend
            if (response.ok && result && result.original && result.differenced) {
                console.log('[Differencing] ✓ Valid data structure received');
                console.log('[Differencing] Original data points:', result.original.dates?.length);
                console.log('[Differencing] Differenced data points:', result.differenced.dates?.length);
                console.log('[Differencing] ACF original lags:', result.acf_original?.lags?.length);
                console.log('[Differencing] PACF original lags:', result.pacf_original?.lags?.length);
                
                this.displayDifferencingAnalysis(result);
                
                const card = Utils.getEl('differencingCard');
                if (card) {
                    card.style.display = 'block';
                    Utils.scrollToElement('differencingCard');
                }
                
                Utils.showToast('Differencing analysis loaded successfully!', 'success');
            } else {
                const errorMsg = result.error || 'No differencing data available';
                console.error('[Differencing] ✗ Invalid response:', errorMsg);
                console.error('[Differencing] Full response:', result);
                Utils.showToast(errorMsg, 'warning');
            }
        } catch (error) {
            Utils.hideLoading();
            console.error('[Differencing] ✗ Exception occurred:', error);
            console.error('[Differencing] Error stack:', error.stack);
            Utils.showToast('Failed to load differencing analysis: ' + error.message, 'error');
        }
    },

    /**
     * Display differencing analysis results
     * Matches backend response structure from api.py
     */
    displayDifferencingAnalysis(data) {
        console.log('[Differencing] Displaying analysis results');
        console.log('[Differencing] Data structure check:', {
            hasOriginal: !!data.original,
            hasDifferenced: !!data.differenced,
            hasACFOriginal: !!data.acf_original,
            hasPACFOriginal: !!data.pacf_original,
            hasACFDifferenced: !!data.acf_differenced,
            hasPACFDifferenced: !!data.pacf_differenced
        });
        
        // Display stationarity tests
        const testsContainer = Utils.getEl('stationarityTests');
        if (testsContainer && data.original && data.differenced) {
            const originalStationary = data.original.is_stationary;
            const diffStationary = data.differenced.is_stationary;
            
            console.log('[Differencing] Stationarity:', {
                original: originalStationary,
                differenced: diffStationary
            });
            
            testsContainer.innerHTML = `
                <div class="stat-card" style="background: ${originalStationary ? 'rgba(76, 175, 80, 0.1)' : 'rgba(244, 67, 54, 0.1)'}; border-left: 4px solid ${originalStationary ? '#4CAF50' : '#F44336'};">
                    <div class="stat-icon" style="font-size: 2.5rem;">${originalStationary ? '✅' : '❌'}</div>
                    <div class="stat-label" style="font-size: 1rem; margin-top: 0.5rem;">Original Series</div>
                    <div class="stat-value" style="font-size: 1.1rem; font-weight: 600; margin-top: 0.5rem; color: ${originalStationary ? '#4CAF50' : '#F44336'};">
                        ${originalStationary ? 'Stationary' : 'Non-Stationary'}
                    </div>
                    <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.1);">
                        <div style="font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 0.3rem;">
                            <strong>ADF Statistic:</strong> ${Utils.formatNumber(data.original.adf_statistic, 4)}
                        </div>
                        <div style="font-size: 0.85rem; color: var(--text-secondary);">
                            <strong>p-value:</strong> ${Utils.formatNumber(data.original.adf_pvalue, 4)}
                        </div>
                    </div>
                </div>
                <div class="stat-card" style="background: ${diffStationary ? 'rgba(76, 175, 80, 0.1)' : 'rgba(244, 67, 54, 0.1)'}; border-left: 4px solid ${diffStationary ? '#4CAF50' : '#F44336'};">
                    <div class="stat-icon" style="font-size: 2.5rem;">${diffStationary ? '✅' : '❌'}</div>
                    <div class="stat-label" style="font-size: 1rem; margin-top: 0.5rem;">Differenced Series</div>
                    <div class="stat-value" style="font-size: 1.1rem; font-weight: 600; margin-top: 0.5rem; color: ${diffStationary ? '#4CAF50' : '#F44336'};">
                        ${diffStationary ? 'Stationary' : 'Non-Stationary'}
                    </div>
                    <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.1);">
                        <div style="font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 0.3rem;">
                            <strong>ADF Statistic:</strong> ${Utils.formatNumber(data.differenced.adf_statistic, 4)}
                        </div>
                        <div style="font-size: 0.85rem; color: var(--text-secondary);">
                            <strong>p-value:</strong> ${Utils.formatNumber(data.differenced.adf_pvalue, 4)}
                        </div>
                    </div>
                </div>
            `;
            console.log('[Differencing] ✓ Stationarity tests displayed');
        }

        // Display time series comparison
        if (data.original && data.differenced) {
            console.log('[Differencing] Rendering time series comparison');
            Charts.renderTimeSeriesComparison('differencingSeriesChart', data.original, data.differenced);
            console.log('[Differencing] ✓ Time series comparison rendered');
        }

        // Display ACF/PACF for original
        if (data.acf_original && data.pacf_original) {
            console.log('[Differencing] Rendering original ACF/PACF');
            Charts.renderACFPlot('acfOriginalChart', data.acf_original);
            Charts.renderPACFPlot('pacfOriginalChart', data.pacf_original);
            console.log('[Differencing] ✓ Original ACF/PACF rendered');
        }

        // Display ACF/PACF for differenced
        if (data.acf_differenced && data.pacf_differenced) {
            console.log('[Differencing] Rendering differenced ACF/PACF');
            Charts.renderACFPlot('acfDiffChart', data.acf_differenced);
            Charts.renderPACFPlot('pacfDiffChart', data.pacf_differenced);
            console.log('[Differencing] ✓ Differenced ACF/PACF rendered');
        }
        
        console.log('[Differencing] ✓ All visualizations completed');
    },

    /**
     * Run basic exploration with enhanced error handling
     */
    async runBasicExploration() {
        Utils.showLoading('Running basic exploratory analysis...');
        
        try {
            const result = await API.runExploration(this.pipelineData.pipeline_id);
            Utils.hideLoading();

            if (result.success && result.data) {
                this.explorationResults = result.data;
                Utils.saveToMemory('exploration_results', result.data);
                this.displayBasicResults();
                Utils.showToast('Basic analysis completed successfully', 'success');
            } else {
                const errorMsg = result.error || 'Unknown error occurred';
                Utils.showToast('Basic analysis failed: ' + errorMsg, 'error');
                console.error('Exploration failed:', result);
            }
        } catch (error) {
            Utils.hideLoading();
            Utils.showToast('Basic analysis failed: ' + error.message, 'error');
            console.error('Exploration error:', error);
        }
    },

    /**
     * Run detailed exploration with enhanced error handling
     */
    async runDetailedExploration() {
        Utils.showLoading('Running detailed analysis (outliers, differencing, seasonality)...');
        
        try {
            const result = await API.request('/explore/detailed', {
                method: 'POST',
                data: { pipeline_id: this.pipelineData.pipeline_id }
            });
            
            Utils.hideLoading();

            if (result.success && result.data) {
                this.detailedResults = result.data;
                Utils.saveToMemory('detailed_exploration', result.data);
                this.displayDetailedResults();
                Utils.showToast('Detailed analysis completed successfully', 'success');
            } else {
                const errorMsg = result.error || 'Unknown error occurred';
                Utils.showToast('Detailed analysis failed: ' + errorMsg, 'error');
                console.error('Detailed exploration failed:', result);
            }
        } catch (error) {
            Utils.hideLoading();
            Utils.showToast('Detailed analysis failed: ' + error.message, 'error');
            console.error('Detailed exploration error:', error);
        }
    },

    /**
     * Display basic exploration results with null safety
     */
    displayBasicResults() {
        const resultsContainer = Utils.getEl('basicResults');
        if (!resultsContainer) return;
        
        resultsContainer.style.display = 'block';
        Utils.getEl('nextStepsCard').style.display = 'block';
        
        // Display each component with safe checks
        if (this.explorationResults && this.explorationResults.basic_stats) {
            this.displayBasicStats();
        }
        
        if (this.explorationResults && this.explorationResults.time_series_data) {
            this.displayTimeSeriesPlot();
        }
        
        if (this.explorationResults && this.explorationResults.trend) {
            this.displayTrendAnalysis();
        }
        
        if (this.explorationResults && this.explorationResults.seasonal) {
            this.displaySeasonality();
        }
        
        if (this.explorationResults && this.explorationResults.statistical_tests) {
            this.displayStatisticalTests();
        }
        
        if (this.explorationResults && this.explorationResults.distribution) {
            this.displayDistribution();
        }
        
        if (this.explorationResults && this.explorationResults.acf) {
            this.displayACF();
        }
    },

    /**
     * Display basic statistics with null safety
     */
    displayBasicStats() {
        const stats = this.explorationResults.basic_stats || {};
        const container = Utils.getEl('basicStats');
        if (!container) return;
        
        const statsData = [
            { icon: '📊', label: 'Mean', value: Utils.formatNumber(stats.mean) },
            { icon: '📈', label: 'Std Dev', value: Utils.formatNumber(stats.std) },
            { icon: '⬇️', label: 'Min', value: Utils.formatNumber(stats.min) },
            { icon: '⬆️', label: 'Max', value: Utils.formatNumber(stats.max) }
        ];

        container.innerHTML = statsData.map(stat => `
            <div class="stat-card">
                <div class="stat-icon">${stat.icon}</div>
                <div class="stat-value">${stat.value}</div>
                <div class="stat-label">${stat.label}</div>
            </div>
        `).join('');
    },

    /**
     * Display time series plot with validation
     */
    displayTimeSeriesPlot() {
        const data = this.explorationResults.time_series_data;
        if (!data || !data.dates || !data.values || data.dates.length === 0) {
            Utils.getEl('timeSeriesChart').innerHTML = '<p style="text-align:center;color:var(--text-secondary);">No time series data available</p>';
            return;
        }
        
        Charts.renderTimeSeries('timeSeriesChart', {
            dates: data.dates,
            values: data.values,
            title: 'Original Time Series'
        });
    },

    /**
     * Display trend analysis with validation
     */
    displayTrendAnalysis() {
        const trend = this.explorationResults.trend;
        if (!trend || !trend.dates || !trend.values || trend.dates.length === 0) {
            Utils.getEl('trendChart').innerHTML = '<p style="text-align:center;color:var(--text-secondary);">No trend data available</p>';
            return;
        }
        
        Charts.renderLine('trendChart', {
            dates: trend.dates,
            values: trend.values,
            title: 'Trend Component',
            color: '#4CAF50'
        });
    },

    /**
     * Display seasonality with validation
     */
    displaySeasonality() {
        const seasonal = this.explorationResults.seasonal;
        if (!seasonal || !seasonal.dates || !seasonal.values || seasonal.dates.length === 0) {
            Utils.getEl('seasonalityChart').innerHTML = '<p style="text-align:center;color:var(--text-secondary);">No seasonal data available</p>';
            return;
        }
        
        Charts.renderLine('seasonalityChart', {
            dates: seasonal.dates,
            values: seasonal.values,
            title: 'Seasonal Component',
            color: '#ff9800'
        });
    },

    /**
     * Display statistical tests with safe value extraction
     */
    displayStatisticalTests() {
        const tests = this.explorationResults.statistical_tests || {};
        const container = Utils.getEl('statisticalTests');
        if (!container) return;
        
        const adfStat = tests.adf_statistic !== undefined ? tests.adf_statistic : 'N/A';
        const adfPval = tests.adf_pvalue !== undefined ? tests.adf_pvalue : 'N/A';
        const isStationary = tests.is_stationary !== undefined ? tests.is_stationary : false;
        const hasSeasonality = tests.has_seasonality !== undefined ? tests.has_seasonality : false;
        
        container.innerHTML = `
            <div class="grid-2">
                <div class="info-item">
                    <span class="info-item-label">ADF Statistic:</span>
                    <span class="info-item-value">${typeof adfStat === 'number' ? Utils.formatNumber(adfStat) : adfStat}</span>
                </div>
                <div class="info-item">
                    <span class="info-item-label">ADF P-Value:</span>
                    <span class="info-item-value">${typeof adfPval === 'number' ? Utils.formatNumber(adfPval, 4) : adfPval}</span>
                </div>
                <div class="info-item">
                    <span class="info-item-label">Stationarity:</span>
                    <span class="info-item-value" style="color: ${isStationary ? 'var(--success)' : 'var(--warning)'}">
                        ${isStationary ? 'Stationary ✓' : 'Non-Stationary'}
                    </span>
                </div>
                <div class="info-item">
                    <span class="info-item-label">Seasonality Detected:</span>
                    <span class="info-item-value" style="color: ${hasSeasonality ? 'var(--info)' : 'var(--text-secondary)'}">
                        ${hasSeasonality ? 'Yes' : 'No'}
                    </span>
                </div>
            </div>
        `;
    },

    /**
     * Display distribution with validation
     */
    displayDistribution() {
        const dist = this.explorationResults.distribution;
        if (!dist || !dist.bins || !dist.counts || dist.bins.length === 0) {
            Utils.getEl('distributionChart').innerHTML = '<p style="text-align:center;color:var(--text-secondary);">No distribution data available</p>';
            return;
        }
        
        Charts.renderHistogram('distributionChart', {
            bins: dist.bins,
            counts: dist.counts,
            title: 'Value Distribution'
        });
    },

    /**
     * Display ACF with validation and proper label conversion
     */
    displayACF() {
        const acf = this.explorationResults.acf;
        if (!acf || !acf.lags || !acf.values || acf.lags.length === 0) {
            Utils.getEl('acfChart').innerHTML = '<p style="text-align:center;color:var(--text-secondary);">No ACF data available</p>';
            return;
        }
        
        // Use labels if available, otherwise convert lags to strings
        const labels = acf.labels || acf.lags.map(l => l.toString());
        
        Charts.renderBar('acfChart', {
            labels: labels,
            values: acf.values,
            title: 'Autocorrelation Function'
        });
    },

    /**
     * Display detailed exploration results with comprehensive null safety
     */
    displayDetailedResults() {
        const detailedContainer = Utils.getEl('detailedResults');
        if (!detailedContainer) return;
        
        detailedContainer.style.display = 'block';
        
        if (this.detailedResults && this.detailedResults.outliers) {
            this.displayOutliers();
        }
        
        if (this.detailedResults && this.detailedResults.differencing) {
            this.displayDifferencing();
        }
        
        if (this.detailedResults && this.detailedResults.seasonality) {
            this.displaySeasonalityStrength();
        }
        
        if (this.detailedResults && this.detailedResults.eda_report) {
            this.displayEDAReport();
        }
        
        Utils.scrollToElement('detailedResults');
    },

    /**
     * Display outlier detection results with safe value extraction
     */
    displayOutliers() {
        const outliers = this.detailedResults.outliers;
        const container = Utils.getEl('outlierStats');
        if (!container || !outliers) return;
        
        const iqr = outliers.iqr || {};
        const zscore = outliers.zscore || {};
        
        const iqrCount = iqr.outlier_count || iqr.count || 0;
        const iqrPercent = iqr.outlier_percentage || 0;
        const zscoreCount = zscore.outlier_count || zscore.count || 0;
        const zscorePercent = zscore.outlier_percentage || 0;
        
        container.innerHTML = `
            <div class="stat-card">
                <div class="stat-icon">🔍</div>
                <div class="stat-value">${iqrCount}</div>
                <div class="stat-label">IQR Method Outliers</div>
                <div style="font-size: 0.85rem; color: var(--text-secondary); margin-top: 0.5rem;">
                    ${Utils.formatNumber(iqrPercent, 2)}% of data
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">🔎</div>
                <div class="stat-value">${zscoreCount}</div>
                <div class="stat-label">Z-Score Method Outliers</div>
                <div style="font-size: 0.85rem; color: var(--text-secondary); margin-top: 0.5rem;">
                    ${Utils.formatNumber(zscorePercent, 2)}% of data
                </div>
            </div>
        `;
    },

    /**
     * Display differencing analysis with safe value extraction
     */
    displayDifferencing() {
        const diff = this.detailedResults.differencing;
        const container = Utils.getEl('differencingInfo');
        if (!container || !diff) return;
        
        const adfOriginal = diff.adf_original !== undefined ? diff.adf_original : 1.0;
        const adfDiff = diff.adf_differenced !== undefined ? diff.adf_differenced : 1.0;
        const isStationary = diff.is_stationary_after_diff !== undefined ? diff.is_stationary_after_diff : false;
        
        container.innerHTML = `
            <div class="info-grid">
                <div class="info-item">
                    <span class="info-item-label">Original ADF P-Value:</span>
                    <span class="info-item-value">${Utils.formatNumber(adfOriginal, 4)}</span>
                </div>
                <div class="info-item">
                    <span class="info-item-label">After Differencing:</span>
                    <span class="info-item-value">${Utils.formatNumber(adfDiff, 4)}</span>
                </div>
                <div class="info-item">
                    <span class="info-item-label">Stationary After Diff:</span>
                    <span class="info-item-value" style="color: ${isStationary ? 'var(--success)' : 'var(--warning)'}">
                        ${isStationary ? 'Yes ✓' : 'No'}
                    </span>
                </div>
                <div class="info-item">
                    <span class="info-item-label">Recommended d:</span>
                    <span class="info-item-value">${isStationary ? '1' : '2'}</span>
                </div>
            </div>
        `;
    },

    /**
     * Display seasonality strength with safe value extraction
     */
    displaySeasonalityStrength() {
        const seasonal = this.detailedResults.seasonality;
        const container = Utils.getEl('seasonalityInfo');
        if (!container || !seasonal) return;
        
        const period = seasonal.period || 'N/A';
        const strength = seasonal.strength !== undefined && seasonal.strength !== null ? seasonal.strength : 0;
        const strengthPercent = typeof strength === 'number' ? (strength * 100).toFixed(1) : '0.0';
        const strengthColor = strength > 0.6 ? 'var(--success)' : 
                             strength > 0.3 ? 'var(--warning)' : 'var(--error)';
        
        container.innerHTML = `
            <div class="info-grid">
                <div class="info-item">
                    <span class="info-item-label">Seasonal Period:</span>
                    <span class="info-item-value">${period}</span>
                </div>
                <div class="info-item">
                    <span class="info-item-label">Seasonality Strength:</span>
                    <span class="info-item-value" style="color: ${strengthColor}">
                        ${strengthPercent}%
                    </span>
                </div>
            </div>
            <div style="margin-top: 1rem;">
                <div style="width: 100%; background: rgba(30, 90, 150, 0.2); border-radius: 10px; height: 20px; overflow: hidden;">
                    <div style="width: ${strengthPercent}%; background: ${strengthColor}; height: 100%; transition: width 0.5s ease;"></div>
                </div>
                <p style="text-align: center; margin-top: 0.5rem; font-size: 0.9rem; color: var(--text-secondary);">
                    ${strength > 0.6 ? 'Strong seasonality detected' : 
                      strength > 0.3 ? 'Moderate seasonality' : 'Weak or no seasonality'}
                </p>
            </div>
        `;
    },

    /**
     * Display comprehensive EDA report with safe nested value extraction
     */
    displayEDAReport() {
        const report = this.detailedResults.eda_report;
        const container = Utils.getEl('edaReport');
        if (!container || !report) return;
        
        const basic = report.basic_stats || {};
        const trend = report.trend_info || {};
        
        // Safe value extraction with defaults
        const median = basic.median !== undefined ? Utils.formatNumber(basic.median) : 'N/A';
        const range = basic.range !== undefined ? Utils.formatNumber(basic.range) : 'N/A';
        const cv = basic.cv !== undefined ? Utils.formatNumber(basic.cv, 2) + '%' : 'N/A';
        const skewness = basic.skewness !== undefined ? Utils.formatNumber(basic.skewness, 3) : 'N/A';
        const kurtosis = basic.kurtosis !== undefined ? Utils.formatNumber(basic.kurtosis, 3) : 'N/A';
        
        const trendDir = trend.trend_direction || 'N/A';
        const slope = trend.slope !== undefined ? Utils.formatNumber(trend.slope, 6) : 'N/A';
        const rSquared = trend.r_squared !== undefined ? Utils.formatNumber(trend.r_squared, 4) : 'N/A';
        const pValue = trend.p_value !== undefined ? Utils.formatNumber(trend.p_value, 6) : 'N/A';
        
        container.innerHTML = `
            <div class="grid-2" style="gap: 2rem;">
                <div>
                    <h4 style="color: var(--light-blue); margin-bottom: 1rem;">Extended Statistics</h4>
                    <div style="display: flex; flex-direction: column; gap: 0.5rem;">
                        ${this.createStatRow('Median', median)}
                        ${this.createStatRow('Range', range)}
                        ${this.createStatRow('CV', cv)}
                        ${this.createStatRow('Skewness', skewness)}
                        ${this.createStatRow('Kurtosis', kurtosis)}
                    </div>
                </div>
                <div>
                    <h4 style="color: var(--light-blue); margin-bottom: 1rem;">Trend Information</h4>
                    <div style="display: flex; flex-direction: column; gap: 0.5rem;">
                        ${this.createStatRow('Trend Direction', trendDir)}
                        ${this.createStatRow('Slope', slope)}
                        ${this.createStatRow('R²', rSquared)}
                        ${this.createStatRow('P-Value', pValue)}
                    </div>
                </div>
            </div>
        `;
    },

    /**
     * Helper to create stat row
     */
    createStatRow(label, value) {
        return `
            <div style="display: flex; justify-content: space-between; padding: 0.5rem 1rem; background: rgba(30, 90, 150, 0.1); border-radius: 6px;">
                <span style="color: var(--off-white);">${label}:</span>
                <span style="color: var(--light-blue); font-weight: 600;">${value}</span>
            </div>
        `;
    }
};

// Make ExplorePage available globally
window.ExplorePage = ExplorePage;
console.log('✓ ExplorePage module loaded');