// ==================== HOME PAGE (FIXED & OPTIMIZED) ====================

const HomePage = {
    initialized: false,
    statsTimeout: null,
    healthTimeout: null,
    
    render() {
        console.log('HomePage: Rendering...');
        const container = Utils.getEl('mainContainer');
        
        if (!container) {
            console.error('Main container not found!');
            return;
        }
        
        // Clear any existing timeouts
        this.cleanup();
        
        // Render content IMMEDIATELY - no async blocking
        container.innerHTML = `
            <div class="page active" id="homePage">
                <!-- Hero Section -->
                <div class="hero-section" style="text-align: center; padding: 4rem 2rem; margin-bottom: 3rem;">
                    <h1 style="font-size: 3.5rem; margin-bottom: 1rem; background: linear-gradient(135deg, var(--bright-blue), var(--light-blue)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; animation: fadeIn 0.8s ease;">
                        SURADATA - Pemodelan Deret Waktu
                    </h1>
                    <p style="font-size: 1.3rem; color: var(--text-secondary); max-width: 800px; margin: 0 auto 2rem; animation: fadeIn 1s ease;">
                        Peramalan Deret Waktu dengan Model Box-Jenkins dan Deep Learning
                    </p>
                    <div style="display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap;">
                        <button class="btn btn-primary btn-lg" onclick="Navigation.navigateTo('upload')" style="animation: slideInUp 0.8s ease;">
                            Mulai Sekarang
                        </button>
                        <button class="btn btn-secondary btn-lg" onclick="Navigation.navigateTo('about')" style="animation: slideInUp 0.9s ease;">
                            Pelajari Lebih Lanjut
                        </button>
                    </div>
                </div>

                <!-- Features Grid -->
                <div class="grid-3" style="margin-bottom: 3rem;">
                    <div class="card animated animated-delay-1" style="animation: fadeIn 1s ease;">
                        <div style="text-align: center;">
                            <div style="font-size: 4rem; margin-bottom: 1rem;">📈</div>
                            <h3 style="color: var(--light-blue); margin-bottom: 1rem;">Multiple Models</h3>
                            <p style="color: var(--text-secondary);">
                                Pilih dari berbagai model peramalan seperti ARIMA, SARIMA, dan Transformer
                            </p>
                        </div>
                    </div>

                    <div class="card animated animated-delay-2" style="animation: fadeIn 1.2s ease;">
                        <div style="text-align: center;">
                            <div style="font-size: 4rem; margin-bottom: 1rem;">🔍</div>
                            <h3 style="color: var(--light-blue); margin-bottom: 1rem;">Deep Analysis</h3>
                            <p style="color: var(--text-secondary);">
                                Analisis data eksploratori komprehensif dengan deteksi tren, musiman, dan stasioneritas
                            </p>
                        </div>
                    </div>

                    <div class="card animated animated-delay-3" style="animation: fadeIn 1.4s ease;">
                        <div style="text-align: center;">
                            <div style="font-size: 4rem; margin-bottom: 1rem;">⚡</div>
                            <h3 style="color: var(--light-blue); margin-bottom: 1rem;">Real-time Results</h3>
                            <p style="color: var(--text-secondary);">
                                Pemrosesan cepat dengan visualisasi interaktif dan hasil yang dapat diunduh
                            </p>
                        </div>
                    </div>
                </div>

                <!-- Stats Section -->
                <div class="card" style="margin-bottom: 3rem;">
                    <h2 class="card-title">Platform Statistics</h2>
                    <div class="grid-4" id="statsContainer">
                        <div class="stat-card">
                            <div class="stat-icon">🎯</div>
                            <div class="stat-value" id="statModels">5</div>
                            <div class="stat-label">Model Tersedia</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-icon">📊</div>
                            <div class="stat-value" id="statPipelines">0</div>
                            <div class="stat-label">Pipelines Aktif</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-icon">✅</div>
                            <div class="stat-value" id="statForecasts">0</div>
                            <div class="stat-label">Peramalan Dihasilkan</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-icon">⚡</div>
                            <div class="stat-value" id="statAccuracy">95%</div>
                            <div class="stat-label">Rata-Rata Akurasi</div>
                        </div>
                    </div>
                </div>

                <!-- How It Works -->
                <div class="card">
                    <h2 class="card-title">How It Works</h2>
                    <div class="grid-4">
                        <div style="text-align: center; padding: 2rem;">
                            <div style="width: 80px; height: 80px; margin: 0 auto 1rem; background: linear-gradient(135deg, var(--accent-blue), var(--bright-blue)); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 2rem; box-shadow: 0 4px 20px var(--glow);">
                                1
                            </div>
                            <h4 style="color: var(--light-blue); margin-bottom: 0.5rem;">Unggah Data</h4>
                            <p style="color: var(--text-secondary); font-size: 0.9rem;">
                                Unggah data deret waktu Anda dalam format CSV
                            </p>
                        </div>

                        <div style="text-align: center; padding: 2rem;">
                            <div style="width: 80px; height: 80px; margin: 0 auto 1rem; background: linear-gradient(135deg, var(--accent-blue), var(--bright-blue)); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 2rem; box-shadow: 0 4px 20px var(--glow);">
                                2
                            </div>
                            <h4 style="color: var(--light-blue); margin-bottom: 0.5rem;">Jelajahi</h4>
                            <p style="color: var(--text-secondary); font-size: 0.9rem;">
                                Analisis pola, tren, dan statistik deskriptif
                            </p>
                        </div>

                        <div style="text-align: center; padding: 2rem;">
                            <div style="width: 80px; height: 80px; margin: 0 auto 1rem; background: linear-gradient(135deg, var(--accent-blue), var(--bright-blue)); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 2rem; box-shadow: 0 4px 20px var(--glow);">
                                3
                            </div>
                            <h4 style="color: var(--light-blue); margin-bottom: 0.5rem;">Latih Model</h4>
                            <p style="color: var(--text-secondary); font-size: 0.9rem;">
                                Pilih dan latih beberapa model peramalan
                            </p>
                        </div>

                        <div style="text-align: center; padding: 2rem;">
                            <div style="width: 80px; height: 80px; margin: 0 auto 1rem; background: linear-gradient(135deg, var(--accent-blue), var(--bright-blue)); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 2rem; box-shadow: 0 4px 20px var(--glow);">
                                4
                            </div>
                            <h4 style="color: var(--light-blue); margin-bottom: 0.5rem;">Peramalan</h4>
                            <p style="color: var(--text-secondary); font-size: 0.9rem;">
                                Jalankan peramalan dan unduh hasilnya
                            </p>
                        </div>
                    </div>
                </div>

                <!-- Quick Actions -->
                <div class="card" style="margin-top: 3rem; text-align: center;">
                    <h3 style="color: var(--light-blue); margin-bottom: 2rem;">Siap untuk Memulai Peramalan?</h3>
                    <button class="btn btn-primary btn-lg" onclick="Navigation.navigateTo('upload')">
                        Unggah Data Sekarang
                    </button>
                </div>
            </div>
        `;

        this.initialized = true;
        console.log('HomePage: Rendered successfully');

        // Load stats asynchronously AFTER render completes
        this.scheduleStatsLoad();
    },

    scheduleStatsLoad() {
        // Wait for DOM to be fully ready, then load stats
        this.statsTimeout = setTimeout(() => {
            this.loadStats()
                .then(() => console.log('HomePage: Stats loaded'))
                .catch(err => {
                    console.warn('HomePage: Stats loading failed (non-critical)', err);
                    this.setDefaultStats();
                });
        }, 1000);

        // Also schedule health check independently
        this.healthTimeout = setTimeout(() => {
            this.checkHealth()
                .then(() => console.log('HomePage: Health check completed'))
                .catch(err => console.warn('HomePage: Health check failed (non-critical)', err));
        }, 1500);
    },

    async loadStats() {
        try {
            const result = await Promise.race([
                API.getPipelines(),
                new Promise((_, reject) => 
                    setTimeout(() => reject(new Error('Stats timeout')), 5000)
                )
            ]);
            
            if (result && result.success && result.data) {
                const pipelines = result.data.pipelines || [];
                const forecasts = Utils.getFromMemory('total_forecasts') || 0;
                
                const pipelinesEl = Utils.getEl('statPipelines');
                const forecastsEl = Utils.getEl('statForecasts');
                
                if (pipelinesEl) {
                    Utils.animateNumber(pipelinesEl, 0, pipelines.length, 1000);
                }
                if (forecastsEl) {
                    Utils.animateNumber(forecastsEl, 0, forecasts, 1000);
                }
            } else {
                this.setDefaultStats();
            }
        } catch (error) {
            console.warn('Stats loading error:', error.message);
            this.setDefaultStats();
        }
    },

    async checkHealth() {
        try {
            const result = await Promise.race([
                API.healthCheck(),
                new Promise((_, reject) => 
                    setTimeout(() => reject(new Error('Health check timeout')), 3000)
                )
            ]);
            
            const statusText = Utils.getEl('statusText');
            const statusDot = document.querySelector('.status-dot');
            
            if (statusText && statusDot) {
                if (result && result.success) {
                    statusText.textContent = 'System Online';
                    statusDot.style.background = 'var(--success)';
                } else {
                    statusText.textContent = 'System Offline';
                    statusDot.style.background = 'var(--error)';
                }
            }
        } catch (error) {
            console.warn('Health check error:', error.message);
            const statusText = Utils.getEl('statusText');
            const statusDot = document.querySelector('.status-dot');
            
            if (statusText) statusText.textContent = 'System Offline';
            if (statusDot) statusDot.style.background = 'var(--error)';
        }
    },

    setDefaultStats() {
        const pipelinesEl = Utils.getEl('statPipelines');
        const forecastsEl = Utils.getEl('statForecasts');
        
        if (pipelinesEl) pipelinesEl.textContent = '0';
        if (forecastsEl) forecastsEl.textContent = '0';
    },

    refresh() {
        if (this.initialized) {
            console.log('HomePage: Refreshing...');
            this.cleanup();
            this.scheduleStatsLoad();
        }
    },

    cleanup() {
        if (this.statsTimeout) {
            clearTimeout(this.statsTimeout);
            this.statsTimeout = null;
        }
        if (this.healthTimeout) {
            clearTimeout(this.healthTimeout);
            this.healthTimeout = null;
        }
    }
};

window.HomePage = HomePage;
console.log('✓ HomePage loaded');