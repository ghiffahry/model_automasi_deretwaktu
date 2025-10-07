// ==================== ABOUT PAGE ====================

const AboutPage = {
    /**
     * Render about page
     */
    render() {
        const container = Utils.getEl('mainContainer');
        
        container.innerHTML = `
            <div class="page active" id="aboutPage">
                <h1 style="margin-bottom: 2rem; animation: fadeIn 0.5s ease;">
                    ℹ️ About SURADATA
                </h1>

                <!-- Platform Overview -->
                <div class="card" style="animation: slideInUp 0.6s ease;">
                    <h2 class="card-title">Platform Overview</h2>
                    <p style="color: var(--text-secondary); line-height: 1.8; margin-bottom: 1rem;">
                        SURADATA is an advanced time series forecasting platform designed to democratize access to 
                        state-of-the-art predictive analytics. Built with modern web technologies and powered by 
                        cutting-edge machine learning algorithms, SURADATA enables businesses and researchers to 
                        generate accurate forecasts without requiring deep technical expertise.
                    </p>
                    <p style="color: var(--text-secondary); line-height: 1.8;">
                        Our platform combines classical statistical methods with modern deep learning approaches, 
                        providing users with a comprehensive toolkit for time series analysis and forecasting.
                    </p>
                </div>

                <!-- Key Features -->
                <div class="card" style="animation: slideInUp 0.7s ease;">
                    <h2 class="card-title">Key Features</h2>
                    <div class="grid-2" style="gap: 2rem;">
                        <div>
                            <h4 style="color: var(--light-blue); margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem;">
                                <span style="font-size: 1.5rem;">🎯</span> Multiple Model Support
                            </h4>
                            <p style="color: var(--text-secondary); line-height: 1.8;">
                                Access to ARIMA, SARIMA, ARIMAX, SARIMAX, and Transformer models for different 
                                forecasting scenarios and data characteristics.
                            </p>
                        </div>
                        <div>
                            <h4 style="color: var(--light-blue); margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem;">
                                <span style="font-size: 1.5rem;">🔍</span> Automated EDA
                            </h4>
                            <p style="color: var(--text-secondary); line-height: 1.8;">
                                Comprehensive exploratory data analysis with trend detection, seasonality identification, 
                                and stationarity testing performed automatically.
                            </p>
                        </div>
                        <div>
                            <h4 style="color: var(--light-blue); margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem;">
                                <span style="font-size: 1.5rem;">📊</span> Interactive Visualizations
                            </h4>
                            <p style="color: var(--text-secondary); line-height: 1.8;">
                                Real-time, interactive charts powered by Plotly and Chart.js for intuitive data 
                                exploration and result interpretation.
                            </p>
                        </div>
                        <div>
                            <h4 style="color: var(--light-blue); margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem;">
                                <span style="font-size: 1.5rem;">⚡</span> Fast Processing
                            </h4>
                            <p style="color: var(--text-secondary); line-height: 1.8;">
                                Optimized backend processing ensures quick model training and forecast generation, 
                                even for large datasets.
                            </p>
                        </div>
                        <div>
                            <h4 style="color: var(--light-blue); margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem;">
                                <span style="font-size: 1.5rem;">📥</span> Export Options
                            </h4>
                            <p style="color: var(--text-secondary); line-height: 1.8;">
                                Download forecast results in CSV or JSON format for seamless integration with 
                                external systems and reporting tools.
                            </p>
                        </div>
                        <div>
                            <h4 style="color: var(--light-blue); margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem;">
                                <span style="font-size: 1.5rem;">🔄</span> Pipeline Management
                            </h4>
                            <p style="color: var(--text-secondary); line-height: 1.8;">
                                Create and manage multiple forecasting pipelines with different configurations 
                                and models for various use cases.
                            </p>
                        </div>
                    </div>
                </div>

                <!-- Available Models -->
                <div class="card" style="animation: slideInUp 0.8s ease;">
                    <h2 class="card-title">Available Forecasting Models</h2>
                    <div class="grid-2" style="gap: 2rem;">
                        ${CONFIG.MODELS.AVAILABLE.map(model => `
                            <div style="padding: 1.5rem; background: rgba(30, 90, 150, 0.1); border-left: 4px solid var(--bright-blue); border-radius: 8px;">
                                <h4 style="color: var(--light-blue); margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem;">
                                    <span style="font-size: 1.5rem;">${model.icon}</span> ${model.name}
                                </h4>
                                <p style="color: var(--text-secondary); line-height: 1.8; margin-bottom: 0.5rem;">
                                    ${model.description}
                                </p>
                                ${model.requiresExogenous ? 
                                    '<span style="font-size: 0.85rem; color: var(--warning);">* Requires exogenous variables</span>' 
                                    : ''}
                            </div>
                        `).join('')}
                    </div>
                </div>

                <!-- Technology Stack -->
                <div class="card" style="animation: slideInUp 0.9s ease;">
                    <h2 class="card-title">Technology Stack</h2>
                    <div class="grid-3">
                        <div style="text-align: center; padding: 2rem;">
                            <div style="font-size: 3rem; margin-bottom: 1rem;">🎨</div>
                            <h4 style="color: var(--light-blue); margin-bottom: 0.5rem;">Frontend</h4>
                            <p style="color: var(--text-secondary); font-size: 0.9rem;">
                                HTML5, CSS3, Vanilla JavaScript<br>
                                Chart.js, Plotly.js
                            </p>
                        </div>
                        <div style="text-align: center; padding: 2rem;">
                            <div style="font-size: 3rem; margin-bottom: 1rem;">⚙️</div>
                            <h4 style="color: var(--light-blue); margin-bottom: 0.5rem;">Backend</h4>
                            <p style="color: var(--text-secondary); font-size: 0.9rem;">
                                Python, FastAPI<br>
                                REST API Architecture
                            </p>
                        </div>
                        <div style="text-align: center; padding: 2rem;">
                            <div style="font-size: 3rem; margin-bottom: 1rem;">🤖</div>
                            <h4 style="color: var(--light-blue); margin-bottom: 0.5rem;">ML Libraries</h4>
                            <p style="color: var(--text-secondary); font-size: 0.9rem;">
                                Statsmodels, scikit-learn<br>
                                PyTorch/TensorFlow
                            </p>
                        </div>
                    </div>
                </div>

                <!-- Team Section -->
                <div class="card" style="animation: slideInUp 1s ease;">
                    <h2 class="card-title">Our Team</h2>
                    <p class="card-subtitle">
                        Meet the passionate team behind SURADATA
                    </p>
                    <div class="team-grid">
                        <div class="team-member">
                            <div class="team-avatar">👨‍💻</div>
                            <div class="team-name">Data Science Team</div>
                            <div class="team-role">ML Engineering & Research</div>
                        </div>
                        <div class="team-member">
                            <div class="team-avatar">👩‍💻</div>
                            <div class="team-name">Frontend Team</div>
                            <div class="team-role">UI/UX Development</div>
                        </div>
                        <div class="team-member">
                            <div class="team-avatar">👨‍🔬</div>
                            <div class="team-name">Backend Team</div>
                            <div class="team-role">API & Infrastructure</div>
                        </div>
                        <div class="team-member">
                            <div class="team-avatar">👩‍🎨</div>
                            <div class="team-name">Design Team</div>
                            <div class="team-role">Visual Design & Branding</div>
                        </div>
                    </div>
                </div>

                <!-- Use Cases -->
                <div class="card" style="animation: slideInUp 1.1s ease;">
                    <h2 class="card-title">Use Cases</h2>
                    <div class="grid-2" style="gap: 2rem;">
                        <div>
                            <h4 style="color: var(--light-blue); margin-bottom: 1rem;">📈 Business Analytics</h4>
                            <ul style="color: var(--text-secondary); line-height: 2; padding-left: 1.5rem;">
                                <li>Sales forecasting and revenue projection</li>
                                <li>Demand prediction and inventory optimization</li>
                                <li>Customer behavior analysis</li>
                                <li>Marketing campaign performance</li>
                            </ul>
                        </div>
                        <div>
                            <h4 style="color: var(--light-blue); margin-bottom: 1rem;">💰 Finance</h4>
                            <ul style="color: var(--text-secondary); line-height: 2; padding-left: 1.5rem;">
                                <li>Stock price prediction and trend analysis</li>
                                <li>Risk assessment and portfolio optimization</li>
                                <li>Credit scoring and fraud detection</li>
                                <li>Market volatility forecasting</li>
                            </ul>
                        </div>
                        <div>
                            <h4 style="color: var(--light-blue); margin-bottom: 1rem;">🏭 Operations</h4>
                            <ul style="color: var(--text-secondary); line-height: 2; padding-left: 1.5rem;">
                                <li>Production planning and capacity management</li>
                                <li>Resource allocation and scheduling</li>
                                <li>Quality control and defect prediction</li>
                                <li>Maintenance scheduling and downtime prevention</li>
                            </ul>
                        </div>
                        <div>
                            <h4 style="color: var(--light-blue); margin-bottom: 1rem;">🌐 IoT & Sensors</h4>
                            <ul style="color: var(--text-secondary); line-height: 2; padding-left: 1.5rem;">
                                <li>Energy consumption forecasting</li>
                                <li>Environmental monitoring and prediction</li>
                                <li>Anomaly detection in sensor data</li>
                                <li>Predictive maintenance for equipment</li>
                            </ul>
                        </div>
                    </div>
                </div>

                <!-- Workflow -->
                <div class="card" style="animation: slideInUp 1.2s ease;">
                    <h2 class="card-title">Complete Forecasting Workflow</h2>
                    <div style="display: flex; flex-direction: column; gap: 2rem;">
                        <div style="display: flex; align-items: start; gap: 1.5rem;">
                            <div style="min-width: 60px; height: 60px; background: linear-gradient(135deg, var(--accent-blue), var(--bright-blue)); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.5rem; font-weight: 700; color: white; box-shadow: 0 4px 20px var(--glow);">1</div>
                            <div>
                                <h4 style="color: var(--light-blue); margin-bottom: 0.5rem;">Data Upload</h4>
                                <p style="color: var(--text-secondary); line-height: 1.8;">Upload your time series data in CSV format. Configure date columns, value columns, and optional exogenous variables for advanced modeling.</p>
                            </div>
                        </div>
                        <div style="display: flex; align-items: start; gap: 1.5rem;">
                            <div style="min-width: 60px; height: 60px; background: linear-gradient(135deg, var(--accent-blue), var(--bright-blue)); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.5rem; font-weight: 700; color: white; box-shadow: 0 4px 20px var(--glow);">2</div>
                            <div>
                                <h4 style="color: var(--light-blue); margin-bottom: 0.5rem;">Exploratory Analysis</h4>
                                <p style="color: var(--text-secondary); line-height: 1.8;">Automatically analyze your data with statistical tests, trend detection, seasonality identification, and comprehensive visualizations.</p>
                            </div>
                        </div>
                        <div style="display: flex; align-items: start; gap: 1.5rem;">
                            <div style="min-width: 60px; height: 60px; background: linear-gradient(135deg, var(--accent-blue), var(--bright-blue)); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.5rem; font-weight: 700; color: white; box-shadow: 0 4px 20px var(--glow);">3</div>
                            <div>
                                <h4 style="color: var(--light-blue); margin-bottom: 0.5rem;">Model Training</h4>
                                <p style="color: var(--text-secondary); line-height: 1.8;">Select from multiple forecasting models, configure training parameters, and train models with automatic hyperparameter optimization.</p>
                            </div>
                        </div>
                        <div style="display: flex; align-items: start; gap: 1.5rem;">
                            <div style="min-width: 60px; height: 60px; background: linear-gradient(135deg, var(--accent-blue), var(--bright-blue)); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.5rem; font-weight: 700; color: white; box-shadow: 0 4px 20px var(--glow);">4</div>
                            <div>
                                <h4 style="color: var(--light-blue); margin-bottom: 0.5rem;">Forecast Generation</h4>
                                <p style="color: var(--text-secondary); line-height: 1.8;">Generate forecasts with confidence intervals, visualize predictions, and export results in multiple formats for further analysis.</p>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Contact & Support -->
                <div class="card" style="animation: slideInUp 1.3s ease;">
                    <h2 class="card-title">Contact & Support</h2>
                    <div style="text-align: center; padding: 2rem;">
                        <p style="color: var(--text-secondary); line-height: 1.8; margin-bottom: 2rem;">
                            Have questions or need assistance? Our team is here to help you get the most out of SURADATA.
                        </p>
                        <div style="display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap;">
                            <button class="btn btn-primary" onclick="AboutPage.openEmail()">
                                📧 Email Support
                            </button>
                            <button class="btn btn-secondary" onclick="AboutPage.openDocs()">
                                📚 Documentation
                            </button>
                            <button class="btn btn-secondary" onclick="AboutPage.openGithub()">
                                🔗 GitHub
                            </button>
                        </div>
                    </div>
                </div>

                <!-- Version Info -->
                <div style="text-align: center; margin-top: 2rem; padding: 2rem; color: var(--text-secondary); font-size: 0.9rem;">
                    <p style="margin-bottom: 0.5rem;">SURADATA Platform v1.0.0</p>
                    <p>© 2025 SURADATA. All rights reserved.</p>
                    <p style="margin-top: 1rem; font-size: 0.85rem; opacity: 0.7;">
                        Built with ❤️ for data scientists and business analysts
                    </p>
                </div>
            </div>
        `;
    },

    /**
     * Open email client
     */
    openEmail() {
        window.location.href = 'mailto:support@suradata.com?subject=SURADATA Support Request';
    },

    /**
     * Open documentation
     */
    openDocs() {
        Utils.showToast('Documentation coming soon!', 'info');
    },

    /**
     * Open GitHub repository
     */
    openGithub() {
        Utils.showToast('GitHub repository coming soon!', 'info');
    }
};

// Make AboutPage available globally
window.AboutPage = AboutPage;