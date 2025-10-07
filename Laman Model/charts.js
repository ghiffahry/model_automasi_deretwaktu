// ==================== CHARTS MODULE - COMPLETE VERSION ====================

const Charts = {
    /**
     * Render time series chart with Plotly - with validation
     * Used for: Basic time series visualization
     */
    renderTimeSeries(containerId, data) {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error(`[Charts] Container ${containerId} not found`);
            return;
        }
        
        // Validate data
        if (!data || !data.dates || !data.values || data.dates.length === 0) {
            container.innerHTML = '<p style="text-align:center;color:var(--text-secondary);padding:2rem;">No data available for visualization</p>';
            return;
        }

        const trace = {
            x: data.dates,
            y: data.values,
            type: 'scatter',
            mode: 'lines',
            name: data.title || 'Time Series',
            line: {
                color: data.color || '#2196F3',
                width: 2
            }
        };

        const layout = {
            ...CONFIG.CHARTS.PLOTLY_LAYOUT_DARK,
            title: {
                text: data.title || 'Time Series Plot',
                font: { color: '#f5f7fa', size: 18 }
            },
            xaxis: {
                title: 'Date',
                ...CONFIG.CHARTS.PLOTLY_LAYOUT_DARK.xaxis
            },
            yaxis: {
                title: 'Value',
                ...CONFIG.CHARTS.PLOTLY_LAYOUT_DARK.yaxis
            }
        };

        try {
            Plotly.newPlot(containerId, [trace], layout, CONFIG.CHARTS.PLOTLY_CONFIG);
            console.log(`[Charts] ✓ Time series rendered in ${containerId}`);
        } catch (error) {
            console.error('[Charts] Plotly rendering error:', error);
            container.innerHTML = '<p style="text-align:center;color:var(--error);padding:2rem;">Failed to render chart</p>';
        }
    },

    /**
     * Render line chart - with validation
     * Used for: Trend, seasonal components
     */
    renderLine(containerId, data) {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error(`[Charts] Container ${containerId} not found`);
            return;
        }
        
        if (!data || !data.dates || !data.values || data.dates.length === 0) {
            container.innerHTML = '<p style="text-align:center;color:var(--text-secondary);padding:2rem;">No data available</p>';
            return;
        }

        const trace = {
            x: data.dates,
            y: data.values,
            type: 'scatter',
            mode: 'lines',
            name: data.title || 'Line Chart',
            line: {
                color: data.color || '#4CAF50',
                width: 2
            }
        };

        const layout = {
            ...CONFIG.CHARTS.PLOTLY_LAYOUT_DARK,
            title: {
                text: data.title || 'Line Chart',
                font: { color: '#f5f7fa', size: 16 }
            }
        };

        try {
            Plotly.newPlot(containerId, [trace], layout, CONFIG.CHARTS.PLOTLY_CONFIG);
            console.log(`[Charts] ✓ Line chart rendered in ${containerId}`);
        } catch (error) {
            console.error('[Charts] Plotly rendering error:', error);
            container.innerHTML = '<p style="text-align:center;color:var(--error);padding:2rem;">Failed to render chart</p>';
        }
    },

    /**
     * Render histogram with Chart.js - with validation and cleanup
     * Used for: Distribution analysis
     */
    renderHistogram(containerId, data) {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error(`[Charts] Container ${containerId} not found`);
            return;
        }
        
        if (!data || !data.bins || !data.counts || data.bins.length === 0) {
            container.innerHTML = '<p style="text-align:center;color:var(--text-secondary);padding:2rem;">No data available</p>';
            return;
        }

        // Destroy existing chart if exists
        const existingCanvas = container.querySelector('canvas');
        if (existingCanvas) {
            const existingChart = Chart.getChart(existingCanvas);
            if (existingChart) {
                existingChart.destroy();
            }
        }

        container.innerHTML = '<canvas></canvas>';
        const canvas = container.querySelector('canvas');
        const ctx = canvas.getContext('2d');

        try {
            new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: data.bins.map(b => Utils.formatNumber(b, 1)),
                    datasets: [{
                        label: 'Frequency',
                        data: data.counts,
                        backgroundColor: 'rgba(33, 150, 243, 0.6)',
                        borderColor: 'rgba(33, 150, 243, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    ...CONFIG.CHARTS.CHART_JS_DEFAULTS,
                    plugins: {
                        ...CONFIG.CHARTS.CHART_JS_DEFAULTS.plugins,
                        title: {
                            display: true,
                            text: data.title || 'Distribution',
                            color: '#f5f7fa',
                            font: { size: 16 }
                        }
                    }
                }
            });
            console.log(`[Charts] ✓ Histogram rendered in ${containerId}`);
        } catch (error) {
            console.error('[Charts] Chart.js rendering error:', error);
            container.innerHTML = '<p style="text-align:center;color:var(--error);padding:2rem;">Failed to render chart</p>';
        }
    },

    /**
     * Render bar chart - with validation and cleanup
     * Used for: ACF basic visualization, categorical data
     */
    renderBar(containerId, data) {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error(`[Charts] Container ${containerId} not found`);
            return;
        }
        
        if (!data || !data.labels || !data.values || data.labels.length === 0) {
            container.innerHTML = '<p style="text-align:center;color:var(--text-secondary);padding:2rem;">No data available</p>';
            return;
        }

        // Destroy existing chart
        const existingCanvas = container.querySelector('canvas');
        if (existingCanvas) {
            const existingChart = Chart.getChart(existingCanvas);
            if (existingChart) {
                existingChart.destroy();
            }
        }

        container.innerHTML = '<canvas></canvas>';
        const canvas = container.querySelector('canvas');
        const ctx = canvas.getContext('2d');

        try {
            new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: data.labels,
                    datasets: [{
                        label: data.title || 'Values',
                        data: data.values,
                        backgroundColor: Utils.generateColorPalette(data.values.length).map(c => c + '99'),
                        borderColor: Utils.generateColorPalette(data.values.length),
                        borderWidth: 2
                    }]
                },
                options: {
                    ...CONFIG.CHARTS.CHART_JS_DEFAULTS,
                    plugins: {
                        ...CONFIG.CHARTS.CHART_JS_DEFAULTS.plugins,
                        title: {
                            display: true,
                            text: data.title || 'Bar Chart',
                            color: '#f5f7fa',
                            font: { size: 16 }
                        }
                    }
                }
            });
            console.log(`[Charts] ✓ Bar chart rendered in ${containerId}`);
        } catch (error) {
            console.error('[Charts] Chart.js rendering error:', error);
            container.innerHTML = '<p style="text-align:center;color:var(--error);padding:2rem;">Failed to render chart</p>';
        }
    },

    /**
     * Render forecast chart with confidence intervals - with validation
     * Used for: Model predictions with uncertainty bounds
     */
    renderForecast(containerId, data) {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error(`[Charts] Container ${containerId} not found`);
            return;
        }
        
        // Validate all required data
        if (!data || !data.historical_dates || !data.historical_values || 
            !data.forecast_dates || !data.forecast_values ||
            data.forecast_dates.length === 0) {
            container.innerHTML = '<p style="text-align:center;color:var(--text-secondary);padding:2rem;">No forecast data available</p>';
            return;
        }

        const traces = [];

        // Historical data
        if (data.historical_dates.length > 0) {
            traces.push({
                x: data.historical_dates,
                y: data.historical_values,
                type: 'scatter',
                mode: 'lines',
                name: 'Historical',
                line: {
                    color: '#2196F3',
                    width: 2
                }
            });
        }

        // Forecast data
        traces.push({
            x: data.forecast_dates,
            y: data.forecast_values,
            type: 'scatter',
            mode: 'lines',
            name: 'Forecast',
            line: {
                color: '#4CAF50',
                width: 2,
                dash: 'dash'
            }
        });

        // Confidence intervals if available
        if (data.upper_bound && data.lower_bound && 
            data.upper_bound.length === data.forecast_dates.length) {
            
            // Upper bound
            traces.push({
                x: data.forecast_dates,
                y: data.upper_bound,
                type: 'scatter',
                mode: 'lines',
                name: 'Upper Bound',
                line: {
                    color: 'rgba(76, 175, 80, 0.3)',
                    width: 1
                },
                showlegend: false
            });

            // Lower bound with fill
            traces.push({
                x: data.forecast_dates,
                y: data.lower_bound,
                type: 'scatter',
                mode: 'lines',
                name: 'Confidence Interval',
                line: {
                    color: 'rgba(76, 175, 80, 0.3)',
                    width: 1
                },
                fill: 'tonexty',
                fillcolor: 'rgba(76, 175, 80, 0.2)'
            });
        }

        const layout = {
            ...CONFIG.CHARTS.PLOTLY_LAYOUT_DARK,
            title: {
                text: 'Forecast with Confidence Intervals',
                font: { color: '#f5f7fa', size: 18 }
            },
            xaxis: {
                title: 'Date',
                ...CONFIG.CHARTS.PLOTLY_LAYOUT_DARK.xaxis
            },
            yaxis: {
                title: 'Value',
                ...CONFIG.CHARTS.PLOTLY_LAYOUT_DARK.yaxis
            },
            hovermode: 'x unified'
        };

        try {
            Plotly.newPlot(containerId, traces, layout, CONFIG.CHARTS.PLOTLY_CONFIG);
            console.log(`[Charts] ✓ Forecast chart rendered in ${containerId}`);
        } catch (error) {
            console.error('[Charts] Plotly rendering error:', error);
            container.innerHTML = '<p style="text-align:center;color:var(--error);padding:2rem;">Failed to render forecast chart</p>';
        }
    },

    /**
     * Render multiple series comparison - with validation
     * Used for: Comparing different models or datasets
     */
    renderComparison(containerId, series) {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error(`[Charts] Container ${containerId} not found`);
            return;
        }
        
        if (!series || series.length === 0) {
            container.innerHTML = '<p style="text-align:center;color:var(--text-secondary);padding:2rem;">No comparison data available</p>';
            return;
        }

        const traces = series.filter(s => s.dates && s.values && s.dates.length > 0).map(s => ({
            x: s.dates,
            y: s.values,
            type: 'scatter',
            mode: 'lines',
            name: s.name,
            line: {
                color: s.color,
                width: 2
            }
        }));

        if (traces.length === 0) {
            container.innerHTML = '<p style="text-align:center;color:var(--text-secondary);padding:2rem;">No valid data series</p>';
            return;
        }

        const layout = {
            ...CONFIG.CHARTS.PLOTLY_LAYOUT_DARK,
            title: {
                text: 'Model Comparison',
                font: { color: '#f5f7fa', size: 18 }
            },
            xaxis: {
                title: 'Date',
                ...CONFIG.CHARTS.PLOTLY_LAYOUT_DARK.xaxis
            },
            yaxis: {
                title: 'Value',
                ...CONFIG.CHARTS.PLOTLY_LAYOUT_DARK.yaxis
            }
        };

        try {
            Plotly.newPlot(containerId, traces, layout, CONFIG.CHARTS.PLOTLY_CONFIG);
            console.log(`[Charts] ✓ Comparison chart rendered in ${containerId}`);
        } catch (error) {
            console.error('[Charts] Plotly rendering error:', error);
            container.innerHTML = '<p style="text-align:center;color:var(--error);padding:2rem;">Failed to render comparison chart</p>';
        }
    },

    /**
     * Render performance metrics comparison - with validation and cleanup
     * Used for: Model performance evaluation
     */
    renderMetricsComparison(containerId, models) {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error(`[Charts] Container ${containerId} not found`);
            return;
        }
        
        if (!models || Object.keys(models).length === 0) {
            container.innerHTML = '<p style="text-align:center;color:var(--text-secondary);padding:2rem;">No model metrics available</p>';
            return;
        }

        // Destroy existing chart
        const existingCanvas = container.querySelector('canvas');
        if (existingCanvas) {
            const existingChart = Chart.getChart(existingCanvas);
            if (existingChart) {
                existingChart.destroy();
            }
        }

        container.innerHTML = '<canvas></canvas>';
        const canvas = container.querySelector('canvas');
        const ctx = canvas.getContext('2d');

        const metrics = ['rmse', 'mae', 'mape'];
        const datasets = metrics.map((metric, idx) => ({
            label: metric.toUpperCase(),
            data: Object.values(models).map(m => m[metric] || 0),
            backgroundColor: Utils.generateColorPalette(3)[idx] + '99',
            borderColor: Utils.generateColorPalette(3)[idx],
            borderWidth: 2
        }));

        try {
            new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: Object.keys(models),
                    datasets: datasets
                },
                options: {
                    ...CONFIG.CHARTS.CHART_JS_DEFAULTS,
                    plugins: {
                        ...CONFIG.CHARTS.CHART_JS_DEFAULTS.plugins,
                        title: {
                            display: true,
                            text: 'Model Performance Comparison',
                            color: '#f5f7fa',
                            font: { size: 16 }
                        }
                    }
                }
            });
            console.log(`[Charts] ✓ Metrics comparison rendered in ${containerId}`);
        } catch (error) {
            console.error('[Charts] Chart.js rendering error:', error);
            container.innerHTML = '<p style="text-align:center;color:var(--error);padding:2rem;">Failed to render metrics chart</p>';
        }
    },

    /**
     * Render residual time series plot
     * Used for: Model diagnostics - residuals over time
     */
    renderResidualPlot(containerId, data) {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error(`[Charts] Container ${containerId} not found`);
            return;
        }

        if (!data || !data.dates || !data.residuals) {
            container.innerHTML = '<p style="text-align:center;color:var(--text-secondary);padding:2rem;">No residual data available</p>';
            return;
        }

        const traces = [
            {
                x: data.dates,
                y: data.residuals,
                type: 'scatter',
                mode: 'lines',
                name: 'Residuals',
                line: {
                    color: '#2196F3',
                    width: 2
                }
            }
        ];

        const layout = {
            ...CONFIG.CHARTS.PLOTLY_LAYOUT_DARK,
            title: {
                text: 'Residuals Over Time',
                font: { color: '#f5f7fa', size: 16 }
            },
            xaxis: {
                title: 'Date',
                ...CONFIG.CHARTS.PLOTLY_LAYOUT_DARK.xaxis
            },
            yaxis: {
                title: 'Residuals',
                ...CONFIG.CHARTS.PLOTLY_LAYOUT_DARK.yaxis
            },
            shapes: [{
                type: 'line',
                x0: data.dates[0],
                x1: data.dates[data.dates.length - 1],
                y0: 0,
                y1: 0,
                line: {
                    color: '#FF5252',
                    width: 2,
                    dash: 'dash'
                }
            }]
        };

        try {
            Plotly.newPlot(containerId, traces, layout, CONFIG.CHARTS.PLOTLY_CONFIG);
            console.log(`[Charts] ✓ Residual plot rendered in ${containerId}`);
        } catch (error) {
            console.error('[Charts] Plotly rendering error:', error);
            container.innerHTML = '<p style="text-align:center;color:var(--error);padding:2rem;">Failed to render residual plot</p>';
        }
    },

    /**
     * Render residual histogram with normal distribution overlay
     * Used for: Checking normality assumption of residuals
     */
    renderResidualHistogram(containerId, data) {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error(`[Charts] Container ${containerId} not found`);
            return;
        }

        if (!data || !data.bins || !data.counts) {
            container.innerHTML = '<p style="text-align:center;color:var(--text-secondary);padding:2rem;">No histogram data available</p>';
            return;
        }

        const traces = [
            {
                x: data.bins,
                y: data.counts,
                type: 'bar',
                name: 'Residuals',
                marker: {
                    color: 'rgba(33, 150, 243, 0.6)',
                    line: {
                        color: 'rgba(33, 150, 243, 1)',
                        width: 1
                    }
                }
            }
        ];

        // Add normal distribution curve if available
        if (data.normal_x && data.normal_y) {
            traces.push({
                x: data.normal_x,
                y: data.normal_y,
                type: 'scatter',
                mode: 'lines',
                name: 'Normal Distribution',
                line: {
                    color: '#FF5252',
                    width: 3
                }
            });
        }

        const layout = {
            ...CONFIG.CHARTS.PLOTLY_LAYOUT_DARK,
            title: {
                text: 'Residual Distribution',
                font: { color: '#f5f7fa', size: 16 }
            },
            xaxis: {
                title: 'Residuals',
                ...CONFIG.CHARTS.PLOTLY_LAYOUT_DARK.xaxis
            },
            yaxis: {
                title: 'Density',
                ...CONFIG.CHARTS.PLOTLY_LAYOUT_DARK.yaxis
            },
            barmode: 'overlay'
        };

        try {
            Plotly.newPlot(containerId, traces, layout, CONFIG.CHARTS.PLOTLY_CONFIG);
            console.log(`[Charts] ✓ Residual histogram rendered in ${containerId}`);
        } catch (error) {
            console.error('[Charts] Plotly rendering error:', error);
            container.innerHTML = '<p style="text-align:center;color:var(--error);padding:2rem;">Failed to render histogram</p>';
        }
    },

    /**
     * Render ACF plot for residuals or time series
     * Used for: Identifying autocorrelation patterns, determining MA(q) order
     * Matches backend response: data.acf_original, data.acf_differenced
     */
    renderACFPlot(containerId, data) {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error(`[Charts] Container ${containerId} not found`);
            return;
        }

        console.log(`[Charts] Rendering ACF plot in ${containerId}`, data);

        if (!data || !data.lags || !data.acf) {
            console.error('[Charts] Invalid ACF data structure:', data);
            container.innerHTML = '<p style="text-align:center;color:var(--text-secondary);padding:2rem;">No ACF data available</p>';
            return;
        }

        const traces = [{
            x: data.lags,
            y: data.acf,
            type: 'bar',
            name: 'ACF',
            marker: {
                color: '#2196F3',
                line: {
                    color: '#1976D2',
                    width: 1
                }
            }
        }];

        // Calculate confidence interval
        const n = data.lags.length;
        const ci = 1.96 / Math.sqrt(n);
        
        console.log(`[Charts] ACF confidence interval: ±${ci.toFixed(4)}`);

        const layout = {
            ...CONFIG.CHARTS.PLOTLY_LAYOUT_DARK,
            title: {
                text: 'Autocorrelation Function (ACF)',
                font: { color: '#f5f7fa', size: 16 }
            },
            xaxis: {
                title: 'Lag',
                ...CONFIG.CHARTS.PLOTLY_LAYOUT_DARK.xaxis,
                dtick: 1
            },
            yaxis: {
                title: 'Autocorrelation',
                ...CONFIG.CHARTS.PLOTLY_LAYOUT_DARK.yaxis,
                range: [-1, 1]
            },
            shapes: [
                {
                    type: 'line',
                    x0: 0,
                    x1: data.lags[data.lags.length - 1],
                    y0: ci,
                    y1: ci,
                    line: {
                        color: '#FF5252',
                        width: 2,
                        dash: 'dash'
                    }
                },
                {
                    type: 'line',
                    x0: 0,
                    x1: data.lags[data.lags.length - 1],
                    y0: -ci,
                    y1: -ci,
                    line: {
                        color: '#FF5252',
                        width: 2,
                        dash: 'dash'
                    }
                },
                {
                    type: 'line',
                    x0: 0,
                    x1: data.lags[data.lags.length - 1],
                    y0: 0,
                    y1: 0,
                    line: {
                        color: 'rgba(255, 255, 255, 0.3)',
                        width: 1
                    }
                }
            ]
        };

        try {
            Plotly.newPlot(containerId, traces, layout, CONFIG.CHARTS.PLOTLY_CONFIG);
            console.log(`[Charts] ✓ ACF plot rendered in ${containerId}`);
        } catch (error) {
            console.error('[Charts] Plotly rendering error:', error);
            container.innerHTML = '<p style="text-align:center;color:var(--error);padding:2rem;">Failed to render ACF plot</p>';
        }
    },

    /**
     * Render PACF plot
     * Used for: Identifying partial autocorrelation patterns, determining AR(p) order
     * Matches backend response: data.pacf_original, data.pacf_differenced
     */
    renderPACFPlot(containerId, data) {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error(`[Charts] Container ${containerId} not found`);
            return;
        }

        console.log(`[Charts] Rendering PACF plot in ${containerId}`, data);

        if (!data || !data.lags || !data.values) {
            console.error('[Charts] Invalid PACF data structure:', data);
            container.innerHTML = '<p style="text-align:center;color:var(--text-secondary);padding:2rem;">No PACF data available</p>';
            return;
        }

        const traces = [{
            x: data.lags,
            y: data.values,
            type: 'bar',
            name: 'PACF',
            marker: {
                color: '#FF9800',
                line: {
                    color: '#F57C00',
                    width: 1
                }
            }
        }];

        // Calculate confidence interval
        const n = data.lags.length;
        const ci = 1.96 / Math.sqrt(n);
        
        console.log(`[Charts] PACF confidence interval: ±${ci.toFixed(4)}`);

        const layout = {
            ...CONFIG.CHARTS.PLOTLY_LAYOUT_DARK,
            title: {
                text: 'Partial Autocorrelation Function (PACF)',
                font: { color: '#f5f7fa', size: 16 }
            },
            xaxis: {
                title: 'Lag',
                ...CONFIG.CHARTS.PLOTLY_LAYOUT_DARK.xaxis,
                dtick: 1
            },
            yaxis: {
                title: 'Partial Autocorrelation',
                ...CONFIG.CHARTS.PLOTLY_LAYOUT_DARK.yaxis,
                range: [-1, 1]
            },
            shapes: [
                {
                    type: 'line',
                    x0: 0,
                    x1: data.lags[data.lags.length - 1],
                    y0: ci,
                    y1: ci,
                    line: {
                        color: '#FF5252',
                        width: 2,
                        dash: 'dash'
                    }
                },
                {
                    type: 'line',
                    x0: 0,
                    x1: data.lags[data.lags.length - 1],
                    y0: -ci,
                    y1: -ci,
                    line: {
                        color: '#FF5252',
                        width: 2,
                        dash: 'dash'
                    }
                },
                {
                    type: 'line',
                    x0: 0,
                    x1: data.lags[data.lags.length - 1],
                    y0: 0,
                    y1: 0,
                    line: {
                        color: 'rgba(255, 255, 255, 0.3)',
                        width: 1
                    }
                }
            ]
        };

        try {
            Plotly.newPlot(containerId, traces, layout, CONFIG.CHARTS.PLOTLY_CONFIG);
            console.log(`[Charts] ✓ PACF plot rendered in ${containerId}`);
        } catch (error) {
            console.error('[Charts] Plotly rendering error:', error);
            container.innerHTML = '<p style="text-align:center;color:var(--error);padding:2rem;">Failed to render PACF plot</p>';
        }
    },

    /**
     * Render time series comparison (original vs differenced)
     * Used for: Visualizing effect of differencing on time series
     * Matches backend response: data.original, data.differenced
     */
    renderTimeSeriesComparison(containerId, originalData, differencedData) {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error(`[Charts] Container ${containerId} not found`);
            return;
        }

        console.log('[Charts] Rendering time series comparison');
        console.log('[Charts] Original data points:', originalData?.dates?.length);
        console.log('[Charts] Differenced data points:', differencedData?.dates?.length);

        if (!originalData || !differencedData) {
            console.error('[Charts] Missing data for comparison');
            container.innerHTML = '<p style="text-align:center;color:var(--text-secondary);padding:2rem;">No data available</p>';
            return;
        }

        if (!originalData.dates || !originalData.values || !differencedData.dates || !differencedData.values) {
            console.error('[Charts] Invalid data structure for comparison');
            container.innerHTML = '<p style="text-align:center;color:var(--text-secondary);padding:2rem;">Invalid data structure</p>';
            return;
        }

        const traces = [
            {
                x: originalData.dates,
                y: originalData.values,
                type: 'scatter',
                mode: 'lines',
                name: 'Original Series',
                line: {
                    color: '#2196F3',
                    width: 2
                },
                yaxis: 'y1'
            },
            {
                x: differencedData.dates,
                y: differencedData.values,
                type: 'scatter',
                mode: 'lines',
                name: 'Differenced Series',
                line: {
                    color: '#4CAF50',
                    width: 2
                },
                yaxis: 'y2'
            }
        ];

        const layout = {
            ...CONFIG.CHARTS.PLOTLY_LAYOUT_DARK,
            title: {
                text: 'Original vs Differenced Series',
                font: { color: '#f5f7fa', size: 16 }
            },
            xaxis: {
                title: 'Date',
                ...CONFIG.CHARTS.PLOTLY_LAYOUT_DARK.xaxis,
                domain: [0, 1]
            },
            yaxis: {
                title: 'Original',
                titlefont: { color: '#2196F3' },
                tickfont: { color: '#2196F3' },
                ...CONFIG.CHARTS.PLOTLY_LAYOUT_DARK.yaxis
            },
            yaxis2: {
                title: 'Differenced',
                titlefont: { color: '#4CAF50' },
                tickfont: { color: '#4CAF50' },
                overlaying: 'y',
                side: 'right',
                gridcolor: 'rgba(255, 255, 255, 0.05)',
                color: '#f5f7fa'
            },
            legend: {
                x: 0.5,
                xanchor: 'center',
                y: 1.15,
                orientation: 'h',
                bgcolor: 'rgba(0,0,0,0.5)',
                font: { color: '#f5f7fa' }
            }
        };

        try {
            Plotly.newPlot(containerId, traces, layout, CONFIG.CHARTS.PLOTLY_CONFIG);
            console.log(`[Charts] ✓ Time series comparison rendered in ${containerId}`);
        } catch (error) {
            console.error('[Charts] Plotly rendering error:', error);
            container.innerHTML = '<p style="text-align:center;color:var(--error);padding:2rem;">Failed to render comparison</p>';
        }
    },

    /**
     * Clear chart safely
     * Used for: Cleanup before re-rendering
     */
    clearChart(containerId) {
        const container = document.getElementById(containerId);
        if (container) {
            // Destroy Chart.js instance if exists
            const canvas = container.querySelector('canvas');
            if (canvas) {
                const chart = Chart.getChart(canvas);
                if (chart) {
                    chart.destroy();
                    console.log(`[Charts] ✓ Chart.js instance destroyed in ${containerId}`);
                }
            }
            
            // Clear Plotly if exists
            try {
                Plotly.purge(containerId);
                console.log(`[Charts] ✓ Plotly instance purged in ${containerId}`);
            } catch (e) {
                // Not a Plotly chart, ignore
            }
            
            container.innerHTML = '';
        }
    },

    /**
     * Validate chart data structure
     * Used for: Pre-render validation
     */
    validateData(data, requiredFields) {
        if (!data) {
            console.error('[Charts] Data is null or undefined');
            return false;
        }

        for (const field of requiredFields) {
            if (!data[field]) {
                console.error(`[Charts] Missing required field: ${field}`);
                return false;
            }
            if (Array.isArray(data[field]) && data[field].length === 0) {
                console.error(`[Charts] Empty array for field: ${field}`);
                return false;
            }
        }

        return true;
    },

    /**
     * Get chart instance
     * Used for: Accessing existing charts for updates
     */
    getChartInstance(containerId) {
        const container = document.getElementById(containerId);
        if (!container) return null;

        // Check for Chart.js instance
        const canvas = container.querySelector('canvas');
        if (canvas) {
            const chart = Chart.getChart(canvas);
            if (chart) return { type: 'chartjs', instance: chart };
        }

        // Check for Plotly instance
        try {
            const plotlyDiv = container.querySelector('.plotly');
            if (plotlyDiv) return { type: 'plotly', instance: plotlyDiv };
        } catch (e) {
            // No Plotly instance
        }

        return null;
    }
};

// Make Charts available globally
window.Charts = Charts;
console.log('✓ Charts module loaded');