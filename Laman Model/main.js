// ==================== MAIN APPLICATION (FIXED & OPTIMIZED) ====================

const App = {
    initialized: false,
    initTimeout: null,

    async init() {
        if (this.initialized) {
            console.log('App: Already initialized');
            return;
        }

        console.log('%c=== SURADATA INITIALIZATION ===', 'color: #2196F3; font-size: 16px; font-weight: bold;');
        console.log('Timestamp:', new Date().toISOString());

        try {
            // STEP 1: Force hide any stuck loading overlay
            Utils.forceHideLoading();
            
            // STEP 2: Setup error handlers FIRST
            this.setupErrorHandlers();
            console.log('✓ Error handlers registered');

            // STEP 3: Setup keyboard shortcuts
            this.setupKeyboardShortcuts();
            console.log('✓ Keyboard shortcuts enabled');

            // STEP 4: Initialize navigation (this will render HomePage)
            Navigation.init();
            console.log('✓ Navigation initialized');

            // STEP 5: Load saved state
            this.loadSavedState();
            console.log('✓ State loaded');

            // STEP 6: Mark as initialized BEFORE async operations
            this.initialized = true;
            console.log('✓ Application initialized');

            // STEP 7: Start health check in background (non-blocking)
            this.scheduleHealthCheck();

            // SUCCESS
            console.log('%c=== INITIALIZATION COMPLETE ===', 'color: #4CAF50; font-size: 16px; font-weight: bold;');

        } catch (error) {
            console.error('%cCRITICAL INITIALIZATION ERROR', 'color: #f44336; font-size: 16px; font-weight: bold;', error);
            Utils.forceHideLoading();
            Utils.showToast('Application initialization failed', 'error');
        }
    },

    scheduleHealthCheck() {
        // Run health check after 2 seconds to avoid blocking
        this.initTimeout = setTimeout(() => {
            console.log('App: Running background health check...');
            this.performHealthCheck()
                .then(() => console.log('✓ Health check completed'))
                .catch(err => {
                    console.warn('Health check failed (non-critical):', err.message);
                    this.updateConnectionStatus(false);
                });
        }, 2000);
    },

    async performHealthCheck() {
        try {
            const result = await Promise.race([
                API.healthCheck(),
                new Promise((_, reject) => 
                    setTimeout(() => reject(new Error('Health check timeout')), 5000)
                )
            ]);
            
            if (result && result.success) {
                console.log('Health check: OK');
                this.updateConnectionStatus(true);
            } else {
                console.warn('Health check: Failed');
                this.updateConnectionStatus(false);
            }
        } catch (error) {
            console.warn('Health check error:', error.message);
            this.updateConnectionStatus(false);
        }
    },

    updateConnectionStatus(isOnline) {
        // Try immediate update
        const statusText = Utils.getEl('statusText');
        const statusDot = document.querySelector('.status-dot');

        if (statusText && statusDot) {
            statusText.textContent = isOnline ? 'System Online' : 'System Offline';
            statusDot.style.background = isOnline ? 'var(--success)' : 'var(--error)';
        } else {
            // Elements not ready, retry once after delay
            setTimeout(() => {
                const statusText = Utils.getEl('statusText');
                const statusDot = document.querySelector('.status-dot');
                
                if (statusText && statusDot) {
                    statusText.textContent = isOnline ? 'System Online' : 'System Offline';
                    statusDot.style.background = isOnline ? 'var(--success)' : 'var(--error)';
                }
            }, 1000);
        }
    },

    setupErrorHandlers() {
        // Global error handler
        window.addEventListener('error', (event) => {
            console.error('Global error:', event.error);
            // Don't show toast for every error - too intrusive
        });

        // Unhandled promise rejection handler
        window.addEventListener('unhandledrejection', (event) => {
            console.error('Unhandled promise rejection:', event.reason);
            // Log but don't interrupt user experience
        });

        // Network status handlers
        window.addEventListener('online', () => {
            console.log('Network: Connection restored');
            Utils.showToast('Connection restored', 'success');
            this.performHealthCheck().catch(() => {});
        });

        window.addEventListener('offline', () => {
            console.log('Network: Connection lost');
            Utils.showToast('Connection lost', 'warning');
            this.updateConnectionStatus(false);
        });
    },

    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + H: Home
            if ((e.ctrlKey || e.metaKey) && e.key === 'h') {
                e.preventDefault();
                Navigation.navigateTo('home');
            }

            // Ctrl/Cmd + U: Upload
            if ((e.ctrlKey || e.metaKey) && e.key === 'u') {
                e.preventDefault();
                Navigation.navigateTo('upload');
            }

            // Ctrl/Cmd + E: Explore
            if ((e.ctrlKey || e.metaKey) && e.key === 'e') {
                e.preventDefault();
                Navigation.navigateTo('explore');
            }

            // Ctrl/Cmd + M: Modeling
            if ((e.ctrlKey || e.metaKey) && e.key === 'm') {
                e.preventDefault();
                Navigation.navigateTo('modeling');
            }

            // Escape: Hide loading overlay
            if (e.key === 'Escape') {
                Utils.forceHideLoading();
            }
        });
    },

    loadSavedState() {
        console.log('Loading saved state...');

        const pipeline = Utils.getFromMemory('current_pipeline');
        if (pipeline) {
            console.log('  Found pipeline:', pipeline.pipeline_name || pipeline.pipeline_id);
        }

        const exploration = Utils.getFromMemory('exploration_results');
        if (exploration) {
            console.log('  Found exploration results');
        }

        const models = Utils.getFromMemory('trained_models');
        if (models) {
            console.log('  Found trained models:', Object.keys(models).length);
        }

        const forecast = Utils.getFromMemory('forecast_results');
        if (forecast) {
            console.log('  Found forecast results');
        }

        if (!pipeline && !exploration && !models && !forecast) {
            console.log('  No saved state (starting fresh)');
        }
    },

    clearData() {
        if (confirm('This will clear all your data. Are you sure?')) {
            Utils.removeFromMemory('current_pipeline');
            Utils.removeFromMemory('exploration_results');
            Utils.removeFromMemory('detailed_exploration');
            Utils.removeFromMemory('trained_models');
            Utils.removeFromMemory('forecast_results');
            Utils.removeFromMemory('total_forecasts');
            
            Utils.showToast('All data cleared', 'success');
            Navigation.navigateTo('home');
            
            console.log('Application data cleared');
        }
    },

    exportState() {
        const state = {
            pipeline: Utils.getFromMemory('current_pipeline'),
            exploration: Utils.getFromMemory('exploration_results'),
            detailed_exploration: Utils.getFromMemory('detailed_exploration'),
            models: Utils.getFromMemory('trained_models'),
            forecast: Utils.getFromMemory('forecast_results'),
            timestamp: new Date().toISOString(),
            version: '1.0.0'
        };

        const filename = `suradata_state_${new Date().toISOString().split('T')[0]}.json`;
        Utils.downloadJSON(state, filename);
        
        console.log('Application state exported');
    },

    restart() {
        if (confirm('This will reload the page. Any unsaved data will be lost. Continue?')) {
            location.reload();
        }
    },

    cleanup() {
        if (this.initTimeout) {
            clearTimeout(this.initTimeout);
            this.initTimeout = null;
        }
    }
};

// Developer Tools
const DevTools = {
    getState() {
        return {
            initialized: App.initialized,
            currentPage: Navigation.getCurrentPage(),
            pipeline: Utils.getFromMemory('current_pipeline'),
            exploration: Utils.getFromMemory('exploration_results'),
            models: Utils.getFromMemory('trained_models'),
            forecast: Utils.getFromMemory('forecast_results'),
            memory: Utils.getMemoryStats()
        };
    },

    clear() {
        App.clearData();
    },

    export() {
        App.exportState();
    },

    goto(page) {
        Navigation.navigateTo(page);
    },

    async testAPI() {
        console.log('Testing API connection...');
        const result = await API.healthCheck();
        console.log('API Health:', result);
        return result;
    },

    showMemory() {
        console.log('Memory Storage:', Utils.memoryStorage);
        console.log('Stats:', Utils.getMemoryStats());
    },

    forceHideLoading() {
        Utils.forceHideLoading();
        console.log('Loading overlay force hidden');
    },

    help() {
        console.log(`
╔═══════════════════════════════════════════════════════════╗
║     SURADATA Developer Tools                              ║
╚═══════════════════════════════════════════════════════════╝

Available Commands:
──────────────────────────────────────────────────────────────
DevTools.getState()      : Get current application state
DevTools.clear()         : Clear all application data
DevTools.export()        : Export application state to JSON
DevTools.goto(page)      : Navigate to page
DevTools.testAPI()       : Test API connection
DevTools.showMemory()    : Show memory storage contents
DevTools.forceHideLoading() : Force hide stuck loading overlay
DevTools.help()          : Show this help message

Global Objects:
──────────────────────────────────────────────────────────────
App                      : Main application controller
Navigation               : Navigation system
Utils                    : Utility functions
API                      : API interface
Charts                   : Chart rendering functions
CONFIG                   : Application configuration

Quick Commands (via suradata object):
──────────────────────────────────────────────────────────────
suradata.goto('page')    : Navigate to page
suradata.refresh()       : Refresh current page
suradata.clear()         : Clear all data
suradata.export()        : Export state
suradata.restart()       : Restart application
suradata.state()         : Get current state
suradata.help()          : Show this help

Keyboard Shortcuts:
──────────────────────────────────────────────────────────────
Ctrl/Cmd + H             : Go to Home
Ctrl/Cmd + U             : Go to Upload
Ctrl/Cmd + E             : Go to Explore
Ctrl/Cmd + M             : Go to Modeling
ESC                      : Force hide loading overlay

Example Usage:
──────────────────────────────────────────────────────────────
DevTools.getState()              // View current state
DevTools.goto('modeling')        // Navigate to modeling
DevTools.testAPI()               // Test API connection
suradata.clear()                 // Clear all data
DevTools.forceHideLoading()      // Fix stuck loading
        `);
    }
};

// Make App and DevTools available globally
window.App = App;
window.DevTools = DevTools;

// Helper methods for console access
window.suradata = {
    version: '1.0.0',
    goto: (page) => Navigation.navigateTo(page),
    refresh: () => Navigation.refresh(),
    clear: () => App.clearData(),
    export: () => App.exportState(),
    restart: () => App.restart(),
    state: () => DevTools.getState(),
    help: () => DevTools.help(),
    fix: () => {
        Utils.forceHideLoading();
        console.log('Force hidden loading overlay');
    }
};

// ==================== AUTO-INITIALIZATION ====================

console.log('%cSURADATA Platform v1.0.0', 'color: #2196F3; font-size: 20px; font-weight: bold;');
console.log('%cAdvanced Time Series Forecasting Platform', 'color: #64B5F6; font-size: 14px;');
console.log('%cType suradata.help() for available commands', 'color: #90CAF9; font-size: 12px;');
console.log('');

// Initialize when DOM is ready
function initializeApp() {
    console.log('DOM State:', document.readyState);
    
    if (document.readyState === 'loading') {
        console.log('Waiting for DOM...');
        document.addEventListener('DOMContentLoaded', () => {
            console.log('DOM loaded, initializing...');
            App.init();
        });
    } else {
        console.log('DOM already loaded, initializing...');
        // Small delay to ensure all scripts are loaded
        setTimeout(() => {
            App.init();
        }, 100);
    }
}

// Start initialization
initializeApp();

// Periodic health check (every 5 minutes)
setInterval(() => {
    if (App.initialized) {
        console.log('Periodic health check...');
        App.performHealthCheck().catch(() => {});
    }
}, 5 * 60 * 1000);

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    App.cleanup();
    HomePage.cleanup();
});

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { App, DevTools, Navigation, Utils, API, Charts };
}

console.log('✓ Main script loaded');