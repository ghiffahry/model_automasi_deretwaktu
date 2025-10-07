// ==================== NAVIGATION SYSTEM ====================

const Navigation = {
    currentPage: 'home',
    pages: {},
    
    /**
     * Initialize navigation system
     */
    init() {
        // Register all pages
        this.pages = {
            home: HomePage,
            upload: UploadPage,
            explore: ExplorePage,
            modeling: ModelingPage,
            about: AboutPage
        };
        
        // Setup navigation button listeners
        const navButtons = Utils.qsa('.nav-btn');
        navButtons.forEach(btn => {
            btn.addEventListener('click', (e) => {
                const pageName = e.currentTarget.dataset.page;
                this.navigateTo(pageName);
            });
        });
        
        // Load initial page
        this.navigateTo('home');
    },
    
    /**
     * Navigate to specific page
     */
    async navigateTo(pageName) {
        if (this.currentPage === pageName) return;
        
        // Update navigation buttons
        Utils.qsa('.nav-btn').forEach(btn => {
            btn.classList.remove('active');
            if (btn.dataset.page === pageName) {
                btn.classList.add('active');
            }
        });
        
        // Get page handler
        const page = this.pages[pageName];
        if (!page) {
            console.error('Page not found:', pageName);
            return;
        }
        
        // Clear container
        const container = Utils.getEl('mainContainer');
        container.innerHTML = '';
        
        // Render new page (await if async)
        try {
            if (page.render.constructor.name === 'AsyncFunction') {
                await page.render();
            } else {
                page.render();
            }
            
            this.currentPage = pageName;
            
            // Scroll to top
            window.scrollTo({ top: 0, behavior: 'smooth' });
        } catch (error) {
            console.error('Error rendering page:', error);
            Utils.showToast('Failed to load page', 'error');
        }
    },
    
    /**
     * Refresh current page
     */
    async refresh() {
        const page = this.pages[this.currentPage];
        if (page && page.refresh) {
            if (page.refresh.constructor.name === 'AsyncFunction') {
                await page.refresh();
            } else {
                page.refresh();
            }
        }
    },
    
    /**
     * Get current page name
     */
    getCurrentPage() {
        return this.currentPage;
    }
};

// Make Navigation available globally
window.Navigation = Navigation;