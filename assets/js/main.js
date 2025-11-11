// Agent Guides - Enhanced Main JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // ===== SYNTAX HIGHLIGHTING =====
    highlightCode();
    
    // ===== SMOOTH SCROLLING =====
    smoothScroll();
    
    // ===== COPY BUTTON FUNCTIONALITY =====
    setupCopyButtons();
    
    // ===== TABLE OF CONTENTS =====
    generateTableOfContents();
    
    // ===== CODE TAB FUNCTIONALITY =====
    setupCodeTabs();
    
    // ===== SIDEBAR TOGGLE =====
    setupSidebar();
});

// ===== SYNTAX HIGHLIGHTING =====
function highlightCode() {
    // Use Highlight.js if available
    if (typeof hljs !== 'undefined') {
        document.querySelectorAll('pre code').forEach((block) => {
            hljs.highlightElement(block);
        });
    }
    
    // Add language badges and line numbers
    document.querySelectorAll('pre').forEach((pre, index) => {
        // Detect language from code fence or class
        const code = pre.querySelector('code');
        let language = 'code';
        
        if (code && code.classList.length > 0) {
            // Extract language from class like 'language-python'
            const langClass = Array.from(code.classList).find(cls => cls.startsWith('language-'));
            if (langClass) {
                language = langClass.replace('language-', '').toUpperCase();
            } else if (code.classList.contains('hljs-python')) {
                language = 'PYTHON';
            } else if (code.classList.contains('hljs-bash')) {
                language = 'BASH';
            }
        }
        
        // Set data-lang attribute for CSS badge
        pre.setAttribute('data-lang', language);
    });
}

// ===== SMOOTH SCROLLING FOR ANCHOR LINKS =====
function smoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                e.preventDefault();
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// ===== ENHANCED COPY BUTTON FUNCTIONALITY =====
function setupCopyButtons() {
    document.querySelectorAll('pre').forEach(pre => {
        // Skip if button already exists
        if (pre.querySelector('.copy-button')) return;
        
        // Create copy button
        const button = document.createElement('button');
        button.className = 'copy-button';
        button.innerHTML = 'Copy';
        button.setAttribute('aria-label', 'Copy code to clipboard');
        button.setAttribute('type', 'button');
        
        // Add click handler
        button.addEventListener('click', async (e) => {
            e.preventDefault();
            e.stopPropagation();
            
            const code = pre.querySelector('code');
            if (!code) return;
            
            // Get the actual code content, removing any line number elements
            let text = '';
            
            // If there are line number elements, extract only the code
            const lineNumberElements = code.querySelectorAll('.line-number');
            if (lineNumberElements.length > 0) {
                // Clone the code element and remove line numbers
                const codeClone = code.cloneNode(true);
                codeClone.querySelectorAll('.line-number, .line-numbers').forEach(el => el.remove());
                text = codeClone.textContent;
            } else {
                // Just get the text content
                text = code.textContent;
            }
            
            // Clean up the text: remove leading/trailing whitespace
            text = text.trim();
            
            // Copy to clipboard
            try {
                await navigator.clipboard.writeText(text);
                
                // Show success state
                button.classList.add('copied');
                button.innerHTML = 'Copied!';
                
                // Reset after 2 seconds
                setTimeout(() => {
                    button.classList.remove('copied');
                    button.innerHTML = 'Copy';
                }, 2000);
            } catch (err) {
                // Fallback for older browsers
                const textarea = document.createElement('textarea');
                textarea.value = text;
                textarea.style.position = 'fixed';
                textarea.style.opacity = '0';
                document.body.appendChild(textarea);
                textarea.select();
                
                try {
                    document.execCommand('copy');
                    button.classList.add('copied');
                    button.innerHTML = 'Copied!';
                    setTimeout(() => {
                        button.classList.remove('copied');
                        button.innerHTML = 'Copy';
                    }, 2000);
                } catch (e) {
                    console.error('Copy failed:', e);
                } finally {
                    document.body.removeChild(textarea);
                }
            }
        });
        
        // Add keyboard accessibility
        button.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                button.click();
            }
        });
        
        pre.appendChild(button);
    });
}

// ===== GENERATE TABLE OF CONTENTS =====
function generateTableOfContents() {
    const headings = document.querySelectorAll('h2, h3');
    if (headings.length < 3) return;
    
    const toc = document.createElement('aside');
    toc.className = 'table-of-contents';
    toc.setAttribute('role', 'navigation');
    toc.setAttribute('aria-label', 'Table of Contents');
    toc.innerHTML = '<h3>Table of Contents</h3><ul></ul>';
    
    const tocList = toc.querySelector('ul');
    
    headings.forEach((heading, index) => {
        if (!heading.id) {
            heading.id = `heading-${index}`;
        }
        
        const level = heading.tagName === 'H2' ? 0 : 1;
        const li = document.createElement('li');
        li.style.marginLeft = `${level * 1.5}rem`;
        
        const a = document.createElement('a');
        a.href = `#${heading.id}`;
        a.textContent = heading.textContent;
        
        li.appendChild(a);
        tocList.appendChild(li);
    });
    
    const firstH2 = document.querySelector('h2');
    if (firstH2) {
        firstH2.parentNode.insertBefore(toc, firstH2);
    }
}

// ===== CODE TAB FUNCTIONALITY =====
function setupCodeTabs() {
    document.querySelectorAll('.code-tabs').forEach(tabContainer => {
        const buttons = tabContainer.querySelectorAll('.tab-button');
        const contents = tabContainer.querySelectorAll('.tab-content');
        
        buttons.forEach((button, index) => {
            button.addEventListener('click', () => {
                // Remove active class from all buttons and contents
                buttons.forEach(btn => btn.classList.remove('active'));
                contents.forEach(content => content.classList.remove('active'));
                
                // Add active class to clicked button and corresponding content
                button.classList.add('active');
                contents[index].classList.add('active');
            });
        });
        
        // Set first tab as active by default
        if (buttons.length > 0) {
            buttons[0].classList.add('active');
            contents[0].classList.add('active');
        }
    });
}

// ===== ENHANCED ACCESSIBILITY =====

// Skip to main content link (hidden but accessible)
document.addEventListener('DOMContentLoaded', function() {
    if (!document.querySelector('.skip-link')) {
        const skipLink = document.createElement('a');
        skipLink.className = 'skip-link';
        skipLink.href = '#main';
        skipLink.textContent = 'Skip to main content';
        document.body.prepend(skipLink);
    }
    
    // Add main landmark if missing
    const main = document.querySelector('main');
    if (main && !main.id) {
        main.id = 'main';
    }
});

// ===== KEYBOARD NAVIGATION =====
document.addEventListener('keydown', function(e) {
    // Escape key to close any open modals/popovers
    if (e.key === 'Escape') {
        // Handle escape events here if needed
    }
    
    // Alt + C for copy (when focused on code block)
    if (e.altKey && e.key === 'c') {
        const copyButton = document.querySelector('.copy-button');
        if (copyButton && copyButton === document.activeElement) {
            copyButton.click();
        }
    }
});

// ===== PERFORMANCE MONITORING =====
function trackPerformance() {
    if (typeof PerformanceObserver !== 'undefined') {
        // Largest Contentful Paint
        new PerformanceObserver((list) => {
            for (const entry of list.getEntries()) {
                console.log('LCP:', entry.renderTime || entry.loadTime);
            }
        }).observe({ entryTypes: ['largest-contentful-paint'] });
        
        // Cumulative Layout Shift
        let cls = 0;
        new PerformanceObserver((list) => {
            for (const entry of list.getEntries()) {
                if (!entry.hadRecentInput) {
                    cls += entry.value;
                    console.log('CLS:', cls);
                }
            }
        }).observe({ entryTypes: ['layout-shift'] });
    }
}

// ===== SIDEBAR TOGGLE =====
function setupSidebar() {
    const sidebarToggle = document.getElementById('sidebarToggle');
    const sidebar = document.getElementById('sidebar');
    
    if (!sidebarToggle || !sidebar) return;
    
    // Toggle sidebar on button click
    sidebarToggle.addEventListener('click', function() {
        sidebar.classList.toggle('active');
    });
    
    // Close sidebar when clicking outside on mobile
    document.addEventListener('click', function(event) {
        const isMobile = window.innerWidth < 768;
        if (isMobile && 
            sidebar.classList.contains('active') && 
            !sidebar.contains(event.target) && 
            !sidebarToggle.contains(event.target)) {
            sidebar.classList.remove('active');
        }
    });
    
    // Highlight active page in sidebar
    const currentPath = window.location.pathname;
    document.querySelectorAll('.sidebar-menu a').forEach(link => {
        if (link.getAttribute('href') === currentPath || 
            currentPath.includes(link.getAttribute('href'))) {
            link.classList.add('active');
        }
    });
}

// ===== RESPONSIVE BEHAVIOR =====
function handleResponsive() {
    const isMobile = window.innerWidth < 768;
    
    // Adjust code block behavior on mobile
    document.querySelectorAll('pre').forEach(pre => {
        if (isMobile) {
            pre.style.fontSize = '0.85rem';
        } else {
            pre.style.fontSize = '';
        }
    });
}

window.addEventListener('resize', handleResponsive);
window.addEventListener('load', handleResponsive);
