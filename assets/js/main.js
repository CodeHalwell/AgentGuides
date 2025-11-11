// Agent Guides - Main JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Add smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                e.preventDefault();
                target.scrollIntoView({
                    behavior: 'smooth'
                });
            }
        });
    });

    // Add syntax highlighting for code blocks
    highlightCode();

    // Add table of contents for long pages
    generateTableOfContents();
});

function highlightCode() {
    const codeBlocks = document.querySelectorAll('pre code');
    codeBlocks.forEach(block => {
        // Add line numbers
        const lines = block.textContent.split('\n');
        let numberedContent = '';
        lines.forEach((line, index) => {
            if (line.trim() !== '') {
                numberedContent += `<span class="line-number">${index + 1}</span>${line}\n`;
            } else {
                numberedContent += line + '\n';
            }
        });
        block.innerHTML = numberedContent;
    });
}

function generateTableOfContents() {
    const headings = document.querySelectorAll('h2, h3');
    if (headings.length < 3) return;

    const toc = document.createElement('div');
    toc.className = 'table-of-contents';
    toc.innerHTML = '<h3>Table of Contents</h3><ul>';

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
        toc.querySelector('ul').appendChild(li);
    });

    toc.innerHTML += '</ul>';

    const firstH2 = document.querySelector('h2');
    if (firstH2) {
        firstH2.parentNode.insertBefore(toc, firstH2);
    }
}

// Copy button for code blocks
document.querySelectorAll('pre').forEach(pre => {
    const button = document.createElement('button');
    button.className = 'copy-button';
    button.textContent = '📋 Copy';
    button.addEventListener('click', () => {
        const code = pre.querySelector('code').textContent;
        navigator.clipboard.writeText(code).then(() => {
            button.textContent = '✓ Copied!';
            setTimeout(() => {
                button.textContent = '📋 Copy';
            }, 2000);
        });
    });
    pre.appendChild(button);
});

