import os
import re
import urllib.parse

def find_markdown_files(root_dir):
    markdown_files = []
    for root, dirs, files in os.walk(root_dir):
        if '.git' in dirs:
            dirs.remove('.git')
        if '_site' in dirs:
            dirs.remove('_site')
        
        for file in files:
            if file.endswith('.md'):
                markdown_files.append(os.path.join(root, file))
    return markdown_files

def validate_links(root_dir):
    markdown_files = find_markdown_files(root_dir)
    broken_links = []
    
    link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
    
    print(f"Scanning {len(markdown_files)} markdown files...")
    
    for file_path in markdown_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            matches = link_pattern.findall(content)
            for text, target in matches:
                # Skip external links, anchors, and mailto
                if target.startswith(('http://', 'https://', 'mailto:', '#')):
                    continue
                
                # Handle anchors in local links (e.g., file.md#section)
                target_path = target.split('#')[0]
                if not target_path: # Just an anchor like #section
                    continue
                
                # Resolve absolute paths (relative to repo root) vs relative paths
                if target_path.startswith('/'):
                    # Assuming / refers to root of repo for checking purposes, though in Jekyll it might be different
                    # Let's treat it as relative to root_dir
                    abs_target_path = os.path.join(root_dir, target_path.lstrip('/'))
                else:
                    abs_target_path = os.path.join(os.path.dirname(file_path), target_path)
                
                # Clean up path (handle .. and .)
                abs_target_path = os.path.normpath(abs_target_path)
                
                if not os.path.exists(abs_target_path):
                    # Check if it might be a directory without trailing slash or vice versa
                    if os.path.exists(abs_target_path + '.md'):
                         # It's a file link without extension, which might be valid in some SSGs but we want .md
                         pass # We might want to flag this as "missing extension" but strictly it "exists" as a file
                    
                    broken_links.append({
                        'source': os.path.relpath(file_path, root_dir),
                        'link_text': text,
                        'target': target,
                        'resolved_path': abs_target_path
                    })
                    
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            
    return broken_links

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    root_directory = os.getcwd()
    print(f"Validating links in {root_directory}")
    
    broken = validate_links(root_directory)
    
    if broken:
        print(f"\nFound {len(broken)} broken links:")
        for link in broken:
            print(f"File: {link['source']}")
            print(f"  Link: [{link['link_text']}]({link['target']})")
            print(f"  Resolved: {link['resolved_path']}")
            print("-" * 40)
    else:
        print("\nNo broken internal links found!")
