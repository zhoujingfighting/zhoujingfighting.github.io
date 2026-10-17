hexo.extend.injector.register('body_end', `
<!-- Mermaid 11 Support -->
<script type="module">
  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
  mermaid.initialize({
    startOnLoad: false,
    securityLevel: 'loose',
    theme: 'default'
  });

  document.addEventListener('DOMContentLoaded', async function() {
    const mermaidElements = document.querySelectorAll('.language-mermaid, pre code.language-mermaid, pre code.highlight.mermaid, pre code.mermaid, .mermaid');
    if (mermaidElements.length > 0) {
      mermaidElements.forEach(el => {
        const code = el.textContent;
        const pre = document.createElement('pre');
        pre.className = 'mermaid';
        pre.style.textAlign = 'center';
        pre.style.background = 'transparent';
        pre.style.border = 'none';
        pre.textContent = code.trim();
        const target = el.closest('figure') || el.closest('pre') || el;
        target.parentNode.replaceChild(pre, target);
      });
      try {
        await mermaid.run({ querySelector: '.mermaid' });
      } catch (err) {
        console.error('Mermaid render error:', err);
      }
    }
  });
</script>
`);
