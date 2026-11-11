// Small, dependency-free Markdown renderer (headings, lists, tables, code, emphasis, citations).
// Input is escaped first, so model output can never inject HTML.
(function () {
  const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  function inline(s) {
    s = esc(s);
    s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
    // simple inline LaTeX the model copies from papers ("$3 \times 4$", "$T_{x}$") as plain symbols
    s = s.replace(/\$([^$\n]*(?:\\[a-zA-Z]+|[_^]\{)[^$\n]*)\$/g, (m, x) => x
      .replace(/\\times/g, "×").replace(/\\cdot/g, "·").replace(/\\cup/g, "∪").replace(/\\cap/g, "∩")
      .replace(/\\leq?/g, "≤").replace(/\\geq?/g, "≥").replace(/\\approx/g, "≈").replace(/\\in\b/g, "∈")
      .replace(/\\(hat|mathcal|mathbf|text|mathrm)\{([^}]*)\}/g, "$2").replace(/_\{([^}]*)\}/g, "_$1")
      .replace(/\\([a-zA-Z]+)/g, "$1").replace(/[{}]/g, ""));
    s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    s = s.replace(/(^|[^*])\*([^*\s][^*]*)\*/g, "$1<em>$2</em>");
    // citations like [S1], [S1, S3], [N2]
    s = s.replace(/\[((?:[SN]\d+)(?:\s*[,;]\s*[SN]?\d+)*)\]/g, (m, inner) =>
      inner.split(/\s*[,;]\s*/).map((id) => {
        const prefix = inner.trim()[0];
        const full = /^[SN]/.test(id) ? id : prefix + id;
        return `<a class="cite" data-src="${full}">${full}</a>`;
      }).join(""));
    return s;
  }

  function isTableSep(line) { return /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/.test(line); }
  function cells(line) {
    return line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());
  }

  window.renderMarkdown = function (src) {
    const lines = src.replace(/\r/g, "").split("\n");
    let html = "", i = 0;
    while (i < lines.length) {
      const line = lines[i];
      if (/^```/.test(line)) {
        const buf = [];
        i++;
        while (i < lines.length && !/^```/.test(lines[i])) buf.push(lines[i++]);
        i++;
        html += `<pre><code>${esc(buf.join("\n"))}</code></pre>`;
        continue;
      }
      const h = line.match(/^(#{1,6})\s+(.*)$/);
      if (h) { const lvl = Math.min(h[1].length + 2, 6); html += `<h${lvl}>${inline(h[2])}</h${lvl}>`; i++; continue; }
      if (line.includes("|") && i + 1 < lines.length && isTableSep(lines[i + 1])) {
        const head = cells(line);
        i += 2;
        let body = "";
        while (i < lines.length && lines[i].includes("|") && lines[i].trim()) {
          body += "<tr>" + cells(lines[i]).map((c) => `<td>${inline(c)}</td>`).join("") + "</tr>";
          i++;
        }
        html += `<div class="table-wrap"><table><thead><tr>${head.map((c) => `<th>${inline(c)}</th>`).join("")}</tr></thead><tbody>${body}</tbody></table></div>`;
        continue;
      }
      if (/^\s*([-*+]|\d+[.)])\s+/.test(line)) {
        const ordered = /^\s*\d+[.)]\s+/.test(line);
        const items = [];
        while (i < lines.length && /^\s*([-*+]|\d+[.)])\s+/.test(lines[i])) {
          let item = lines[i].replace(/^\s*([-*+]|\d+[.)])\s+/, "");
          const num = (lines[i].match(/^\s*(\d+)[.)]\s/) || [])[1];
          const indent = lines[i].match(/^\s*/)[0].length;
          i++;
          // continuation / nested lines
          while (i < lines.length && lines[i].trim() && !/^\s*([-*+]|\d+[.)])\s+/.test(lines[i]) && /^\s+/.test(lines[i])) {
            item += " " + lines[i].trim(); i++;
          }
          items.push({ text: item, nested: indent >= 2, num });
        }
        const tag = ordered ? "ol" : "ul";
        // keep the model's own numbering ("4. Key results" stays 4 even after a paragraph break),
        // and show bullets nested inside a numbered list as bullets
        html += `<${tag}>` + items.map((it) => {
          const attrs = (it.nested ? ' class="nested"' : "") + (ordered && it.num ? ` value="${it.num}"` : "") +
            (ordered && !it.num ? ' style="list-style-type: disc"' : "");
          return `<li${attrs}>${inline(it.text)}</li>`;
        }).join("") + `</${tag}>`;
        continue;
      }
      if (/^\s*(---|\*\*\*)\s*$/.test(line)) { html += "<hr>"; i++; continue; }
      if (!line.trim()) { i++; continue; }
      const para = [line];
      i++;
      while (i < lines.length && lines[i].trim() && !/^(#{1,6}\s|```|\s*([-*+]|\d+[.)])\s+)/.test(lines[i]) &&
             !(lines[i].includes("|") && i + 1 < lines.length && isTableSep(lines[i + 1]))) {
        para.push(lines[i++]);
      }
      html += `<p>${para.map(inline).join("<br>")}</p>`;
    }
    return html;
  };
})();
