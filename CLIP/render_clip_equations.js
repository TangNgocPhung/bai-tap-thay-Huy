"use strict";

const path = require("path");
const sharp = require("sharp");
const { mathjax } = require("mathjax-full/js/mathjax.js");
const { TeX } = require("mathjax-full/js/input/tex.js");
const { SVG } = require("mathjax-full/js/output/svg.js");
const { liteAdaptor } = require("mathjax-full/js/adaptors/liteAdaptor.js");
const { RegisterHTMLHandler } = require("mathjax-full/js/handlers/html.js");
const { AllPackages } = require("mathjax-full/js/input/tex/AllPackages.js");

const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const tex = new TeX({ packages: AllPackages });
const svgOutput = new SVG({ fontCache: "local" });
const document = mathjax.document("", { InputJax: tex, OutputJax: svgOutput });

async function render(latex, filename) {
  const node = document.convert(latex, { display: true });
  let html = adaptor.outerHTML(node);
  const start = html.indexOf("<svg");
  const end = html.indexOf("</svg>");
  let svg = html.slice(start, end + 6)
    .replace(/<\?xml[^>]*>/g, "")
    .replace(/currentColor/g, "#000000");
  if (!svg.includes('xmlns="http://www.w3.org/2000/svg"')) {
    svg = svg.replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ');
  }
  await sharp(Buffer.from(svg), { density: 300 })
    .png()
    .toFile(path.join(__dirname, "doc_assets", filename));
}

Promise.all([
  render(String.raw`\widehat{\mathbf I}_i=\frac{\mathbf I_i}{\lVert\mathbf I_i\rVert_2},\qquad \widehat{\mathbf T}_j=\frac{\mathbf T_j}{\lVert\mathbf T_j\rVert_2}`, "equation-normalize.png"),
  render(String.raw`S_{ij}=\exp(s)\,\widehat{\mathbf I}_i^{\mathsf T}\widehat{\mathbf T}_j`, "equation-similarity.png"),
  render(String.raw`\mathcal L=\frac{1}{2}\left[\operatorname{CE}(S,\mathbf y)+\operatorname{CE}(S^{\mathsf T},\mathbf y)\right],\qquad \mathbf y=(0,1,\ldots,N-1)`, "equation-loss.png")
]).catch((error) => {
  console.error(error);
  process.exit(1);
});
