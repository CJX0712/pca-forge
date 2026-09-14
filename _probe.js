// Probe: extra property checks + numerical character of PCA on pca-forge.
const fs = require('fs');
const vm = require('vm');
const html = fs.readFileSync('index.html', 'utf8');
const m = html.match(/<script id="engine">([\s\S]*?)<\/script>/);
const ctx = vm.createContext({ Math, console, Array, Float64Array });
vm.runInContext(m[1] + '\nthis.PCA = PCA;', ctx);
const E = ctx.PCA;

console.log('pca-forge probe');

// A) anisotropic gaussian: PC1 should align with axis 0 (the stretched one)
{
  const X=E.makeGauss(400, 42);
  const p=E.pca(X, 3);
  // PC1 = loadings[0]; |cos| with e0
  const e0=[1,0,0];
  const c=Math.abs(E.dot(p.loadings[0], e0))/E.norm(p.loadings[0]);
  console.log('gauss PC1 |cos| with e0 (stretched axis, should ~1):', c.toFixed(4), 'explained[0]=', (p.explained[0]*100).toFixed(1)+'%');
  console.log('  eigenvalues:', p.values.map(v=>v.toFixed(3)).join(', '), ' (expect λ0 >> λ1≈λ2)');
}

// B) reconstruction error bound tightness on roll (3D -> 2D keeps most variance)
{
  const X=E.makeRoll(300, 5);
  const p2=E.pca(X, 2);
  const st2=E.reconStats(X, p2);
  const p3=E.pca(X, 3);
  const st3=E.reconStats(X, p3);
  console.log('swiss-roll: k=2 ratio='+(st2.ratio*100).toFixed(1)+'% err='+st2.err.toFixed(1)+' | k=3 ratio='+(st3.ratio*100).toFixed(1)+'% err='+st3.err.toFixed(3));
}

// C) spiral is non-linear: PCA captures little variance in 1 component, but bound still holds
{
  const X=E.makeSpiral(300, 9);
  const p=E.pca(X, 2);
  const st=E.reconStats(X, p);
  console.log('spiral k=2: explained[0]='+(p.explained[0]*100).toFixed(1)+'% explained[1]='+(p.explained[1]*100).toFixed(1)+'% err='+st.err.toFixed(2)+' bound='+((1-st.ratio)*st.totalVar).toFixed(2));
}

// D) jacobi accuracy: A V == V Λ (residual)
{
  const X=E.makeGauss(100, 6);
  const S=E.covariance(E.center(X).C);
  const jac=E.jacobiEig(S, 200);
  // build A V
  const AV=jac.vectors.map(col=>E.matVec(S, col));
  let res=0;
  for(let i=0;i<jac.vectors.length;i++) for(let j=0;j<jac.vectors.length;j++){
    res=Math.max(res, Math.abs(AV[i][j]-jac.vectors[i][j]*jac.values[i]));
  }
  console.log('jacobi residual ||A v - λ v|| max:', res.toExponential(2), '(should be ~0)');
}

// E) deflation cross-check: compare deflateEig values vs jacobi sorted values (gauss is 3D)
{
  const X=E.makeGauss(150, 5);
  const S=E.covariance(E.center(X).C);
  const jac=E.jacobiEig(S,200);
  const sorted=E.sortDesc(jac.values, jac.vectors);
  const def=E.deflateEig(S, 3, 4000);
  let d=0;
  for(let j=0;j<3;j++) d=Math.max(d, Math.abs(def.values[j]-sorted.values[j]));
  console.log('deflateEig eigenvalues vs Jacobi:', def.values.map(v=>v.toFixed(3)).join(', '));
  console.log('  max |diff|:', d.toExponential(2));
}
