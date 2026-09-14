// Headless verification for pca-forge. Mirrors the in-browser runTests() logic.
// Extracts the <script id="engine"> block from index.html and runs it under Node vm.
const fs = require('fs');
const vm = require('vm');

const html = fs.readFileSync('index.html', 'utf8');
const m = html.match(/<script id="engine">([\s\S]*?)<\/script>/);
if (!m) { console.error('engine script not found'); process.exit(2); }
const ctx = vm.createContext({ Math, console, Array, Float64Array });
vm.runInContext(m[1] + '\nthis.PCA = PCA;', ctx);
const E = ctx.PCA;

let pass = 0, fail = 0;
function T(name, ok, detail){
  if (ok) { pass++; console.log('  ✅ ' + name + (detail ? '  [' + detail + ']' : '')); }
  else { fail++; console.log('  ❌ ' + name + (detail ? '  [' + detail + ']' : '')); }
}
function maxDiff(a,b){ let mx=0; for(let i=0;i<a.length;i++) for(let j=0;j<a[0].length;j++) mx=Math.max(mx,Math.abs(a[i][j]-b[i][j])); return mx; }

console.log('pca-forge headless self-check');
// 1) covariance symmetric + semi-definite
{
  const X=E.makeGauss(120, 7);
  const S=E.covariance(E.center(X).C);
  let symErr=0, minEig=Infinity;
  for(let i=0;i<S.length;i++) for(let j=0;j<S.length;j++) symErr=Math.max(symErr,Math.abs(S[i][j]-S[j][i]));
  const jac=E.jacobiEig(S,200);
  for(const v of jac.values) minEig=Math.min(minEig,v);
  T('covariance symmetric + PSD (min eig >= -1e-9)', symErr<1e-12 && minEig>=-1e-9, 'symErr='+symErr.toExponential(1)+' minEig='+minEig.toFixed(4));
}
// 2) eigenvectors orthonormal
{
  const X=E.makeGauss(100, 5);
  const S=E.covariance(E.center(X).C);
  const jac=E.jacobiEig(S,200);
  let off=0;
  for(let i=0;i<jac.vectors.length;i++) for(let j=0;j<jac.vectors.length;j++){
    let s=0; for(let k=0;k<jac.vectors.length;k++) s+=jac.vectors[k][i]*jac.vectors[k][j];
    off=Math.max(off,Math.abs(s-(i===j?1:0)));
  }
  T('eigenvectors orthonormal (V^T V = I)', off<1e-12, 'max offdiag='+off.toExponential(2));
}
// 3) power iteration vs Jacobi cross-validation
{
  const X=E.makeGauss(200, 4);
  const S=E.covariance(E.center(X).C);
  const jac=E.jacobiEig(S,200);
  const sorted=E.sortDesc(jac.values, jac.vectors);
  const k=3;
  const def=E.deflateEig(S, k, 3000);
  let worst=0;
  for(let j=0;j<k;j++){
    const c=Math.abs(E.dot(def.vectors[j], sorted.vectors[j]))/(E.norm(def.vectors[j])*E.norm(sorted.vectors[j]));
    worst=Math.max(worst, 1-c);
  }
  T('power-iteration PCs == Jacobi eigvecs (|cos|>=0.999)', worst<1e-3, 'max(1-|cos|)='+worst.toExponential(2));
}
// 4) projected covariance diagonalized
{
  const X=E.makeMoon(160, 3);
  const p=E.pca(X, 2);
  const Xp=E.project(X, p);
  const Sp=E.covariance(E.center(Xp).C);
  let off=0; for(let i=0;i<Sp.length;i++) for(let j=0;j<Sp.length;j++) if(i!==j) off=Math.max(off,Math.abs(Sp[i][j]));
  T('projected covariance diagonalized', off<1e-9, 'max offdiag='+off.toExponential(2));
}
// 5) explained ratios sum to 1
{
  const X=E.makeSpiral(180, 9);
  const p=E.pca(X, 2);
  const sum=p.explained.reduce((a,b)=>a+b,0);
  T('explained variance ratios sum to 1', Math.abs(sum-1)<1e-12, 'sum='+(sum*100).toFixed(4)+'%');
}
// 6) reconstruction error upper bound
{
  const X=E.makeRoll(200, 5);
  const p=E.pca(X, 2);
  const st=E.reconStats(X, p);
  const bound=(1-st.ratio)*st.totalVar + 1e-9;
  T('recon error bound ||X-Xhat||^2 <= (1-ratio)*totalVar', st.err<=bound, 'err='+st.err.toFixed(2)+' bound='+bound.toFixed(2));
}
// 7) rotation equivariance
{
  const X=E.makeGauss(150, 4);
  const Q=E.randomOrthogonal(4, 13);
  const QX=X.map(r=>E.matVec(Q, r));
  const p1=E.pca(X, 2), p2=E.pca(QX, 2);
  const Xhat1=E.reconstruct(E.project(X,p1), p1);
  const Xhat2=E.reconstruct(E.project(QX,p2), p2);
  const QXhat1=Xhat1.map(r=>E.matVec(Q, r));
  let d=0; for(let i=0;i<QXhat1.length;i++) for(let j=0;j<4;j++) d=Math.max(d,Math.abs(QXhat1[i][j]-Xhat2[i][j]));
  T('rotation equivariance: recon(QX) == Q*recon(X)', d<1e-9, 'max diff='+d.toExponential(2));
}
// 8) determinism
{
  const p1=E.pca(E.makeGauss(120, 21), 3);
  const p2=E.pca(E.makeGauss(120, 21), 3);
  let d=0; for(let i=0;i<3;i++) d=Math.max(d,Math.abs(p1.values[i]-p2.values[i]));
  T('determinism: same seed -> same eigenvalues', d<1e-12, 'max diff='+d.toExponential(2));
}

console.log('\n' + (fail===0 ? 'ALL PASS' : 'FAILURES: '+fail) + '  (' + pass + '/' + (pass+fail) + ')');
process.exit(fail===0?0:1);
