"""collision_<sess>.glb の単体HTMLビューア生成 (WebGL, 依存なし1ファイル)

点群viewer.htmlのメッシュ版。層ごと(構造/スラブ/障害物)に色分けし、
回転/ズーム/平行移動、高さクリップ、障害物・構造の表示切替を持つ。

実行: IMG_SESSION=0312 conda run -n panovggt python -m img_align.mesh.collision_viewer
出力: results/backend/collision_viewer_<sess>.html
"""
import base64
import json

import numpy as np
import trimesh

from img_align.mesh.collision_mesh import OUT_DIR
from img_align.config import SESSION

GROUP_COLOR = {
    "slab": (110, 115, 135),
    "structure": (160, 168, 185),
    "obstacle": (240, 140, 40),
}


def group_of(name):
    if name.startswith("obstacle"):
        return "obstacle"
    if "slab" in name:
        return "slab"
    return "structure"


PLY_NAME = {"0828": "aligned_house.ply", "0312": "aligned_house_0312.ply"}
CURTAIN_COLORS = [(60, 90, 255), (90, 200, 255)]
MAX_OVERLAY_PTS = 450_000


def main():
    from img_align.core.io_ply import read_ply_xyzrgb
    scene = trimesh.load(OUT_DIR / f"collision_{SESSION}.glb")
    groups = {}
    # 注意: glTFはY-up規約のためtrimeshがシーングラフに座標変換を入れる。
    # 生バッファ(scene.geometry)ではなくノード変換を適用して読むこと
    for node in scene.graph.nodes_geometry:
        T, geom_name = scene.graph[node]
        g = scene.geometry[geom_name].copy()
        g.apply_transform(T)
        key = group_of(geom_name)
        tri = g.vertices[g.faces].reshape(-1, 3)   # 三角形スープ (flat shading用)
        groups.setdefault(key, []).append(tri)
    payload = {}
    lo = np.array([np.inf] * 3)
    hi = -lo
    for key, tris in groups.items():
        v = np.concatenate(tris).astype(np.float64)
        lo = np.minimum(lo, v.min(axis=0))
        hi = np.maximum(hi, v.max(axis=0))
        payload[key] = v

    # 点群オーバーレイ (位置合わせ済み点群、合成カーテン点は除外)
    xyz, rgb = read_ply_xyzrgb(OUT_DIR / PLY_NAME[SESSION])
    keep = np.ones(len(xyz), bool)
    for c in CURTAIN_COLORS:
        keep &= ~np.all(rgb == np.array(c, np.uint8), axis=1)
    xyz, rgb = xyz[keep], rgb[keep]
    if len(xyz) > MAX_OVERLAY_PTS:
        step = len(xyz) // MAX_OVERLAY_PTS + 1
        xyz, rgb = xyz[::step], rgb[::step]
    lo = np.minimum(lo, xyz.min(axis=0))
    hi = np.maximum(hi, xyz.max(axis=0))

    meta = {"lo": lo.tolist(), "hi": hi.tolist(), "groups": {}}
    blob = b""
    for key, v in payload.items():
        q = np.round((v - lo) / (hi - lo) * 65535).astype(np.uint16)
        meta["groups"][key] = {"n": len(v), "offset": len(blob),
                               "color": GROUP_COLOR[key]}
        blob += q.tobytes()
    qp = np.round((xyz - lo) / (hi - lo) * 65535).astype(np.uint16)
    meta["points"] = {"n": len(xyz), "offset": len(blob)}
    blob += qp.tobytes()
    meta["points"]["rgb_offset"] = len(blob)
    blob += rgb.astype(np.uint8).tobytes()
    b64 = base64.b64encode(blob).decode()

    html = TEMPLATE.replace("/*DATA*/",
                            f"const META={json.dumps(meta)};\n"
                            f"const B64=\"{b64}\";")
    html = html.replace("__TITLE__", f"collision model {SESSION}")
    out = OUT_DIR / f"collision_viewer_{SESSION}.html"
    out.write_text(html)
    print(f"-> {out} ({sum(len(v) for v in payload.values()) // 3:,} tris "
          f"+ {len(xyz):,} pts)")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  html,body{margin:0;height:100%;background:#101014;color:#ddd;font-family:sans-serif;overflow:hidden}
  #c{width:100%;height:100%;display:block;cursor:grab}
  #ui{position:fixed;top:10px;left:10px;background:rgba(20,20,28,.85);padding:10px 14px;border-radius:8px;font-size:12px;line-height:1.9}
  #hint{position:fixed;bottom:8px;left:10px;font-size:11px;color:#888}
</style>
</head>
<body>
<canvas id="c"></canvas>
<div id="ui">
  <b>__TITLE__</b><br>
  <label><input id="show_st" type="checkbox" checked> 構造 (図面由来)</label><br>
  <label><input id="show_sl" type="checkbox" checked> スラブ</label><br>
  <label><input id="show_ob" type="checkbox" checked> 障害物 (点群由来)</label><br>
  <label><input id="show_pc" type="checkbox"> 点群オーバーレイ</label><br>
  <label>点サイズ <input id="ps" type="range" min="1" max="5" step="0.5" value="1.5"></label><br>
  <label>高さ上限 <input id="h1" type="range" min="0" max="6.2" step="0.1" value="6.2"> <span id="h1v"></span>m</label>
</div>
<div id="hint">左ドラッグ=回転 / ホイール=ズーム / 右ドラッグ=平行移動</div>
<script>
/*DATA*/
</script>
<script>
"use strict";
const raw = Uint8Array.from(atob(B64), c => c.charCodeAt(0));
const lo = META.lo, hi = META.hi;
function decode(g){
  const q = new Uint16Array(raw.buffer, g.offset, g.n*3);
  const pos = new Float32Array(g.n*3);
  for(let i=0;i<g.n;i++){
    pos[i*3]   = lo[0]+q[i*3]/65535*(hi[0]-lo[0]);
    pos[i*3+1] = -(lo[1]+q[i*3+1]/65535*(hi[1]-lo[1]));  // 図面yは下向き→反転
    pos[i*3+2] = lo[2]+q[i*3+2]/65535*(hi[2]-lo[2]);
  }
  const nrm = new Float32Array(g.n*3);
  for(let i=0;i<g.n;i+=3){
    const ax=pos[i*3],ay=pos[i*3+1],az=pos[i*3+2];
    const bx=pos[i*3+3]-ax,by=pos[i*3+4]-ay,bz=pos[i*3+5]-az;
    const cx=pos[i*3+6]-ax,cy=pos[i*3+7]-ay,cz=pos[i*3+8]-az;
    let nx=by*cz-bz*cy, ny=bz*cx-bx*cz, nz=bx*cy-by*cx;
    const l=Math.hypot(nx,ny,nz)||1; nx/=l;ny/=l;nz/=l;
    for(let k=0;k<3;k++){nrm[(i+k)*3]=nx;nrm[(i+k)*3+1]=ny;nrm[(i+k)*3+2]=nz;}
  }
  return {pos, nrm, n:g.n, color:g.color};
}
const parts = {};
for(const k in META.groups) parts[k] = decode(META.groups[k]);
// 点群オーバーレイ
let pcloud = null;
if(META.points){
  const g = META.points;
  const q = new Uint16Array(raw.buffer, g.offset, g.n*3);
  const pos = new Float32Array(g.n*3);
  for(let i=0;i<g.n;i++){
    pos[i*3]   = lo[0]+q[i*3]/65535*(hi[0]-lo[0]);
    pos[i*3+1] = -(lo[1]+q[i*3+1]/65535*(hi[1]-lo[1]));
    pos[i*3+2] = lo[2]+q[i*3+2]/65535*(hi[2]-lo[2]);
  }
  pcloud = {pos, col: new Uint8Array(raw.buffer, g.rgb_offset, g.n*3), n: g.n};
}
const cx=(lo[0]+hi[0])/2, cy=-(lo[1]+hi[1])/2, cz=(lo[2]+hi[2])/2;

const canvas=document.getElementById("c");
const gl=canvas.getContext("webgl",{antialias:true});
const vs=`attribute vec3 aP;attribute vec3 aN;uniform mat4 uM;
varying vec3 vN;varying float vZ;
void main(){gl_Position=uM*vec4(aP,1.0);vN=aN;vZ=aP.z;}`;
const fs=`precision mediump float;varying vec3 vN;varying float vZ;
uniform vec3 uC;uniform float uA,uH1;
void main(){if(vZ>uH1)discard;
float d=0.45+0.55*abs(dot(normalize(vN),normalize(vec3(0.4,0.6,0.8))));
gl_FragColor=vec4(uC*d,uA);}`;
function sh(t,s){const o=gl.createShader(t);gl.shaderSource(o,s);gl.compileShader(o);
if(!gl.getShaderParameter(o,gl.COMPILE_STATUS))throw gl.getShaderInfoLog(o);return o;}
const prog=gl.createProgram();
gl.attachShader(prog,sh(gl.VERTEX_SHADER,vs));gl.attachShader(prog,sh(gl.FRAGMENT_SHADER,fs));
gl.linkProgram(prog);gl.useProgram(prog);
const loc={aP:gl.getAttribLocation(prog,"aP"),aN:gl.getAttribLocation(prog,"aN"),
uM:gl.getUniformLocation(prog,"uM"),uC:gl.getUniformLocation(prog,"uC"),
uA:gl.getUniformLocation(prog,"uA"),uH1:gl.getUniformLocation(prog,"uH1")};
for(const k in parts){
  const p=parts[k];
  p.bP=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,p.bP);
  gl.bufferData(gl.ARRAY_BUFFER,p.pos,gl.STATIC_DRAW);
  p.bN=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,p.bN);
  gl.bufferData(gl.ARRAY_BUFFER,p.nrm,gl.STATIC_DRAW);
}
// 点群用プログラム
const vsP=`attribute vec3 aP;attribute vec3 aC;uniform mat4 uM;uniform float uS;
varying vec3 vC;varying float vZ;
void main(){gl_Position=uM*vec4(aP,1.0);gl_PointSize=uS;vC=aC/255.0;vZ=aP.z;}`;
const fsP=`precision mediump float;varying vec3 vC;varying float vZ;uniform float uH1;
void main(){if(vZ>uH1)discard;gl_FragColor=vec4(vC,1.0);}`;
const progP=gl.createProgram();
gl.attachShader(progP,sh(gl.VERTEX_SHADER,vsP));gl.attachShader(progP,sh(gl.FRAGMENT_SHADER,fsP));
gl.linkProgram(progP);
let locP=null;
if(pcloud){
  pcloud.bP=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,pcloud.bP);
  gl.bufferData(gl.ARRAY_BUFFER,pcloud.pos,gl.STATIC_DRAW);
  pcloud.bC=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,pcloud.bC);
  gl.bufferData(gl.ARRAY_BUFFER,pcloud.col,gl.STATIC_DRAW);
  locP={aP:gl.getAttribLocation(progP,"aP"),aC:gl.getAttribLocation(progP,"aC"),
        uM:gl.getUniformLocation(progP,"uM"),uS:gl.getUniformLocation(progP,"uS"),
        uH1:gl.getUniformLocation(progP,"uH1")};
}
let theta=-1.0, phi=0.7, dist=16, panX=0, panY=0, panZ=0;
function mat(){
  const w=canvas.width,h=canvas.height,f=1/Math.tan(0.4),asp=w/h,zn=0.1,zf=200;
  const ex=cx+panX+dist*Math.cos(phi)*Math.cos(theta),
        ey=cy+panY+dist*Math.cos(phi)*Math.sin(theta),
        ez=cz+panZ+dist*Math.sin(phi);
  const tx=cx+panX,ty=cy+panY,tz=cz+panZ;
  let zx=ex-tx,zy=ey-ty,zz=ez-tz;const zl=Math.hypot(zx,zy,zz);zx/=zl;zy/=zl;zz/=zl;
  let xx=-zy, xy=zx, xz=0; const xl=Math.hypot(xx,xy,xz)||1;xx/=xl;xy/=xl;
  const yx=zy*xz-zz*xy, yy=zz*xx-zx*xz, yz=zx*xy-zy*xx;
  const tex=-(xx*ex+xy*ey), tey=-(yx*ex+yy*ey+yz*ez), tez=-(zx*ex+zy*ey+zz*ez);
  const V=[xx,yx,zx,0, xy,yy,zy,0, xz,yz,zz,0, tex,tey,tez,1];
  const P=[f/asp,0,0,0, 0,f,0,0, 0,0,(zf+zn)/(zn-zf),-1, 0,0,2*zf*zn/(zn-zf),0];
  const M=new Float32Array(16);
  for(let r=0;r<4;r++)for(let c2=0;c2<4;c2++){let s=0;for(let k=0;k<4;k++)s+=V[r*4+k]*P[k*4+c2];M[r*4+c2]=s;}
  return M;
}
function drawPart(p,alpha){
  gl.bindBuffer(gl.ARRAY_BUFFER,p.bP);
  gl.enableVertexAttribArray(loc.aP);gl.vertexAttribPointer(loc.aP,3,gl.FLOAT,false,0,0);
  gl.bindBuffer(gl.ARRAY_BUFFER,p.bN);
  gl.enableVertexAttribArray(loc.aN);gl.vertexAttribPointer(loc.aN,3,gl.FLOAT,false,0,0);
  gl.uniform3f(loc.uC,p.color[0]/255,p.color[1]/255,p.color[2]/255);
  gl.uniform1f(loc.uA,alpha);
  gl.drawArrays(gl.TRIANGLES,0,p.n);
}
function draw(){
  canvas.width=innerWidth;canvas.height=innerHeight;
  gl.viewport(0,0,canvas.width,canvas.height);
  gl.clearColor(0.06,0.06,0.08,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST);gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);
  const M=mat();
  h1v.textContent=h1.value;
  if(show_pc.checked&&pcloud){
    gl.useProgram(progP);
    gl.bindBuffer(gl.ARRAY_BUFFER,pcloud.bP);
    gl.enableVertexAttribArray(locP.aP);gl.vertexAttribPointer(locP.aP,3,gl.FLOAT,false,0,0);
    gl.bindBuffer(gl.ARRAY_BUFFER,pcloud.bC);
    gl.enableVertexAttribArray(locP.aC);gl.vertexAttribPointer(locP.aC,3,gl.UNSIGNED_BYTE,true,0,0);
    gl.uniformMatrix4fv(locP.uM,false,M);
    gl.uniform1f(locP.uS,parseFloat(ps.value));
    gl.uniform1f(locP.uH1,parseFloat(h1.value));
    gl.drawArrays(gl.POINTS,0,pcloud.n);
  }
  gl.useProgram(prog);
  gl.uniformMatrix4fv(loc.uM,false,M);
  gl.uniform1f(loc.uH1,parseFloat(h1.value));
  if(show_ob.checked&&parts.obstacle)drawPart(parts.obstacle,1.0);
  if(show_st.checked&&parts.structure)drawPart(parts.structure,0.45);
  if(show_sl.checked&&parts.slab)drawPart(parts.slab,0.3);
  requestAnimationFrame(draw);
}
let drag=null;
canvas.addEventListener("mousedown",e=>{drag={x:e.clientX,y:e.clientY,b:e.button,sh:e.shiftKey};});
addEventListener("mouseup",()=>drag=null);
addEventListener("mousemove",e=>{
  if(!drag)return;
  const dx=e.clientX-drag.x, dy=e.clientY-drag.y;drag.x=e.clientX;drag.y=e.clientY;
  if(drag.b===2||drag.sh){
    const s=dist*0.0015;
    panX+=(-dx*Math.sin(theta)-dy*Math.cos(theta)*Math.sin(phi))*s;
    panY+=( dx*Math.cos(theta)-dy*Math.sin(theta)*Math.sin(phi))*s;
    panZ+=dy*Math.cos(phi)*s;
  }else{theta-=dx*0.005;phi=Math.min(1.55,Math.max(-0.2,phi+dy*0.005));}
});
canvas.addEventListener("wheel",e=>{dist*=e.deltaY>0?1.1:0.9;dist=Math.max(1,Math.min(80,dist));e.preventDefault()},{passive:false});
canvas.addEventListener("contextmenu",e=>e.preventDefault());
draw();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
