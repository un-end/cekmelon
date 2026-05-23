import { useState, useEffect, useRef } from "react";

// ─── Dataset awal ──────────────────────────────────────────────────────────
const SEED_DATASET = [
  { id:1,  berat:2.1, gula:14.5, netting:5, kerusakan:1.2,  kelas:"Layak Ekspor" },
  { id:2,  berat:1.8, gula:13.2, netting:4, kerusakan:3.5,  kelas:"Layak Ekspor" },
  { id:3,  berat:3.2, gula:15.0, netting:5, kerusakan:0.8,  kelas:"Layak Ekspor" },
  { id:4,  berat:1.2, gula:11.5, netting:3, kerusakan:8.0,  kelas:"Tidak Layak Ekspor" },
  { id:5,  berat:2.5, gula:13.8, netting:4, kerusakan:2.1,  kelas:"Layak Ekspor" },
  { id:6,  berat:0.9, gula:10.2, netting:2, kerusakan:12.5, kelas:"Tidak Layak Ekspor" },
  { id:7,  berat:2.8, gula:14.2, netting:5, kerusakan:1.5,  kelas:"Layak Ekspor" },
  { id:8,  berat:3.5, gula:15.8, netting:5, kerusakan:0.5,  kelas:"Layak Ekspor" },
  { id:9,  berat:1.4, gula:12.0, netting:3, kerusakan:6.5,  kelas:"Tidak Layak Ekspor" },
  { id:10, berat:2.2, gula:13.5, netting:4, kerusakan:4.8,  kelas:"Layak Ekspor" },
];

const NETTING_OPTIONS = [
  { value:1, label:"1 – Sangat Renggang / Polos" },
  { value:2, label:"2 – Renggang" },
  { value:3, label:"3 – Sedang" },
  { value:4, label:"4 – Rapat" },
  { value:5, label:"5 – Sangat Rapat" },
];

// ─── Naive Bayes ───────────────────────────────────────────────────────────
function naiveBayes(trainData, inp) {
  const classes = ["Layak Ekspor","Tidak Layak Ekspor"];
  const feats   = ["berat","gula","netting","kerusakan"];
  const alpha = 1, total = trainData.length, scores = {};
  classes.forEach(cls => {
    const cd = trainData.filter(d => d.kelas === cls);
    if (!cd.length) { scores[cls] = -Infinity; return; }
    const prior = (cd.length + alpha) / (total + alpha * classes.length);
    let ll = Math.log(prior);
    feats.forEach(f => {
      const vals = cd.map(d => d[f]);
      const mean = vals.reduce((a,b) => a+b, 0) / vals.length;
      const variance = vals.reduce((a,b) => a + Math.pow(b-mean,2), 0) / vals.length + alpha;
      ll += Math.log((1/Math.sqrt(2*Math.PI*variance)) * Math.exp(-Math.pow(inp[f]-mean,2)/(2*variance)) + 1e-9);
    });
    scores[cls] = ll;
  });
  const ls=scores["Layak Ekspor"], ts=scores["Tidak Layak Ekspor"];
  const mx=Math.max(ls,ts), el=Math.exp(ls-mx), et=Math.exp(ts-mx), sm=el+et;
  return { predicted: ls>=ts?"Layak Ekspor":"Tidak Layak Ekspor", probLayak:((el/sm)*100).toFixed(1), probTidak:((et/sm)*100).toFixed(1) };
}

// ─── Colors ────────────────────────────────────────────────────────────────
const C = {
  g1:"#0a1c12", g2:"#1a5c2a", g3:"#2d8a45", g4:"#4ab860", g5:"#a8e6b5",
  gold:"#c8930a", goldL:"#f6c44a", white:"#ffffff", offW:"#f4fbf6",
  border:"#b5dfc2", text:"#1a2e1f", muted:"#4a6b52"
};

// ─── Global CSS ────────────────────────────────────────────────────────────
const GCSS = `
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:Georgia,serif}
  @keyframes fadeUp{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}
  @keyframes popIn{from{opacity:0;transform:scale(0.91)}to{opacity:1;transform:scale(1)}}
  @keyframes spin{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}
  @keyframes spinR{from{transform:rotate(0deg)}to{transform:rotate(-360deg)}}
  @keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-10px)}}
  @keyframes textGlow{0%,100%{text-shadow:0 0 20px rgba(74,184,96,0.4)}50%{text-shadow:0 0 50px rgba(74,184,96,0.9)}}
  @keyframes bgFade{from{opacity:0}to{opacity:1}}
  @keyframes slideL{from{transform:translateX(50px);opacity:0}to{transform:translateX(0);opacity:1}}
  @keyframes slideR{from{transform:translateX(-50px);opacity:0}to{transform:translateX(0);opacity:1}}
  .hov:hover{filter:brightness(1.12)!important;transform:translateY(-1px)}
  .hov{transition:all 0.18s ease}
  .navH:hover{background:rgba(74,184,96,0.18)!important}
  .inpF:focus{border-color:#4ab860!important;box-shadow:0 0 0 3px rgba(74,184,96,0.18)!important;outline:none}
  .trH:hover{background:rgba(74,184,96,0.06)!important}
  .cardHov:hover{transform:translateY(-3px);box-shadow:0 12px 32px rgba(0,0,0,0.22)!important}
  .cardHov{transition:all 0.22s ease}
  .landLink:hover{color:#4ab860!important}
  ::-webkit-scrollbar{width:5px}
  ::-webkit-scrollbar-thumb{background:rgba(74,184,96,0.35);border-radius:3px}
`;

// ─── UI Atoms ──────────────────────────────────────────────────────────────
function Inp({ value, onChange, ...r }) {
  return <input className="inpF" value={value} onChange={onChange}
    style={{width:"100%",padding:"10px 13px",borderRadius:"9px",border:`1.5px solid ${C.border}`,
      background:C.offW,color:C.text,fontSize:"14px",fontFamily:"Georgia,serif",display:"block",transition:"all 0.2s"}} {...r}/>;
}
function Lbl({ c }) {
  return <label style={{fontSize:"11px",fontWeight:"700",color:C.muted,display:"block",marginBottom:"4px",textTransform:"uppercase",letterSpacing:"0.5px"}}>{c}</label>;
}
function BtnG({ children, onClick, s={} }) {
  return <button className="hov" onClick={onClick}
    style={{padding:"11px 22px",borderRadius:"9px",border:"none",background:`linear-gradient(135deg,${C.g3},${C.g4})`,
      color:"#fff",cursor:"pointer",fontSize:"14px",fontFamily:"Georgia,serif",fontWeight:"700",...s}}>{children}</button>;
}
function BtnAu({ children, onClick, s={} }) {
  return <button className="hov" onClick={onClick}
    style={{padding:"11px 22px",borderRadius:"9px",border:"none",background:`linear-gradient(135deg,${C.gold},${C.goldL})`,
      color:"#fff",cursor:"pointer",fontSize:"14px",fontFamily:"Georgia,serif",fontWeight:"700",...s}}>{children}</button>;
}
function BtnO({ children, onClick, color=C.g4, s={} }) {
  return <button className="hov" onClick={onClick}
    style={{padding:"11px 22px",borderRadius:"9px",border:`2px solid ${color}`,background:"transparent",
      color,cursor:"pointer",fontSize:"14px",fontFamily:"Georgia,serif",fontWeight:"700",...s}}>{children}</button>;
}
function Card({ children, s={} }) {
  return <div style={{background:"rgba(255,255,255,0.07)",border:"1px solid rgba(74,184,96,0.28)",borderRadius:"14px",padding:"1.5rem",...s}}>{children}</div>;
}
function CardW({ children, s={} }) {
  return <div style={{background:C.white,border:`1px solid ${C.border}`,borderRadius:"14px",padding:"1.5rem",color:C.text,...s}}>{children}</div>;
}
function Badge({ ok, children }) {
  return <span style={{display:"inline-block",padding:"3px 11px",borderRadius:"999px",
    background:ok?"#d4f7dd":"#fde8e8",color:ok?"#1a5c2a":"#9b1c1c",fontWeight:"700",fontSize:"12px"}}>{children}</span>;
}
function H2({ children, s={} }) {
  return <h2 style={{fontSize:"1.1rem",fontWeight:"700",color:C.g4,margin:"0 0 1rem",...s}}>{children}</h2>;
}

// ─── Modal (DEFINED OUTSIDE APP – fixes re-mount bug) ─────────────────────
function ModalBox({ show, onClose, children }) {
  if (!show) return null;
  return (
    <div onClick={onClose} style={{position:"fixed",inset:0,background:"rgba(0,0,0,0.65)",zIndex:9999,
      display:"flex",alignItems:"center",justifyContent:"center",padding:"1rem",backdropFilter:"blur(4px)"}}>
      <div onClick={e => e.stopPropagation()}
        style={{background:C.white,borderRadius:"20px",padding:"2rem",maxWidth:"460px",width:"100%",
          color:C.text,animation:"popIn 0.25s ease",boxShadow:"0 32px 80px rgba(0,0,0,0.4)"}}>
        {children}
      </div>
    </div>
  );
}

// ─── Avatar SVG ────────────────────────────────────────────────────────────
function AvatarSVG({ seed="x", size=80, role="user" }) {
  const hash = seed.split("").reduce((a,c) => a + c.charCodeAt(0), 0);
  const hue  = (hash * 37) % 360;
  const skin = `hsl(${(hash*13)%40+20},60%,${role==="dosen"?65:72}%)`;
  const hair = `hsl(${(hash*7)%60+10},55%,${22+(hash%20)}%)`;
  const shirt= `hsl(${hue},55%,40%)`;
  const bg   = `hsl(${hue},40%,92%)`;
  const hasGlasses = hash%3===0;
  return (
    <svg width={size} height={size} viewBox="0 0 80 80" style={{borderRadius:"50%",display:"block"}}>
      <circle cx="40" cy="40" r="40" fill={bg}/>
      <ellipse cx="40" cy="72" rx="22" ry="14" fill={shirt}/>
      <rect x="35" y="52" width="10" height="8" rx="3" fill={skin}/>
      <ellipse cx="40" cy="36" rx="16" ry="18" fill={skin}/>
      <ellipse cx="40" cy="22" rx="16" ry="10" fill={hair}/>
      <rect x="24" y="22" width="32" height="8" fill={hair}/>
      <ellipse cx="24" cy="37" rx="3" ry="4" fill={skin}/>
      <ellipse cx="56" cy="37" rx="3" ry="4" fill={skin}/>
      <ellipse cx="34" cy="36" rx="3" ry="3.5" fill="#fff"/>
      <ellipse cx="46" cy="36" rx="3" ry="3.5" fill="#fff"/>
      <circle cx="34.8" cy="36.8" r="2" fill="#2a1a0e"/>
      <circle cx="46.8" cy="36.8" r="2" fill="#2a1a0e"/>
      <circle cx="35.5" cy="35.5" r="0.7" fill="#fff"/>
      <circle cx="47.5" cy="35.5" r="0.7" fill="#fff"/>
      <path d="M 30 31 Q 34 29 38 31" fill="none" stroke={hair} strokeWidth="1.5" strokeLinecap="round"/>
      <path d="M 42 31 Q 46 29 50 31" fill="none" stroke={hair} strokeWidth="1.5" strokeLinecap="round"/>
      <ellipse cx="40" cy="41" rx="2" ry="1.5" fill={`hsl(${(hash*13)%40+10},50%,58%)`}/>
      <path d="M 35 46 Q 40 50 45 46" fill="none" stroke={`hsl(${(hash*13)%40+5},50%,45%)`} strokeWidth="1.5" strokeLinecap="round"/>
      {hasGlasses && <>
        <rect x="28" y="33" width="10" height="7" rx="3" fill="none" stroke="#555" strokeWidth="1.2"/>
        <rect x="42" y="33" width="10" height="7" rx="3" fill="none" stroke="#555" strokeWidth="1.2"/>
        <line x1="38" y1="36" x2="42" y2="36" stroke="#555" strokeWidth="1.2"/>
      </>}
      {role==="dosen" && <>
        <polygon points="40,58 37,64 40,78 43,64" fill="#c0392b"/>
        <polygon points="38,58 40,61 42,58" fill="#922b21"/>
      </>}
    </svg>
  );
}

// ─── Carousel (outside App) ────────────────────────────────────────────────
const SLIDES = [
  { bg:"linear-gradient(145deg,#051a0a 0%,#0e4a1e 50%,#061508 100%)",  accent:"#4ab860", emoji:"🍈", title:"Melon Premium Sleman",         sub:"Dibudidayakan dengan standar internasional di lahan subur Kabupaten Sleman, Yogyakarta" },
  { bg:"linear-gradient(145deg,#0e2004 0%,#2d5a0a 50%,#0a1a04 100%)",  accent:"#a3d977", emoji:"👨‍🌾", title:"Petani Koperasi Sleman",       sub:"Bergabung bersama ratusan petani melon berpengalaman dalam ekosistem koperasi yang kuat" },
  { bg:"linear-gradient(145deg,#04101e 0%,#0a3060 50%,#020c16 100%)",  accent:"#60a5fa", emoji:"📊", title:"Teknologi Klasifikasi Cerdas",   sub:"Algoritma Naive Bayes Laplace Smoothing menentukan kelayakan ekspor secara akurat" },
  { bg:"linear-gradient(145deg,#120a00 0%,#5a2c00 50%,#0e0800 100%)",  accent:"#fbbf24", emoji:"🌍", title:"Ekspor ke Mancanegara",          sub:"Melon Sleman menembus pasar Asia Tenggara, Timur Tengah, dan mancanegara lainnya" },
];

function Carousel() {
  const [cur, setCur]   = useState(0);
  const [dir, setDir]   = useState("L");
  const [anim, setAnim] = useState(true);
  const timerRef = useRef(null);

  function go(n, d) {
    const next = (n + SLIDES.length) % SLIDES.length;
    setDir(d);
    setAnim(false);
    setTimeout(() => { setCur(next); setAnim(true); }, 30);
  }

  useEffect(() => {
    timerRef.current = setInterval(() => go(cur + 1, "L"), 5500);
    return () => clearInterval(timerRef.current);
  }, [cur]);

  const sl = SLIDES[cur];
  const accent = sl.accent;

  // SVG overlays per slide
  const overlays = [
    // melons
    <svg key="m" width="100%" height="100%" style={{position:"absolute",inset:0,opacity:0.1}} viewBox="0 0 800 500" preserveAspectRatio="xMidYMid slice">
      {[0,1,2,3,4,5,6,7,8,9,10,11].map(i => (
        <g key={i} transform={`translate(${(i%4)*210+55},${Math.floor(i/4)*165+40})`}>
          <ellipse cx="0" cy="0" rx="52" ry="40" fill="none" stroke={accent} strokeWidth="2"/>
          {[0,1,2,3,4,5].map(j=><path key={j} d={`M ${52*Math.cos(j*60*Math.PI/180)} ${40*Math.sin(j*60*Math.PI/180)} Q 0 0 ${52*Math.cos((j+1)*60*Math.PI/180)} ${40*Math.sin((j+1)*60*Math.PI/180)}`} fill="none" stroke={accent} strokeWidth="0.8"/>)}
          <circle cx="10" cy="-46" r="3" fill={accent}/><path d="M 10 -43 Q 16 -33 8 -26" fill="none" stroke={accent} strokeWidth="1.5"/>
        </g>
      ))}
    </svg>,
    // farmer
    <svg key="f" width="100%" height="100%" style={{position:"absolute",inset:0,opacity:0.09}} viewBox="0 0 800 500" preserveAspectRatio="xMidYMid slice">
      {[0,1,2,3,4,5].map(i=>(
        <g key={i} transform={`translate(${i%3*260+90},${Math.floor(i/3)*230+40})`}>
          <circle cx="0" cy="0" r="28" fill="none" stroke={accent} strokeWidth="2"/>
          <circle cx="0" cy="-10" r="11" fill="none" stroke={accent} strokeWidth="1.5"/>
          <path d="M -20 18 Q -20 4 0 4 Q 20 4 20 18 L 20 55 L -20 55 Z" fill="none" stroke={accent} strokeWidth="1.5"/>
          <path d="M -20 23 L -42 48" stroke={accent} strokeWidth="1.5"/>
          <path d="M 20 23 L 42 48" stroke={accent} strokeWidth="1.5"/>
          <rect x="-55" y="60" width="110" height="5" rx="2" fill="none" stroke={accent} strokeWidth="1"/>
        </g>
      ))}
    </svg>,
    // tech network
    <svg key="t" width="100%" height="100%" style={{position:"absolute",inset:0,opacity:0.1}} viewBox="0 0 800 500" preserveAspectRatio="xMidYMid slice">
      {[[120,100],[350,80],[600,120],[180,300],[420,280],[650,320],[280,420],[520,400]].map(([x,y],i)=>(
        <g key={i}><rect x={x-18} y={y-18} width="36" height="36" rx="7" fill="none" stroke={accent} strokeWidth="1.5"/><circle cx={x} cy={y} r="5" fill={accent}/></g>
      ))}
      {[[120,100,350,80],[350,80,600,120],[120,100,180,300],[350,80,420,280],[600,120,650,320],[180,300,420,280],[420,280,280,420],[650,320,520,400]].map(([x1,y1,x2,y2],i)=>(
        <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke={accent} strokeWidth="1" strokeDasharray="6,4"/>
      ))}
    </svg>,
    // globe/export
    <svg key="e" width="100%" height="100%" style={{position:"absolute",inset:0,opacity:0.09}} viewBox="0 0 800 500" preserveAspectRatio="xMidYMid slice">
      <ellipse cx="400" cy="250" rx="340" ry="210" fill="none" stroke={accent} strokeWidth="1.5"/>
      <ellipse cx="400" cy="250" rx="190" ry="210" fill="none" stroke={accent} strokeWidth="1"/>
      <ellipse cx="400" cy="250" rx="75" ry="210" fill="none" stroke={accent} strokeWidth="1"/>
      <line x1="60" y1="250" x2="740" y2="250" stroke={accent} strokeWidth="1"/>
      {[[180,130],[580,160],[250,380],[620,340]].map(([x,y],i)=>(
        <g key={i}><circle cx={x} cy={y} r="7" fill={accent}/><circle cx={x} cy={y} r="15" fill="none" stroke={accent} strokeWidth="1" opacity="0.5"/></g>
      ))}
      {[[180,130,580,160],[580,160,620,340],[620,340,250,380],[250,380,180,130]].map(([x1,y1,x2,y2],i)=>(
        <path key={i} d={`M ${x1} ${y1} Q 400 ${(y1+y2)/2-40} ${x2} ${y2}`} fill="none" stroke={accent} strokeWidth="1.5" strokeDasharray="8,5"/>
      ))}
    </svg>
  ];

  return (
    <div style={{position:"relative",height:"100vh",overflow:"hidden"}}>
      {/* Background layer with its own animation */}
      <div style={{position:"absolute",inset:0,background:sl.bg,transition:"background 0.8s ease",zIndex:0}}>
        {overlays[cur]}
        {[0,1,2].map(i=>(
          <div key={i} style={{position:"absolute",borderRadius:"50%",
            border:`1px solid rgba(255,255,255,${0.03+i*0.03})`,
            width:`${300+i*200}px`,height:`${300+i*200}px`,
            top:"50%",left:"50%",transform:"translate(-50%,-50%)",
            animation:`${i%2?"spinR":"spin"} ${15+i*8}s linear infinite`,pointerEvents:"none"}}/>
        ))}
        {/* Accent glow */}
        <div style={{position:"absolute",top:"50%",left:"50%",transform:"translate(-50%,-50%)",
          width:"600px",height:"600px",borderRadius:"50%",
          background:`radial-gradient(circle,${sl.accent}15 0%,transparent 70%)`,
          pointerEvents:"none"}}/>
      </div>

      {/* Content layer */}
      <div key={cur} style={{position:"relative",zIndex:2,display:"flex",flexDirection:"column",
        alignItems:"center",justifyContent:"center",height:"100%",textAlign:"center",
        padding:"2rem 2rem 10rem",
        animation:anim?(dir==="L"?"slideL 0.55s ease":"slideR 0.55s ease"):"none"}}>
        <div style={{fontSize:"110px",marginBottom:"1rem",animation:"float 3s ease infinite",filter:`drop-shadow(0 0 30px ${sl.accent}80)`}}>{sl.emoji}</div>
        <div style={{fontSize:"11px",letterSpacing:"5px",color:sl.accent,textTransform:"uppercase",marginBottom:"0.6rem",fontFamily:"Georgia,serif"}}>
          CekMelon · Koperasi Petani Sleman
        </div>
        <h2 style={{fontSize:"2.8rem",fontWeight:"700",color:"#fff",margin:"0 0 0.75rem",lineHeight:"1.1",
          textShadow:`0 0 40px ${sl.accent}80`,maxWidth:"700px",fontFamily:"Georgia,serif"}}>{sl.title}</h2>
        <p style={{color:"rgba(255,255,255,0.75)",fontSize:"16px",maxWidth:"540px",lineHeight:"1.8",fontFamily:"Georgia,serif"}}>{sl.sub}</p>
      </div>

      {/* Dots */}
      <div style={{position:"absolute",bottom:"5rem",left:"50%",transform:"translateX(-50%)",display:"flex",gap:"10px",zIndex:10}}>
        {SLIDES.map((_,i)=>(
          <button key={i} onClick={()=>go(i,i>cur?"L":"R")}
            style={{width:i===cur?"28px":"10px",height:"10px",borderRadius:"999px",border:"none",
              background:i===cur?SLIDES[i].accent:"rgba(255,255,255,0.3)",cursor:"pointer",
              transition:"all 0.4s ease",padding:0}}/>
        ))}
      </div>

      {/* Arrows */}
      {[{d:"L",ch:"‹"},{d:"R",ch:"›"}].map(({d,ch})=>(
        <button key={d} className="hov" onClick={()=>go(d==="L"?cur-1:cur+1, d==="L"?"R":"L")}
          style={{position:"absolute",top:"50%",transform:"translateY(-50%)",
            [d==="L"?"left":"right"]:"1.5rem",width:"46px",height:"46px",borderRadius:"50%",
            border:`2px solid rgba(255,255,255,0.3)`,background:"rgba(0,0,0,0.35)",
            color:"#fff",cursor:"pointer",fontSize:"22px",zIndex:10,fontFamily:"Georgia,serif",fontWeight:"700"}}>
          {ch}
        </button>
      ))}

      {/* Slide indicator */}
      <div style={{position:"absolute",bottom:"3rem",right:"2rem",zIndex:10,
        fontSize:"12px",color:"rgba(255,255,255,0.4)",fontFamily:"Georgia,serif"}}>
        {cur+1} / {SLIDES.length}
      </div>
    </div>
  );
}

// ─── ClassifyPanel (outside App) ──────────────────────────────────────────
function ClassifyPanel({ adminDS, userId, userRecords, setUserRecords }) {
  const [inp, setInp] = useState({ berat:"", gula:"", netting:"3", kerusakan:"" });
  const [res, setRes] = useState(null);

  function go() {
    const v = { berat:parseFloat(inp.berat), gula:parseFloat(inp.gula), netting:parseInt(inp.netting), kerusakan:parseFloat(inp.kerusakan) };
    if ([v.berat,v.gula,v.kerusakan].some(isNaN) || !v.netting) { alert("Isi semua kolom dengan benar."); return; }
    const r = naiveBayes(adminDS, v);
    setRes({ ...r, input:v });
    const rec = { ...v, kelas:r.predicted, probLayak:r.probLayak, probTidak:r.probTidak,
      time:new Date().toLocaleString("id-ID"), id:Date.now(), status:"pending" };
    setUserRecords(prev => ({ ...prev, [userId]: [rec, ...(prev[userId]||[]).slice(0,99)] }));
  }

  const nettingLabel = NETTING_OPTIONS.find(o=>o.value===parseInt(inp.netting))?.label || "";

  return (
    <div>
      <div style={{textAlign:"center",marginBottom:"1.75rem"}}>
        <h1 style={{fontSize:"1.8rem",fontWeight:"700",color:C.g5,margin:"0 0 0.3rem"}}>🔬 Klasifikasi Kelayakan Ekspor</h1>
        <p style={{color:"rgba(168,230,181,0.65)",fontSize:"14px"}}>Isi data fisik buah melon, lalu tekan tombol Cek Kelayakan</p>
      </div>
      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:"1.5rem"}}>
        <CardW>
          <H2 s={{color:C.g2}}>📋 Data Input Melon</H2>
          <div style={{marginBottom:"0.9rem"}}>
            <Lbl c="⚖️ Berat (kg)"/>
            <Inp type="number" step="0.1" min="0" placeholder="Contoh: 2.1" value={inp.berat} onChange={e=>setInp(p=>({...p,berat:e.target.value}))}/>
          </div>
          <div style={{marginBottom:"0.9rem"}}>
            <Lbl c="🍬 Kadar Gula (°Bx)"/>
            <Inp type="number" step="0.1" min="0" placeholder="Contoh: 14.5" value={inp.gula} onChange={e=>setInp(p=>({...p,gula:e.target.value}))}/>
          </div>
          <div style={{marginBottom:"0.9rem"}}>
            <Lbl c="🕸️ Kerapatan Netting / Reticula"/>
            <select value={inp.netting} onChange={e=>setInp(p=>({...p,netting:e.target.value}))}
              style={{width:"100%",padding:"10px 13px",borderRadius:"9px",border:`1.5px solid ${C.border}`,
                background:C.offW,color:C.text,fontSize:"14px",fontFamily:"Georgia,serif",outline:"none"}}>
              {NETTING_OPTIONS.map(o=><option key={o.value} value={o.value}>{o.label}</option>)}
            </select>
          </div>
          <div style={{marginBottom:"0.9rem"}}>
            <Lbl c="🔍 Kerusakan Fisik (%)"/>
            <Inp type="number" step="0.1" min="0" max="100" placeholder="Contoh: 1.2" value={inp.kerusakan} onChange={e=>setInp(p=>({...p,kerusakan:e.target.value}))}/>
          </div>
          <BtnAu onClick={go} s={{width:"100%",padding:"13px",fontSize:"15px",marginTop:"0.25rem"}}>
            🔎 Cek Kelayakan Ekspor
          </BtnAu>
        </CardW>

        <div>
          {!res ? (
            <Card s={{textAlign:"center",padding:"3rem 2rem",height:"100%",display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center"}}>
              <div style={{fontSize:"52px",marginBottom:"0.75rem",opacity:"0.3",animation:"float 3s ease infinite"}}>🍈</div>
              <p style={{color:"rgba(168,230,181,0.45)",fontSize:"14px",lineHeight:"1.7"}}>Hasil klasifikasi muncul di sini setelah data diisi</p>
            </Card>
          ) : (
            <div style={{animation:"popIn 0.35s ease"}}>
              <div style={{background:res.predicted==="Layak Ekspor"?"linear-gradient(135deg,#1a5c2a,#2d8a45)":"linear-gradient(135deg,#7f1d1d,#991b1b)",
                borderRadius:"14px",padding:"1.75rem",textAlign:"center",marginBottom:"1rem",
                border:`2px solid ${res.predicted==="Layak Ekspor"?"#4ab860":"#f87171"}`,
                boxShadow:res.predicted==="Layak Ekspor"?"0 8px 32px rgba(74,184,96,0.25)":"0 8px 32px rgba(248,113,113,0.25)"}}>
                <div style={{fontSize:"52px",marginBottom:"0.5rem"}}>{res.predicted==="Layak Ekspor"?"✅":"❌"}</div>
                <div style={{fontSize:"10px",letterSpacing:"3px",textTransform:"uppercase",color:"rgba(255,255,255,0.6)",marginBottom:"0.3rem"}}>Hasil Klasifikasi</div>
                <div style={{fontSize:"1.55rem",fontWeight:"700",color:"#fff",marginBottom:"0.3rem"}}>{res.predicted}</div>
                <div style={{fontSize:"13px",color:"rgba(255,255,255,0.72)"}}>
                  {res.predicted==="Layak Ekspor"?"Memenuhi standar kualitas ekspor":"Tidak memenuhi standar kualitas ekspor"}
                </div>
              </div>
              <CardW s={{marginBottom:"1rem"}}>
                <H2 s={{color:C.g2,fontSize:"0.95rem"}}>📊 Probabilitas Naive Bayes</H2>
                {[{l:"✅ Layak Ekspor",p:res.probLayak,c:"#2d8a45",bg:"#e8f8ec",bar:"linear-gradient(90deg,#2d8a45,#4ab860)"},
                  {l:"❌ Tidak Layak",p:res.probTidak,c:"#dc2626",bg:"#fef2f2",bar:"linear-gradient(90deg,#dc2626,#f87171)"}].map((pb,i)=>(
                  <div key={i} style={{marginBottom:i===0?"0.85rem":0}}>
                    <div style={{display:"flex",justifyContent:"space-between",marginBottom:"4px"}}>
                      <span style={{fontSize:"13px",color:C.muted}}>{pb.l}</span>
                      <strong style={{color:pb.c}}>{pb.p}%</strong>
                    </div>
                    <div style={{background:pb.bg,borderRadius:"999px",height:"9px",overflow:"hidden"}}>
                      <div style={{width:`${pb.p}%`,background:pb.bar,height:"100%",borderRadius:"999px",transition:"width 0.8s ease"}}/>
                    </div>
                  </div>
                ))}
              </CardW>
              <CardW s={{marginBottom:"1rem"}}>
                <H2 s={{color:C.g2,fontSize:"0.95rem"}}>📋 Data Input</H2>
                {[["⚖️ Berat",`${res.input.berat} kg`],["🍬 Gula",`${res.input.gula} °Bx`],
                  ["🕸️ Netting",`${NETTING_OPTIONS.find(o=>o.value===res.input.netting)?.label||res.input.netting}`],
                  ["🔍 Kerusakan",`${res.input.kerusakan}%`]].map(([k,v])=>(
                  <div key={k} style={{display:"flex",justifyContent:"space-between",padding:"6px 0",borderBottom:`1px solid ${C.border}`}}>
                    <span style={{fontSize:"13px",color:C.muted}}>{k}</span><strong style={{fontSize:"13px"}}>{v}</strong>
                  </div>
                ))}
                <div style={{marginTop:"0.75rem",padding:"8px 11px",background:"#fffbeb",borderRadius:"8px",fontSize:"12px",color:"#92400e"}}>
                  ⏳ Menunggu persetujuan admin untuk masuk ke dataset pelatihan.
                </div>
              </CardW>
              <button className="hov" onClick={()=>{ setRes(null); setInp({berat:"",gula:"",netting:"3",kerusakan:""}); }}
                style={{width:"100%",padding:"11px",borderRadius:"9px",border:`2px solid ${C.g4}`,background:"transparent",
                  color:C.g5,cursor:"pointer",fontSize:"14px",fontFamily:"Georgia,serif",fontWeight:"700"}}>
                🔄 Klasifikasi Baru
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ─── Riwayat User Table with filter (outside App) ─────────────────────────
function RiwayatAdmin({ allRecs, setApproveConfirm, setRejectConfirm }) {
  const [ft, setFt] = useState("semua");
  const pending  = allRecs.filter(r => r.status==="pending");
  const approved = allRecs.filter(r => r.status==="approved");
  const rejected = allRecs.filter(r => r.status==="rejected");
  const filtered = ft==="semua"?allRecs : ft==="pending"?pending : ft==="approved"?approved : rejected;

  return (
    <div style={{maxWidth:"1060px",margin:"0 auto",padding:"2rem 1.5rem",animation:"fadeUp 0.4s ease"}}>
      <div style={{marginBottom:"1.25rem"}}>
        <h1 style={{fontSize:"1.7rem",fontWeight:"700",color:C.g5,margin:"0 0 0.2rem"}}>📋 Riwayat Klasifikasi Semua User</h1>
        <p style={{color:"rgba(168,230,181,0.55)",fontSize:"13px",margin:0}}>
          Total: {allRecs.length} · <span style={{color:"#fbbf24"}}>⏳ {pending.length} pending</span> · <span style={{color:C.g4}}>✅ {approved.length} disetujui</span> · <span style={{color:"#f87171"}}>❌ {rejected.length} ditolak</span>
        </p>
      </div>
      <div style={{display:"flex",gap:"6px",marginBottom:"1.25rem",flexWrap:"wrap"}}>
        {[{v:"semua",l:`Semua (${allRecs.length})`},{v:"pending",l:`⏳ Pending (${pending.length})`},
          {v:"approved",l:`✅ Disetujui (${approved.length})`},{v:"rejected",l:`❌ Ditolak (${rejected.length})`}].map(t=>(
          <button key={t.v} className="hov" onClick={()=>setFt(t.v)}
            style={{padding:"8px 16px",borderRadius:"8px",border:"none",
              background:ft===t.v?C.g4:"rgba(74,184,96,0.15)",color:ft===t.v?C.g1:C.g5,
              cursor:"pointer",fontSize:"13px",fontFamily:"Georgia,serif",fontWeight:"700"}}>{t.l}</button>
        ))}
      </div>
      {filtered.length===0 ? (
        <Card s={{textAlign:"center",padding:"3rem"}}><p style={{color:"rgba(168,230,181,0.4)",fontSize:"14px"}}>Tidak ada data.</p></Card>
      ) : (
        <Card s={{overflowX:"auto"}}>
          <table style={{width:"100%",borderCollapse:"collapse",fontSize:"12px"}}>
            <thead>
              <tr>{["No","User","Waktu","Berat","Gula","Netting","Rusak","Kelas","P.Layak","Status","Aksi"].map(h=>(
                <th key={h} style={{textAlign:"left",padding:"8px 8px",color:C.g4,borderBottom:"2px solid rgba(74,184,96,0.35)",fontWeight:"700",whiteSpace:"nowrap"}}>{h}</th>
              ))}</tr>
            </thead>
            <tbody>
              {filtered.map((r,i)=>{
                const nLabel = NETTING_OPTIONS.find(o=>o.value===r.netting)?.label || r.netting;
                return (
                  <tr key={r.id} className="trH" style={{borderBottom:"1px solid rgba(74,184,96,0.08)"}}>
                    <td style={{padding:"7px 8px",color:"rgba(168,230,181,0.4)"}}>{i+1}</td>
                    <td style={{padding:"7px 8px",fontWeight:"600",color:C.g4}}>👤 {r.userName}</td>
                    <td style={{padding:"7px 8px",color:"rgba(168,230,181,0.5)",fontSize:"11px",whiteSpace:"nowrap"}}>{r.time}</td>
                    <td style={{padding:"7px 8px"}}>{r.berat}</td>
                    <td style={{padding:"7px 8px"}}>{r.gula}</td>
                    <td style={{padding:"7px 8px",maxWidth:"120px",fontSize:"11px"}}>{nLabel}</td>
                    <td style={{padding:"7px 8px"}}>{r.kerusakan}</td>
                    <td style={{padding:"7px 8px"}}><Badge ok={r.kelas==="Layak Ekspor"}>{r.kelas==="Layak Ekspor"?"Layak":"Tidak"}</Badge></td>
                    <td style={{padding:"7px 8px",color:"#2d8a45",fontWeight:"700"}}>{r.probLayak}%</td>
                    <td style={{padding:"7px 8px"}}>
                      {r.status==="approved" && <span style={{fontSize:"11px",background:"rgba(74,184,96,0.2)",color:C.g4,padding:"2px 8px",borderRadius:"999px",fontWeight:"700"}}>✅ Disetujui</span>}
                      {r.status==="rejected" && <span style={{fontSize:"11px",background:"rgba(248,113,113,0.2)",color:"#f87171",padding:"2px 8px",borderRadius:"999px",fontWeight:"700"}}>❌ Ditolak</span>}
                      {r.status==="pending"  && <span style={{fontSize:"11px",background:"rgba(251,191,36,0.2)",color:"#fbbf24",padding:"2px 8px",borderRadius:"999px",fontWeight:"700"}}>⏳ Pending</span>}
                    </td>
                    <td style={{padding:"7px 8px"}}>
                      {r.status==="pending" && (
                        <div style={{display:"flex",gap:"5px"}}>
                          <button className="hov" onClick={()=>setApproveConfirm({userId:r.userId,recId:r.id,userName:r.userName,rec:r})}
                            style={{padding:"4px 10px",borderRadius:"6px",border:"none",background:"rgba(74,184,96,0.25)",color:C.g4,cursor:"pointer",fontSize:"11px",fontFamily:"Georgia,serif",fontWeight:"700"}}>✅ Terima</button>
                          <button className="hov" onClick={()=>setRejectConfirm({userId:r.userId,recId:r.id,userName:r.userName})}
                            style={{padding:"4px 10px",borderRadius:"6px",border:"none",background:"rgba(248,113,113,0.2)",color:"#f87171",cursor:"pointer",fontSize:"11px",fontFamily:"Georgia,serif",fontWeight:"700"}}>❌ Tolak</button>
                        </div>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </Card>
      )}
    </div>
  );
}

// ─── Pages enum ────────────────────────────────────────────────────────────
const P = {
  LAND:"land",
  ADMIN_HOME:"ahome", ADMIN_DS:"ads", ADMIN_DS_FORM:"adsf",
  ADMIN_CLASSIFY:"acls", ADMIN_USERS:"ausers", ADMIN_RIWAYAT:"ariwayat",
  USER_HOME:"uhome", USER_CLASSIFY:"ucls", USER_HIST:"uhist"
};

const DEVS = [
  { name:"Nama Dosen Pembimbing", nim:"NIDN: XXXXXXXXX", role:"Dosen Pembimbing I", prodi:"Ilmu Komputer / Data Science", univ:"Universitas XYZ", type:"dosen" },
  { name:"Nama Mahasiswa 1",      nim:"NIM: XXXXXXXXX",  role:"Peneliti Utama & Ketua Tim",         prodi:"Program Studi Informatika", univ:"Universitas XYZ", type:"mhs" },
  { name:"Nama Mahasiswa 2",      nim:"NIM: XXXXXXXXX",  role:"Pengembang Backend & Algoritma",     prodi:"Program Studi Informatika", univ:"Universitas XYZ", type:"mhs" },
  { name:"Nama Mahasiswa 3",      nim:"NIM: XXXXXXXXX",  role:"Pengembang Frontend & UI/UX",        prodi:"Program Studi Informatika", univ:"Universitas XYZ", type:"mhs" },
];

// ── Shared Nav ──────────────────────────────────────────────────────────────
function AppNav({ items, extraRight, page, setPage }) {
  return (
    <nav style={{background:"rgba(8,22,14,0.97)",borderBottom:"2px solid rgba(74,184,96,0.55)",padding:"0 1.5rem",
      display:"flex",alignItems:"center",justifyContent:"space-between",position:"sticky",top:0,zIndex:100}}>
      <div style={{display:"flex",alignItems:"center",gap:"10px",padding:"11px 0"}}>
        <span style={{fontSize:"22px"}}>🍈</span>
        <div>
          <div style={{fontSize:"13px",fontWeight:"700",color:C.g5}}>CekMelon</div>
          <div style={{fontSize:"10px",color:"rgba(168,230,181,0.45)"}}>Koperasi Petani Sleman</div>
        </div>
      </div>
      <div style={{display:"flex",gap:"2px",alignItems:"center",flexWrap:"wrap"}}>
        {items.map(n=>(
          <button key={n.k} className="navH" onClick={()=>setPage(n.k)}
            style={{padding:"8px 13px",borderRadius:"7px",border:"none",background:page===n.k?"#4ab860":"transparent",
              color:page===n.k?C.g1:C.g5,cursor:"pointer",fontSize:"13px",fontFamily:"Georgia,serif",
              fontWeight:page===n.k?"700":"400",transition:"all 0.2s",position:"relative"}}>
            {n.l}
            {n.badge>0 && <span style={{position:"absolute",top:"2px",right:"2px",background:"#dc2626",color:"#fff",
              borderRadius:"999px",minWidth:"16px",height:"16px",fontSize:"10px",fontWeight:"700",
              display:"flex",alignItems:"center",justifyContent:"center",padding:"0 3px"}}>{n.badge}</span>}
          </button>
        ))}
        <div style={{width:"1px",height:"20px",background:"rgba(74,184,96,0.3)",margin:"0 6px"}}/>
        {extraRight}
      </div>
    </nav>
  );
}

// ══════════════════════════════════════════════════════════════════════════
export default function App() {
  const [users,    setUsers]    = useState([{id:1,username:"admin",password:"admin123",role:"admin"}]);
  const [user,     setUser]     = useState(null);
  const [page,     setPage]     = useState(P.LAND);
  const [adminDS,  setAdminDS]  = useState(SEED_DATASET);
  const [editRow,  setEditRow]  = useState(null);
  const [dsForm,   setDsForm]   = useState({berat:"",gula:"",netting:"3",kerusakan:"",kelas:"Layak Ekspor"});
  const [dsErr,    setDsErr]    = useState("");
  const [delDS,    setDelDS]    = useState(null);
  const [userRecords, setUserRecords] = useState({});
  const [landSec,  setLandSec]  = useState("hero");

  // Auth modal state
  const [showModal,  setShowModal]  = useState(false);
  const [modalMode,  setModalMode]  = useState("login");
  const [mLF,        setMLF]        = useState({username:"",password:""});
  const [mRF,        setMRF]        = useState({username:"",password:"",confirm:""});
  const [mErr,       setMErr]       = useState("");

  // User management
  const [editUser,     setEditUser]     = useState(null);
  const [userForm,     setUserForm]     = useState({username:"",password:"",role:"user"});
  const [userFormErr,  setUserFormErr]  = useState("");
  const [delUser,      setDelUser]      = useState(null);
  const [showUserForm, setShowUserForm] = useState(false);

  // Approve/Reject confirm
  const [approveConfirm, setApproveConfirm] = useState(null);
  const [rejectConfirm,  setRejectConfirm]  = useState(null);

  const nextDsId = () => adminDS.length ? Math.max(...adminDS.map(d=>d.id))+1 : 1;

  // ── Auth ────────────────────────────────────────────────────────────────
  function doLogin(un, pw) {
    const f = users.find(u => u.username===un && u.password===pw);
    if (!f) return "Username atau password salah.";
    setUser(f); setPage(f.role==="admin" ? P.ADMIN_HOME : P.USER_HOME); return null;
  }
  function doRegister(un, pw, cf) {
    if (!un||!pw||!cf) return "Semua kolom wajib diisi.";
    if (un.length<3) return "Username minimal 3 karakter.";
    if (pw!==cf) return "Password tidak cocok.";
    if (users.find(u=>u.username===un)) return "Username sudah terdaftar.";
    const nu = { id:Date.now(), username:un, password:pw, role:"user" };
    setUsers(prev => [...prev, nu]); setUser(nu); setPage(P.USER_HOME); return null;
  }
  function doLogout() { setUser(null); setPage(P.LAND); }

  function handleMLogin() {
    const e = doLogin(mLF.username, mLF.password);
    if (e) { setMErr(e); } else { setMErr(""); setMLF({username:"",password:""}); setShowModal(false); }
  }
  function handleMReg() {
    const e = doRegister(mRF.username, mRF.password, mRF.confirm);
    if (e) { setMErr(e); } else { setMErr(""); setMRF({username:"",password:"",confirm:""}); setShowModal(false); }
  }

  // ── Dataset CRUD ──────────────────────────────────────────────────────
  function openCreate() { setEditRow(null); setDsForm({berat:"",gula:"",netting:"3",kerusakan:"",kelas:"Layak Ekspor"}); setDsErr(""); setPage(P.ADMIN_DS_FORM); }
  function openEdit(row) { setEditRow(row); setDsForm({berat:String(row.berat),gula:String(row.gula),netting:String(row.netting),kerusakan:String(row.kerusakan),kelas:row.kelas}); setDsErr(""); setPage(P.ADMIN_DS_FORM); }
  function saveDs() {
    const b=parseFloat(dsForm.berat),g=parseFloat(dsForm.gula),n=parseInt(dsForm.netting),k=parseFloat(dsForm.kerusakan);
    if ([b,g,k].some(isNaN)||!n) { setDsErr("Semua nilai wajib diisi."); return; }
    if (editRow) setAdminDS(prev=>prev.map(d=>d.id===editRow.id?{...d,berat:b,gula:g,netting:n,kerusakan:k,kelas:dsForm.kelas}:d));
    else setAdminDS(prev=>[...prev,{id:nextDsId(),berat:b,gula:g,netting:n,kerusakan:k,kelas:dsForm.kelas}]);
    setPage(P.ADMIN_DS);
  }

  // ── Approve / Reject ───────────────────────────────────────────────────
  function doApprove(userId, recId) {
    const recs = userRecords[userId]||[];
    const rec  = recs.find(r=>r.id===recId);
    if (!rec) return;
    setAdminDS(prev => [...prev, {id:nextDsId(),berat:rec.berat,gula:rec.gula,netting:rec.netting,kerusakan:rec.kerusakan,kelas:rec.kelas}]);
    setUserRecords(prev => ({...prev,[userId]:recs.map(r=>r.id===recId?{...r,status:"approved"}:r)}));
    setApproveConfirm(null);
  }
  function doReject(userId, recId) {
    const recs = userRecords[userId]||[];
    setUserRecords(prev => ({...prev,[userId]:recs.map(r=>r.id===recId?{...r,status:"rejected"}:r)}));
    setRejectConfirm(null);
  }

  // ── User CRUD ──────────────────────────────────────────────────────────
  function openCreateUser() { setEditUser(null); setUserForm({username:"",password:"",role:"user"}); setUserFormErr(""); setShowUserForm(true); }
  function openEditUser(u)  { setEditUser(u); setUserForm({username:u.username,password:u.password,role:u.role}); setUserFormErr(""); setShowUserForm(true); }
  function saveUser() {
    if (!userForm.username||!userForm.password) { setUserFormErr("Username dan password wajib diisi."); return; }
    if (userForm.username.length<3) { setUserFormErr("Username minimal 3 karakter."); return; }
    if (editUser) {
      if (users.find(u=>u.username===userForm.username&&u.id!==editUser.id)) { setUserFormErr("Username sudah digunakan."); return; }
      setUsers(prev=>prev.map(u=>u.id===editUser.id?{...u,...userForm}:u));
    } else {
      if (users.find(u=>u.username===userForm.username)) { setUserFormErr("Username sudah digunakan."); return; }
      setUsers(prev=>[...prev,{id:Date.now(),...userForm}]);
    }
    setShowUserForm(false); setEditUser(null);
  }
  function deleteUser(id) { setUsers(prev=>prev.filter(u=>u.id!==id)); setDelUser(null); }

  // ── Computed ───────────────────────────────────────────────────────────
  const allRecs = users.flatMap(u => (userRecords[u.id]||[]).map(r => ({...r,userName:u.username,userId:u.id})));
  const pendingCount = allRecs.filter(r=>r.status==="pending").length;

  const appBg = { minHeight:"100vh", background:`linear-gradient(145deg,${C.g1} 0%,${C.g2} 55%,${C.g3} 100%)`, fontFamily:"Georgia,serif", color:"#fff" };

  // ════════════════════════════════════════════════════════════════════════
  // LANDING
  // ════════════════════════════════════════════════════════════════════════
  if (page===P.LAND) {
    const lnav=[{id:"hero",l:"Beranda"},{id:"petunjuk",l:"Petunjuk"},{id:"tentang",l:"Tentang"},{id:"pengembang",l:"Pengembang"}];
    return (
      <div style={{...appBg,minHeight:"100vh"}}>
        <style>{GCSS}</style>

        {/* NAV */}
        <nav style={{background:"rgba(8,22,14,0.92)",borderBottom:"1.5px solid rgba(74,184,96,0.45)",padding:"0 2rem",
          display:"flex",alignItems:"center",justifyContent:"space-between",position:"fixed",top:0,left:0,right:0,zIndex:200,backdropFilter:"blur(12px)"}}>
          <div style={{display:"flex",alignItems:"center",gap:"10px",padding:"12px 0"}}>
            <span style={{fontSize:"24px"}}>🍈</span>
            <div><div style={{fontSize:"14px",fontWeight:"700",color:C.g5}}>CekMelon</div><div style={{fontSize:"10px",color:"rgba(168,230,181,0.5)"}}>Koperasi Petani Sleman</div></div>
          </div>
          <div style={{display:"flex",gap:"2px",alignItems:"center"}}>
            {lnav.map(n=>(
              <button key={n.id} className="navH" onClick={()=>setLandSec(n.id)}
                style={{padding:"8px 14px",borderRadius:"7px",border:"none",background:landSec===n.id?"#4ab860":"transparent",
                  color:landSec===n.id?C.g1:C.g5,cursor:"pointer",fontSize:"13px",fontFamily:"Georgia,serif",fontWeight:landSec===n.id?"700":"400",transition:"all 0.2s"}}>{n.l}
              </button>
            ))}
            <div style={{width:"1px",height:"20px",background:"rgba(74,184,96,0.3)",margin:"0 8px"}}/>
            <BtnAu onClick={()=>{setShowModal(true);setModalMode("login");setMErr("");}} s={{padding:"8px 18px",fontSize:"13px"}}>Masuk / Daftar</BtnAu>
          </div>
        </nav>

        {/* HERO */}
        {landSec==="hero" && (
          <div style={{paddingTop:"60px",position:"relative"}}>
            <Carousel/>
            <div style={{position:"absolute",bottom:"11rem",left:"50%",transform:"translateX(-50%)",
              display:"flex",gap:"1rem",zIndex:10,flexWrap:"wrap",justifyContent:"center"}}>
              <BtnAu onClick={()=>{setShowModal(true);setModalMode("login");setMErr("");}}
                s={{padding:"14px 36px",fontSize:"16px",boxShadow:"0 8px 24px rgba(200,147,10,0.4)"}}>✦ Mulai Klasifikasi</BtnAu>
              <BtnO onClick={()=>setLandSec("petunjuk")} s={{padding:"14px 36px",fontSize:"16px"}}>Petunjuk Penggunaan</BtnO>
            </div>
            <div style={{background:"rgba(8,22,14,0.92)",borderTop:"1px solid rgba(74,184,96,0.28)",display:"flex",justifyContent:"center",flexWrap:"wrap",backdropFilter:"blur(8px)"}}>
              {[{n:"10+",l:"Data Pelatihan"},{n:"4",l:"Parameter Kualitas"},{n:"Naive Bayes",l:"Algoritma AI"},{n:"100%",l:"Objektif & Akurat"}].map((st,i)=>(
                <div key={i} style={{padding:"1.25rem 2.5rem",textAlign:"center",borderRight:i<3?"1px solid rgba(74,184,96,0.2)":"none"}}>
                  <div style={{fontSize:"1.4rem",fontWeight:"700",color:C.g4,marginBottom:"2px"}}>{st.n}</div>
                  <div style={{fontSize:"12px",color:"rgba(168,230,181,0.6)"}}>{st.l}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* PETUNJUK */}
        {landSec==="petunjuk" && (
          <div style={{maxWidth:"800px",margin:"0 auto",padding:"5rem 1.5rem 3rem",animation:"fadeUp 0.5s ease"}}>
            <div style={{textAlign:"center",marginBottom:"2.5rem"}}>
              <div style={{fontSize:"48px",marginBottom:"0.5rem"}}>📖</div>
              <h2 style={{fontSize:"1.9rem",fontWeight:"700",color:C.g5,margin:"0 0 0.4rem"}}>Petunjuk Penggunaan</h2>
            </div>
            {[{st:"01",ic:"🔐",ti:"Daftar / Login",de:"Klik 'Masuk / Daftar' di kanan atas. Login atau buat akun baru.",tips:["Demo admin: admin / admin123","Username minimal 3 karakter"]},
              {st:"02",ic:"🔬",ti:"Lakukan Klasifikasi",de:"Masuk ke Klasifikasi, isi 4 parameter fisik melon, tekan Cek Kelayakan.",tips:["Berat (kg)","Kadar Gula (°Bx)","Netting — pilih dari dropdown","Kerusakan Fisik (%)"]},
              {st:"03",ic:"📊",ti:"Baca Hasil & Menunggu Admin",de:"Sistem menampilkan hasil. Admin mereview dan menyetujui/menolak data untuk masuk dataset.",tips:["Hasil tersimpan di riwayat Anda","Status: ⏳ Pending / ✅ Disetujui / ❌ Ditolak"]},
              {st:"04",ic:"🗂️",ti:"Admin: Kelola Data",de:"Admin approve/tolak riwayat user, CRUD dataset, dan kelola user.",tips:["Data yang disetujui masuk ke dataset pelatihan","CRUD penuh hanya untuk admin"]}].map(item=>(
              <Card key={item.st} s={{display:"grid",gridTemplateColumns:"60px 1fr",gap:"1.25rem",marginBottom:"1rem"}} className="cardHov">
                <div style={{textAlign:"center"}}><div style={{fontSize:"28px",marginBottom:"4px"}}>{item.ic}</div><div style={{fontSize:"10px",letterSpacing:"2px",color:C.g4,fontWeight:"700"}}>STEP {item.st}</div></div>
                <div>
                  <div style={{fontWeight:"700",color:C.g5,fontSize:"1rem",marginBottom:"0.3rem"}}>{item.ti}</div>
                  <p style={{color:"rgba(168,230,181,0.8)",fontSize:"13px",lineHeight:"1.7",margin:"0 0 0.5rem"}}>{item.de}</p>
                  <ul style={{margin:0,paddingLeft:"1rem",color:C.g4,fontSize:"13px",lineHeight:"1.9"}}>{item.tips.map((t,i)=><li key={i}>{t}</li>)}</ul>
                </div>
              </Card>
            ))}
          </div>
        )}

        {/* TENTANG */}
        {landSec==="tentang" && (
          <div style={{maxWidth:"860px",margin:"0 auto",padding:"5rem 1.5rem 3rem",animation:"fadeUp 0.5s ease"}}>
            <div style={{textAlign:"center",marginBottom:"2.5rem"}}>
              <div style={{fontSize:"48px",marginBottom:"0.5rem"}}>🌿</div>
              <h2 style={{fontSize:"1.9rem",fontWeight:"700",color:C.g5,margin:"0 0 0.4rem"}}>Tentang Objek Penelitian</h2>
            </div>
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:"1.25rem",marginBottom:"1.25rem"}}>
              {[{ic:"🍈",ti:"Buah Melon (Cucumis melo)",de:"Tanaman hortikultura keluarga Cucurbitaceae banyak dibudidayakan di Sleman, Yogyakarta. Nilai ekonominya tinggi sebagai komoditas ekspor ke Asia Tenggara dan Timur Tengah. Kualitasnya ditentukan berat, kadar gula, kerapatan netting, dan kerusakan fisik."},
                {ic:"🏢",ti:"Koperasi Petani Sleman",de:"Menaungi petani melon di Kabupaten Sleman, DIY. Berperan dalam pemasaran, standarisasi kualitas, dan fasilitasi ekspor. CekMelon meningkatkan efisiensi seleksi buah layak ekspor secara objektif dan konsisten."}].map((x,i)=>(
                <Card key={i} className="cardHov" s={{transition:"all 0.2s"}}>
                  <div style={{fontSize:"28px",marginBottom:"0.5rem"}}>{x.ic}</div>
                  <H2>{x.ti}</H2>
                  <p style={{color:"rgba(168,230,181,0.85)",fontSize:"14px",lineHeight:"1.8",margin:0}}>{x.de}</p>
                </Card>
              ))}
            </div>
            <Card s={{marginBottom:"1.25rem"}}>
              <H2>📐 Standar Parameter Kelayakan Ekspor</H2>
              <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(180px,1fr))",gap:"0.9rem"}}>
                {[{ic:"⚖️",p:"Berat Buah",r:"1.8–3.5 kg",d:"Berat optimal pasar ekspor"},
                  {ic:"🍬",p:"Kadar Gula",r:"≥ 13 °Bx",d:"Diukur refraktometer"},
                  {ic:"🕸️",p:"Netting",r:"Rapat–Sangat Rapat",d:"Kerapatan jaring kulit"},
                  {ic:"🔍",p:"Kerusakan",r:"≤ 5%",d:"Persentase luas cacat"}].map((p,i)=>(
                  <div key={i} style={{background:"rgba(255,255,255,0.06)",borderRadius:"10px",padding:"0.9rem",border:"1px solid rgba(74,184,96,0.18)"}}>
                    <div style={{fontSize:"22px",marginBottom:"0.3rem"}}>{p.ic}</div>
                    <div style={{fontWeight:"700",color:C.g5,fontSize:"13px",marginBottom:"3px"}}>{p.p}</div>
                    <div style={{background:"rgba(74,184,96,0.18)",display:"inline-block",padding:"2px 8px",borderRadius:"999px",fontSize:"11px",color:C.g4,marginBottom:"6px"}}>{p.r}</div>
                    <div style={{fontSize:"12px",color:"rgba(168,230,181,0.65)",lineHeight:"1.5"}}>{p.d}</div>
                  </div>
                ))}
              </div>
            </Card>
            <Card>
              <H2>🧠 Metode: Naive Bayes Laplace Smoothing</H2>
              <p style={{color:"rgba(168,230,181,0.85)",fontSize:"14px",lineHeight:"1.8",marginBottom:"0.9rem"}}>Gaussian Naive Bayes dengan Laplace Smoothing (α=1). Efektif pada dataset kecil, hasil probabilistik, dan mencegah probabilitas nol.</p>
              <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:"0.9rem"}}>
                {[{ti:"Keunggulan",it:["Efektif pada dataset kecil","Komputasi ringan & cepat","Hasil berupa probabilitas","Cegah probabilitas nol"]},
                  {ti:"Alur Kerja",it:["Hitung prior probability","Estimasi likelihood Gaussian","Gabungkan prior × likelihood","Pilih posterior tertinggi"]}].map((b,i)=>(
                  <div key={i} style={{background:"rgba(255,255,255,0.06)",borderRadius:"10px",padding:"0.9rem"}}>
                    <div style={{fontWeight:"700",color:C.g4,marginBottom:"0.4rem",fontSize:"13px"}}>{b.ti}</div>
                    <ul style={{margin:0,paddingLeft:"1rem",color:"rgba(168,230,181,0.8)",fontSize:"13px",lineHeight:"1.9"}}>{b.it.map((it,j)=><li key={j}>{it}</li>)}</ul>
                  </div>
                ))}
              </div>
            </Card>
          </div>
        )}

        {/* PENGEMBANG */}
        {landSec==="pengembang" && (
          <div style={{maxWidth:"860px",margin:"0 auto",padding:"5rem 1.5rem 3rem",animation:"fadeUp 0.5s ease"}}>
            <div style={{textAlign:"center",marginBottom:"2.5rem"}}>
              <div style={{fontSize:"48px",marginBottom:"0.5rem"}}>👨‍💻</div>
              <h2 style={{fontSize:"1.9rem",fontWeight:"700",color:C.g5,margin:"0 0 0.4rem"}}>Pengembang Aplikasi</h2>
              <p style={{color:"rgba(168,230,181,0.65)",fontSize:"14px"}}>Tim di balik sistem CekMelon</p>
            </div>
            <div style={{display:"flex",justifyContent:"center",marginBottom:"1.75rem"}}>
              <Card s={{textAlign:"center",maxWidth:"300px",width:"100%",background:"rgba(200,147,10,0.12)",border:"1px solid rgba(200,147,10,0.4)"}}>
                <div style={{display:"flex",justifyContent:"center",marginBottom:"1rem"}}>
                  <div style={{borderRadius:"50%",overflow:"hidden",width:"88px",height:"88px",border:"3px solid rgba(200,147,10,0.6)",boxShadow:"0 4px 20px rgba(200,147,10,0.3)"}}>
                    <AvatarSVG seed={DEVS[0].name} size={88} role="dosen"/>
                  </div>
                </div>
                <div style={{fontWeight:"700",color:C.goldL,fontSize:"1rem",marginBottom:"0.2rem"}}>{DEVS[0].name}</div>
                <div style={{fontSize:"12px",color:C.goldL,opacity:0.8,marginBottom:"0.4rem"}}>{DEVS[0].nim}</div>
                <div style={{background:"rgba(200,147,10,0.25)",display:"inline-block",padding:"3px 12px",borderRadius:"999px",fontSize:"12px",color:C.goldL,marginBottom:"0.6rem",fontWeight:"700"}}>{DEVS[0].role}</div>
                <div style={{fontSize:"13px",color:"rgba(246,196,74,0.7)",lineHeight:"1.7"}}><div>{DEVS[0].prodi}</div><div>{DEVS[0].univ}</div></div>
              </Card>
            </div>
            <div style={{display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:"1.25rem",marginBottom:"1.5rem"}}>
              {DEVS.slice(1).map((d,i)=>(
                <Card key={i} className="cardHov" s={{textAlign:"center",transition:"all 0.22s"}}>
                  <div style={{display:"flex",justifyContent:"center",marginBottom:"0.9rem"}}>
                    <div style={{borderRadius:"50%",overflow:"hidden",width:"72px",height:"72px",border:"2px solid rgba(74,184,96,0.5)",boxShadow:"0 4px 14px rgba(74,184,96,0.2)"}}>
                      <AvatarSVG seed={d.name} size={72} role="mhs"/>
                    </div>
                  </div>
                  <div style={{fontWeight:"700",color:C.g5,fontSize:"0.9rem",marginBottom:"0.2rem"}}>{d.name}</div>
                  <div style={{fontSize:"11px",color:C.g4,marginBottom:"0.35rem"}}>{d.nim}</div>
                  <div style={{background:"rgba(74,184,96,0.18)",display:"inline-block",padding:"3px 10px",borderRadius:"999px",fontSize:"11px",color:C.g4,marginBottom:"0.5rem",fontWeight:"700"}}>{d.role}</div>
                  <div style={{fontSize:"12px",color:"rgba(168,230,181,0.6)",lineHeight:"1.6"}}><div>{d.prodi}</div><div>{d.univ}</div></div>
                </Card>
              ))}
            </div>
            <div style={{textAlign:"center",color:"rgba(168,230,181,0.3)",fontSize:"12px"}}>© 2024 CekMelon · Koperasi Petani Sleman</div>
          </div>
        )}

        {/* AUTH MODAL */}
        <ModalBox show={showModal} onClose={()=>setShowModal(false)}>
          <div style={{textAlign:"center",marginBottom:"1.25rem"}}>
            <div style={{fontSize:"40px",marginBottom:"0.4rem"}}>🍈</div>
            <h2 style={{fontSize:"1.2rem",color:C.g2,margin:"0 0 0.2rem",fontWeight:"700"}}>
              {modalMode==="login"?"Masuk ke Sistem":"Daftar Akun Baru"}
            </h2>
            <p style={{fontSize:"12px",color:C.muted,margin:0}}>CekMelon · Koperasi Petani Sleman</p>
          </div>
          {mErr && <div style={{background:"#fde8e8",color:"#9b1c1c",padding:"9px 13px",borderRadius:"8px",fontSize:"13px",marginBottom:"0.9rem"}}>⚠ {mErr}</div>}
          {modalMode==="login" ? (
            <div>
              <div style={{marginBottom:"0.8rem"}}><Lbl c="Username"/><Inp type="text" placeholder="Masukkan username" value={mLF.username} onChange={e=>setMLF(p=>({...p,username:e.target.value}))} onKeyDown={e=>{if(e.key==="Enter")handleMLogin();}}/></div>
              <div style={{marginBottom:"1.2rem"}}><Lbl c="Password"/><Inp type="password" placeholder="Masukkan password" value={mLF.password} onChange={e=>setMLF(p=>({...p,password:e.target.value}))} onKeyDown={e=>{if(e.key==="Enter")handleMLogin();}}/></div>
              <BtnG onClick={handleMLogin} s={{width:"100%",padding:"12px"}}>Masuk →</BtnG>
              <div style={{marginTop:"0.6rem",padding:"8px 12px",background:"#e8f8ec",borderRadius:"8px",fontSize:"12px",color:C.muted}}><strong>Demo:</strong> admin / admin123</div>
              <p style={{textAlign:"center",fontSize:"13px",color:C.muted,marginTop:"0.9rem"}}>Belum punya akun?{" "}<span style={{color:C.g3,cursor:"pointer",fontWeight:"700"}} onClick={()=>{setModalMode("register");setMErr("");}}>Daftar di sini</span></p>
              <p style={{textAlign:"center",fontSize:"12px",marginTop:"0.3rem"}}><span className="landLink" style={{color:C.muted,cursor:"pointer"}} onClick={()=>setShowModal(false)}>← Kembali ke Landing Page</span></p>
            </div>
          ) : (
            <div>
              <div style={{marginBottom:"0.75rem"}}><Lbl c="Username"/><Inp type="text" placeholder="Min. 3 karakter" value={mRF.username} onChange={e=>setMRF(p=>({...p,username:e.target.value}))}/></div>
              <div style={{marginBottom:"0.75rem"}}><Lbl c="Password"/><Inp type="password" placeholder="Buat password" value={mRF.password} onChange={e=>setMRF(p=>({...p,password:e.target.value}))}/></div>
              <div style={{marginBottom:"1.2rem"}}><Lbl c="Konfirmasi Password"/><Inp type="password" placeholder="Ulangi password" value={mRF.confirm} onChange={e=>setMRF(p=>({...p,confirm:e.target.value}))} onKeyDown={e=>{if(e.key==="Enter")handleMReg();}}/></div>
              <BtnG onClick={handleMReg} s={{width:"100%",padding:"12px"}}>Daftar & Masuk →</BtnG>
              <p style={{textAlign:"center",fontSize:"13px",color:C.muted,marginTop:"0.9rem"}}>Sudah punya akun?{" "}<span style={{color:C.g3,cursor:"pointer",fontWeight:"700"}} onClick={()=>{setModalMode("login");setMErr("");}}>Masuk di sini</span></p>
              <p style={{textAlign:"center",fontSize:"12px",marginTop:"0.3rem"}}><span className="landLink" style={{color:C.muted,cursor:"pointer"}} onClick={()=>setShowModal(false)}>← Kembali ke Landing Page</span></p>
            </div>
          )}
        </ModalBox>
      </div>
    );
  }

  // ════════════════════════════════════════════════════════════════════════
  // ADMIN
  // ════════════════════════════════════════════════════════════════════════
  if (user?.role==="admin") {
    const anav = [
      {k:P.ADMIN_HOME,     l:"🏠 Dashboard"},
      {k:P.ADMIN_CLASSIFY, l:"🔬 Klasifikasi"},
      {k:P.ADMIN_DS,       l:"🗂️ Dataset"},
      {k:P.ADMIN_RIWAYAT,  l:"📋 Riwayat User", badge:pendingCount},
      {k:P.ADMIN_USERS,    l:"👥 Kelola User"},
    ];
    const extraRight = (
      <>
        <span style={{fontSize:"12px",color:C.g5,marginRight:"4px"}}>
          👤 {user.username}
          <span style={{fontSize:"10px",background:C.gold,color:"#fff",padding:"2px 6px",borderRadius:"4px",marginLeft:"5px"}}>ADMIN</span>
        </span>
        <button className="hov" onClick={doLogout} style={{padding:"7px 12px",borderRadius:"7px",border:"none",background:"transparent",color:"#f87171",cursor:"pointer",fontSize:"12px",fontFamily:"Georgia,serif",fontWeight:"600"}}>Keluar</button>
      </>
    );

    return (
      <div style={appBg}>
        <style>{GCSS}</style>

        {/* Delete DS */}
        <ModalBox show={!!delDS} onClose={()=>setDelDS(null)}>
          <div style={{textAlign:"center"}}><div style={{fontSize:"48px",marginBottom:"0.75rem"}}>⚠️</div>
            <h3 style={{color:"#9b1c1c",margin:"0 0 0.5rem"}}>Hapus Data ID #{delDS}?</h3>
            <p style={{color:C.muted,fontSize:"14px",margin:"0 0 1.5rem"}}>Tidak dapat dibatalkan.</p>
            <div style={{display:"flex",gap:"0.75rem",justifyContent:"center"}}>
              <BtnO onClick={()=>setDelDS(null)} color={C.muted}>Batal</BtnO>
              <button className="hov" onClick={()=>{setAdminDS(prev=>prev.filter(d=>d.id!==delDS));setDelDS(null);}} style={{padding:"11px 22px",borderRadius:"9px",border:"none",background:"#dc2626",color:"#fff",cursor:"pointer",fontSize:"14px",fontFamily:"Georgia,serif",fontWeight:"700"}}>Ya, Hapus</button>
            </div>
          </div>
        </ModalBox>

        {/* Delete User */}
        <ModalBox show={!!delUser} onClose={()=>setDelUser(null)}>
          <div style={{textAlign:"center"}}><div style={{fontSize:"48px",marginBottom:"0.75rem"}}>⚠️</div>
            <h3 style={{color:"#9b1c1c",margin:"0 0 0.5rem"}}>Hapus User?</h3>
            <p style={{color:C.muted,fontSize:"14px",margin:"0 0 1.5rem"}}>User dihapus permanen.</p>
            <div style={{display:"flex",gap:"0.75rem",justifyContent:"center"}}>
              <BtnO onClick={()=>setDelUser(null)} color={C.muted}>Batal</BtnO>
              <button className="hov" onClick={()=>deleteUser(delUser)} style={{padding:"11px 22px",borderRadius:"9px",border:"none",background:"#dc2626",color:"#fff",cursor:"pointer",fontSize:"14px",fontFamily:"Georgia,serif",fontWeight:"700"}}>Ya, Hapus</button>
            </div>
          </div>
        </ModalBox>

        {/* User Form */}
        <ModalBox show={showUserForm} onClose={()=>setShowUserForm(false)}>
          <div style={{textAlign:"center",marginBottom:"1.25rem"}}><div style={{fontSize:"36px",marginBottom:"0.3rem"}}>{editUser?"✏️":"➕"}</div>
            <h2 style={{fontSize:"1.2rem",color:C.g2,margin:"0 0 0.2rem",fontWeight:"700"}}>{editUser?"Edit User":"Tambah User Baru"}</h2></div>
          {userFormErr && <div style={{background:"#fde8e8",color:"#9b1c1c",padding:"9px 13px",borderRadius:"8px",fontSize:"13px",marginBottom:"0.9rem"}}>⚠ {userFormErr}</div>}
          <div style={{marginBottom:"0.8rem"}}><Lbl c="Username"/><Inp type="text" placeholder="Min. 3 karakter" value={userForm.username} onChange={e=>setUserForm(p=>({...p,username:e.target.value}))}/></div>
          <div style={{marginBottom:"0.8rem"}}><Lbl c="Password"/><Inp type="password" placeholder="Password" value={userForm.password} onChange={e=>setUserForm(p=>({...p,password:e.target.value}))}/></div>
          <div style={{marginBottom:"1.2rem"}}><Lbl c="Role"/>
            <select value={userForm.role} onChange={e=>setUserForm(p=>({...p,role:e.target.value}))} style={{width:"100%",padding:"10px 13px",borderRadius:"9px",border:`1.5px solid ${C.border}`,background:C.offW,color:C.text,fontSize:"14px",fontFamily:"Georgia,serif",outline:"none"}}>
              <option value="user">User</option><option value="admin">Admin</option>
            </select>
          </div>
          <div style={{display:"flex",gap:"0.75rem"}}>
            <BtnG onClick={saveUser} s={{flex:1,padding:"12px"}}>{editUser?"💾 Simpan":"➕ Tambah"}</BtnG>
            <BtnO onClick={()=>setShowUserForm(false)} color={C.muted} s={{padding:"12px 16px"}}>Batal</BtnO>
          </div>
        </ModalBox>

        {/* Approve Confirm */}
        <ModalBox show={!!approveConfirm} onClose={()=>setApproveConfirm(null)}>
          {approveConfirm && (() => {
            const rec = approveConfirm.rec;
            return (
              <div style={{textAlign:"center"}}>
                <div style={{fontSize:"48px",marginBottom:"0.75rem"}}>✅</div>
                <h3 style={{color:C.g2,margin:"0 0 0.4rem",fontSize:"1.1rem"}}>Terima Data ini?</h3>
                <p style={{color:C.muted,fontSize:"13px",margin:"0 0 0.75rem"}}>Data dari <strong>{approveConfirm.userName}</strong> akan ditambahkan ke dataset pelatihan.</p>
                <div style={{background:"#e8f8ec",borderRadius:"10px",padding:"0.9rem",marginBottom:"1.25rem",textAlign:"left"}}>
                  {[["Berat",`${rec.berat} kg`],["Gula",`${rec.gula} °Bx`],
                    ["Netting",NETTING_OPTIONS.find(o=>o.value===rec.netting)?.label||rec.netting],
                    ["Kerusakan",`${rec.kerusakan}%`],["Kelas",rec.kelas]].map(([k,v])=>(
                    <div key={k} style={{display:"flex",justifyContent:"space-between",padding:"4px 0",borderBottom:`1px solid ${C.border}`,fontSize:"13px",color:C.text}}><span style={{color:C.muted}}>{k}</span><strong>{v}</strong></div>
                  ))}
                </div>
                <div style={{display:"flex",gap:"0.75rem",justifyContent:"center"}}>
                  <BtnO onClick={()=>setApproveConfirm(null)} color={C.muted}>Batal</BtnO>
                  <BtnG onClick={()=>doApprove(approveConfirm.userId,approveConfirm.recId)}>✅ Ya, Terima</BtnG>
                </div>
              </div>
            );
          })()}
        </ModalBox>

        {/* Reject Confirm */}
        <ModalBox show={!!rejectConfirm} onClose={()=>setRejectConfirm(null)}>
          {rejectConfirm && (
            <div style={{textAlign:"center"}}>
              <div style={{fontSize:"48px",marginBottom:"0.75rem"}}>❌</div>
              <h3 style={{color:"#9b1c1c",margin:"0 0 0.4rem",fontSize:"1.1rem"}}>Tolak Data ini?</h3>
              <p style={{color:C.muted,fontSize:"13px",margin:"0 0 1.25rem"}}>Data dari <strong>{rejectConfirm.userName}</strong> akan ditolak dan tidak masuk ke dataset pelatihan.</p>
              <div style={{display:"flex",gap:"0.75rem",justifyContent:"center"}}>
                <BtnO onClick={()=>setRejectConfirm(null)} color={C.muted}>Batal</BtnO>
                <button className="hov" onClick={()=>doReject(rejectConfirm.userId,rejectConfirm.recId)}
                  style={{padding:"11px 22px",borderRadius:"9px",border:"none",background:"#dc2626",color:"#fff",cursor:"pointer",fontSize:"14px",fontFamily:"Georgia,serif",fontWeight:"700"}}>❌ Ya, Tolak</button>
              </div>
            </div>
          )}
        </ModalBox>

        <AppNav items={anav} extraRight={extraRight} page={page} setPage={setPage}/>

        {/* DASHBOARD */}
        {page===P.ADMIN_HOME && (
          <div style={{maxWidth:"900px",margin:"0 auto",padding:"2rem 1.5rem",animation:"fadeUp 0.4s ease"}}>
            <div style={{marginBottom:"1.75rem"}}>
              <h1 style={{fontSize:"1.8rem",fontWeight:"700",color:C.g5,margin:"0 0 0.3rem"}}>👋 Selamat datang, {user.username}!</h1>
              <p style={{color:"rgba(168,230,181,0.6)",fontSize:"14px",margin:0}}>Panel administrasi CekMelon</p>
            </div>
            <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(155px,1fr))",gap:"1rem",marginBottom:"2rem"}}>
              {[{ic:"📦",v:adminDS.length,l:"Total Dataset",pg:P.ADMIN_DS},
                {ic:"✅",v:adminDS.filter(d=>d.kelas==="Layak Ekspor").length,l:"Layak Ekspor",pg:null},
                {ic:"❌",v:adminDS.filter(d=>d.kelas!=="Layak Ekspor").length,l:"Tidak Layak",pg:null},
                {ic:"⏳",v:pendingCount,l:"Menunggu Review",pg:P.ADMIN_RIWAYAT},
                {ic:"👥",v:users.length,l:"Total User",pg:P.ADMIN_USERS}].map((st,i)=>(
                <Card key={i} className="cardHov" s={{textAlign:"center",padding:"1.25rem",cursor:st.pg?"pointer":"default",transition:"all 0.22s"}} onClick={()=>st.pg&&setPage(st.pg)}>
                  <div style={{fontSize:"26px",marginBottom:"0.3rem"}}>{st.ic}</div>
                  <div style={{fontSize:"1.4rem",fontWeight:"700",color:st.l==="Menunggu Review"&&st.v>0?"#fbbf24":C.g5}}>{st.v}</div>
                  <div style={{fontSize:"12px",color:"rgba(168,230,181,0.6)"}}>{st.l}</div>
                </Card>
              ))}
            </div>
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:"1.25rem"}}>
              <Card>
                <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:"1rem"}}>
                  <H2 s={{margin:0}}>⏳ Menunggu Review</H2>
                  <button className="hov" onClick={()=>setPage(P.ADMIN_RIWAYAT)} style={{fontSize:"12px",color:C.g4,background:"none",border:"none",cursor:"pointer",fontFamily:"Georgia,serif",fontWeight:"700"}}>Lihat Semua →</button>
                </div>
                {allRecs.filter(r=>r.status==="pending").slice(0,4).length===0 ? (
                  <p style={{color:"rgba(168,230,181,0.4)",fontSize:"13px"}}>Tidak ada data pending.</p>
                ) : allRecs.filter(r=>r.status==="pending").slice(0,4).map(r=>(
                  <div key={r.id} style={{display:"flex",justifyContent:"space-between",alignItems:"center",padding:"7px 0",borderBottom:"1px solid rgba(74,184,96,0.1)"}}>
                    <div>
                      <div style={{fontSize:"12px",color:C.g5}}>👤 {r.userName}</div>
                      <div style={{fontSize:"11px",color:"rgba(168,230,181,0.5)"}}>{r.berat}kg · {r.gula}°Bx · <Badge ok={r.kelas==="Layak Ekspor"}>{r.kelas==="Layak Ekspor"?"Layak":"Tidak"}</Badge></div>
                    </div>
                    <div style={{display:"flex",gap:"4px"}}>
                      <button className="hov" onClick={()=>setApproveConfirm({userId:r.userId,recId:r.id,userName:r.userName,rec:r})}
                        style={{padding:"3px 8px",borderRadius:"5px",border:"none",background:"rgba(74,184,96,0.25)",color:C.g4,cursor:"pointer",fontSize:"11px",fontFamily:"Georgia,serif",fontWeight:"700"}}>✅</button>
                      <button className="hov" onClick={()=>setRejectConfirm({userId:r.userId,recId:r.id,userName:r.userName})}
                        style={{padding:"3px 8px",borderRadius:"5px",border:"none",background:"rgba(248,113,113,0.2)",color:"#f87171",cursor:"pointer",fontSize:"11px",fontFamily:"Georgia,serif",fontWeight:"700"}}>❌</button>
                    </div>
                  </div>
                ))}
              </Card>
              <Card>
                <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:"1rem"}}>
                  <H2 s={{margin:0}}>👥 User Terdaftar</H2>
                  <button className="hov" onClick={()=>setPage(P.ADMIN_USERS)} style={{fontSize:"12px",color:C.g4,background:"none",border:"none",cursor:"pointer",fontFamily:"Georgia,serif",fontWeight:"700"}}>Kelola →</button>
                </div>
                {users.slice(0,5).map(u=>(
                  <div key={u.id} style={{display:"flex",justifyContent:"space-between",alignItems:"center",padding:"7px 0",borderBottom:"1px solid rgba(74,184,96,0.1)"}}>
                    <span style={{fontSize:"13px"}}>👤 {u.username}</span>
                    <div style={{display:"flex",gap:"6px",alignItems:"center"}}>
                      <span style={{fontSize:"11px",color:"rgba(168,230,181,0.5)"}}>{(userRecords[u.id]||[]).length}x</span>
                      <span style={{fontSize:"10px",background:u.role==="admin"?"rgba(200,147,10,0.25)":"rgba(74,184,96,0.18)",color:u.role==="admin"?C.goldL:C.g4,padding:"2px 7px",borderRadius:"999px",fontWeight:"700"}}>{u.role}</span>
                    </div>
                  </div>
                ))}
              </Card>
            </div>
          </div>
        )}

        {page===P.ADMIN_CLASSIFY && (
          <div style={{maxWidth:"900px",margin:"0 auto",padding:"2rem 1.5rem",animation:"fadeUp 0.4s ease"}}>
            <ClassifyPanel adminDS={adminDS} userId={user.id} userRecords={userRecords} setUserRecords={setUserRecords}/>
          </div>
        )}

        {page===P.ADMIN_RIWAYAT && (
          <RiwayatAdmin allRecs={allRecs} setApproveConfirm={setApproveConfirm} setRejectConfirm={setRejectConfirm}/>
        )}

        {page===P.ADMIN_DS && (
          <div style={{maxWidth:"960px",margin:"0 auto",padding:"2rem 1.5rem",animation:"fadeUp 0.4s ease"}}>
            <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:"1.5rem",flexWrap:"wrap",gap:"1rem"}}>
              <div><h1 style={{fontSize:"1.6rem",fontWeight:"700",color:C.g5,margin:"0 0 0.2rem"}}>🗂️ Kelola Dataset</h1><p style={{color:"rgba(168,230,181,0.55)",fontSize:"13px",margin:0}}>Total: {adminDS.length} data pelatihan</p></div>
              <BtnAu onClick={openCreate} s={{padding:"10px 20px",fontSize:"13px"}}>＋ Tambah Data</BtnAu>
            </div>
            <Card s={{overflowX:"auto"}}>
              <table style={{width:"100%",borderCollapse:"collapse",fontSize:"13px"}}>
                <thead><tr>{["No","Berat(kg)","Gula(°Bx)","Netting","Kerusakan(%)","Kelas","Aksi"].map(h=>(
                  <th key={h} style={{textAlign:"left",padding:"9px 10px",color:C.g4,borderBottom:"2px solid rgba(74,184,96,0.35)",fontWeight:"700",whiteSpace:"nowrap"}}>{h}</th>
                ))}</tr></thead>
                <tbody>{adminDS.map((d,i)=>(
                  <tr key={d.id} className="trH" style={{borderBottom:"1px solid rgba(74,184,96,0.08)"}}>
                    <td style={{padding:"8px 10px",color:"rgba(168,230,181,0.4)"}}>{i+1}</td>
                    <td style={{padding:"8px 10px"}}>{d.berat}</td>
                    <td style={{padding:"8px 10px"}}>{d.gula}</td>
                    <td style={{padding:"8px 10px",fontSize:"12px"}}>{NETTING_OPTIONS.find(o=>o.value===d.netting)?.label||d.netting}</td>
                    <td style={{padding:"8px 10px"}}>{d.kerusakan}</td>
                    <td style={{padding:"8px 10px"}}><Badge ok={d.kelas==="Layak Ekspor"}>{d.kelas}</Badge></td>
                    <td style={{padding:"8px 10px"}}>
                      <div style={{display:"flex",gap:"6px"}}>
                        <button className="hov" onClick={()=>openEdit(d)} style={{padding:"5px 11px",borderRadius:"6px",border:"none",background:"rgba(74,184,96,0.22)",color:C.g4,cursor:"pointer",fontSize:"12px",fontFamily:"Georgia,serif",fontWeight:"600"}}>✏️ Edit</button>
                        <button className="hov" onClick={()=>setDelDS(d.id)} style={{padding:"5px 11px",borderRadius:"6px",border:"none",background:"rgba(220,38,38,0.16)",color:"#f87171",cursor:"pointer",fontSize:"12px",fontFamily:"Georgia,serif",fontWeight:"600"}}>🗑️ Hapus</button>
                      </div>
                    </td>
                  </tr>
                ))}</tbody>
              </table>
            </Card>
          </div>
        )}

        {page===P.ADMIN_DS_FORM && (
          <div style={{maxWidth:"500px",margin:"0 auto",padding:"2rem 1.5rem",animation:"fadeUp 0.4s ease"}}>
            <div style={{marginBottom:"1.5rem"}}><h1 style={{fontSize:"1.6rem",fontWeight:"700",color:C.g5,margin:"0 0 0.2rem"}}>{editRow?"✏️ Edit Data":"➕ Tambah Data Baru"}</h1></div>
            <CardW>
              {dsErr && <div style={{background:"#fde8e8",color:"#9b1c1c",padding:"9px 13px",borderRadius:"8px",fontSize:"13px",marginBottom:"1rem"}}>⚠ {dsErr}</div>}
              <div style={{marginBottom:"0.9rem"}}><Lbl c="⚖️ Berat (kg)"/><Inp type="number" step="0.1" min="0" placeholder="Contoh: 2.1" value={dsForm.berat} onChange={e=>setDsForm(p=>({...p,berat:e.target.value}))}/></div>
              <div style={{marginBottom:"0.9rem"}}><Lbl c="🍬 Kadar Gula (°Bx)"/><Inp type="number" step="0.1" min="0" placeholder="Contoh: 14.5" value={dsForm.gula} onChange={e=>setDsForm(p=>({...p,gula:e.target.value}))}/></div>
              <div style={{marginBottom:"0.9rem"}}><Lbl c="🕸️ Kerapatan Netting"/>
                <select value={dsForm.netting} onChange={e=>setDsForm(p=>({...p,netting:e.target.value}))} style={{width:"100%",padding:"10px 13px",borderRadius:"9px",border:`1.5px solid ${C.border}`,background:C.offW,color:C.text,fontSize:"14px",fontFamily:"Georgia,serif",outline:"none"}}>
                  {NETTING_OPTIONS.map(o=><option key={o.value} value={o.value}>{o.label}</option>)}
                </select>
              </div>
              <div style={{marginBottom:"0.9rem"}}><Lbl c="🔍 Kerusakan (%)"/><Inp type="number" step="0.1" min="0" max="100" placeholder="Contoh: 1.2" value={dsForm.kerusakan} onChange={e=>setDsForm(p=>({...p,kerusakan:e.target.value}))}/></div>
              <div style={{marginBottom:"1.2rem"}}><Lbl c="🏷️ Kelas / Label"/>
                <select value={dsForm.kelas} onChange={e=>setDsForm(p=>({...p,kelas:e.target.value}))} style={{width:"100%",padding:"10px 13px",borderRadius:"9px",border:`1.5px solid ${C.border}`,background:C.offW,color:C.text,fontSize:"14px",fontFamily:"Georgia,serif",outline:"none"}}>
                  <option value="Layak Ekspor">Layak Ekspor</option><option value="Tidak Layak Ekspor">Tidak Layak Ekspor</option>
                </select>
              </div>
              <div style={{display:"flex",gap:"0.75rem"}}>
                <BtnG onClick={saveDs} s={{flex:1,padding:"12px"}}>{editRow?"💾 Simpan":"➕ Tambah"}</BtnG>
                <BtnO onClick={()=>setPage(P.ADMIN_DS)} color={C.muted} s={{padding:"12px 16px"}}>Batal</BtnO>
              </div>
            </CardW>
          </div>
        )}

        {page===P.ADMIN_USERS && (
          <div style={{maxWidth:"860px",margin:"0 auto",padding:"2rem 1.5rem",animation:"fadeUp 0.4s ease"}}>
            <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:"1.5rem",flexWrap:"wrap",gap:"1rem"}}>
              <div><h1 style={{fontSize:"1.6rem",fontWeight:"700",color:C.g5,margin:"0 0 0.2rem"}}>👥 Kelola User</h1><p style={{color:"rgba(168,230,181,0.55)",fontSize:"13px",margin:0}}>Total: {users.length} akun</p></div>
              <BtnAu onClick={openCreateUser} s={{padding:"10px 20px",fontSize:"13px"}}>＋ Tambah User</BtnAu>
            </div>
            <Card s={{overflowX:"auto"}}>
              <table style={{width:"100%",borderCollapse:"collapse",fontSize:"13px"}}>
                <thead><tr>{["No","Avatar","Username","Role","Klasifikasi","Aksi"].map(h=>(
                  <th key={h} style={{textAlign:"left",padding:"9px 10px",color:C.g4,borderBottom:"2px solid rgba(74,184,96,0.35)",fontWeight:"700"}}>{h}</th>
                ))}</tr></thead>
                <tbody>{users.map((u,i)=>(
                  <tr key={u.id} className="trH" style={{borderBottom:"1px solid rgba(74,184,96,0.08)"}}>
                    <td style={{padding:"8px 10px",color:"rgba(168,230,181,0.4)"}}>{i+1}</td>
                    <td style={{padding:"8px 10px"}}>
                      <div style={{borderRadius:"50%",overflow:"hidden",width:"36px",height:"36px",border:"1.5px solid rgba(74,184,96,0.4)"}}>
                        <AvatarSVG seed={u.username} size={36} role={u.role==="admin"?"dosen":"mhs"}/>
                      </div>
                    </td>
                    <td style={{padding:"8px 10px",fontWeight:"600"}}>👤 {u.username}</td>
                    <td style={{padding:"8px 10px"}}><span style={{fontSize:"11px",background:u.role==="admin"?"rgba(200,147,10,0.25)":"rgba(74,184,96,0.18)",color:u.role==="admin"?C.goldL:C.g4,padding:"2px 10px",borderRadius:"999px",fontWeight:"700"}}>{u.role}</span></td>
                    <td style={{padding:"8px 10px",color:"rgba(168,230,181,0.6)"}}>{(userRecords[u.id]||[]).length} kali</td>
                    <td style={{padding:"8px 10px"}}>
                      <div style={{display:"flex",gap:"6px"}}>
                        <button className="hov" onClick={()=>openEditUser(u)} style={{padding:"5px 11px",borderRadius:"6px",border:"none",background:"rgba(74,184,96,0.22)",color:C.g4,cursor:"pointer",fontSize:"12px",fontFamily:"Georgia,serif",fontWeight:"600"}}>✏️ Edit</button>
                        {u.id!==user.id && <button className="hov" onClick={()=>setDelUser(u.id)} style={{padding:"5px 11px",borderRadius:"6px",border:"none",background:"rgba(220,38,38,0.16)",color:"#f87171",cursor:"pointer",fontSize:"12px",fontFamily:"Georgia,serif",fontWeight:"600"}}>🗑️ Hapus</button>}
                      </div>
                    </td>
                  </tr>
                ))}</tbody>
              </table>
            </Card>
          </div>
        )}
      </div>
    );
  }

  // ════════════════════════════════════════════════════════════════════════
  // USER
  // ════════════════════════════════════════════════════════════════════════
  if (user?.role==="user") {
    const my = userRecords[user.id]||[];
    const unav = [
      {k:P.USER_HOME,     l:"🏠 Beranda"},
      {k:P.USER_CLASSIFY, l:"🔬 Klasifikasi"},
      {k:P.USER_HIST,     l:"📋 Riwayat Saya"},
    ];
    const extraRight = (
      <>
        <span style={{fontSize:"12px",color:C.g5,marginRight:"4px"}}>👤 {user.username}</span>
        <button className="hov" onClick={doLogout} style={{padding:"7px 12px",borderRadius:"7px",border:"none",background:"transparent",color:"#f87171",cursor:"pointer",fontSize:"12px",fontFamily:"Georgia,serif",fontWeight:"600"}}>Keluar</button>
      </>
    );
    return (
      <div style={appBg}>
        <style>{GCSS}</style>
        <AppNav items={unav} extraRight={extraRight} page={page} setPage={setPage}/>

        {page===P.USER_HOME && (
          <div style={{maxWidth:"860px",margin:"0 auto",padding:"2rem 1.5rem",animation:"fadeUp 0.4s ease"}}>
            <div style={{textAlign:"center",padding:"2rem 0 2.5rem"}}>
              <div style={{fontSize:"72px",marginBottom:"0.6rem",animation:"float 3s ease infinite"}}>🍈</div>
              <h1 style={{fontSize:"2rem",fontWeight:"700",color:C.g5,margin:"0 0 0.4rem"}}>Selamat datang, {user.username}!</h1>
              <p style={{color:"rgba(168,230,181,0.75)",fontSize:"15px",maxWidth:"460px",margin:"0 auto 1.75rem",lineHeight:"1.7"}}>Gunakan sistem untuk menentukan kelayakan ekspor buah melon.</p>
              <div style={{display:"flex",gap:"1rem",justifyContent:"center",flexWrap:"wrap"}}>
                <BtnAu onClick={()=>setPage(P.USER_CLASSIFY)} s={{padding:"13px 30px",fontSize:"15px"}}>✦ Mulai Klasifikasi</BtnAu>
                <BtnO onClick={()=>setPage(P.USER_HIST)} s={{padding:"13px 30px",fontSize:"15px"}}>📋 Riwayat Saya</BtnO>
              </div>
            </div>
            <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(155px,1fr))",gap:"1rem",marginBottom:"1.5rem"}}>
              {[{ic:"📦",v:adminDS.length,l:"Data Pelatihan"},
                {ic:"📋",v:my.length,l:"Klasifikasi Saya"},
                {ic:"✅",v:my.filter(r=>r.status==="approved").length,l:"Disetujui"},
                {ic:"⏳",v:my.filter(r=>r.status==="pending").length,l:"Menunggu Review"},
                {ic:"❌",v:my.filter(r=>r.status==="rejected").length,l:"Ditolak"}].map((st,i)=>(
                <Card key={i} className="cardHov" s={{textAlign:"center",padding:"1.1rem",transition:"all 0.22s"}}>
                  <div style={{fontSize:"24px",marginBottom:"0.2rem"}}>{st.ic}</div>
                  <div style={{fontSize:"1.3rem",fontWeight:"700",color:C.g5}}>{st.v}</div>
                  <div style={{fontSize:"11px",color:"rgba(168,230,181,0.6)"}}>{st.l}</div>
                </Card>
              ))}
            </div>
            <Card>
              <H2>📌 Panduan Nilai Standar Ekspor</H2>
              <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:"0.75rem",fontSize:"14px",color:"rgba(168,230,181,0.85)"}}>
                <div>⚖️ <strong>Berat ideal:</strong> 1.8 – 3.5 kg</div>
                <div>🍬 <strong>Kadar gula:</strong> ≥ 13 °Bx</div>
                <div>🕸️ <strong>Netting layak:</strong> Rapat / Sangat Rapat</div>
                <div>🔍 <strong>Kerusakan maks:</strong> ≤ 5%</div>
              </div>
            </Card>
          </div>
        )}

        {page===P.USER_CLASSIFY && (
          <div style={{maxWidth:"900px",margin:"0 auto",padding:"2rem 1.5rem",animation:"fadeUp 0.4s ease"}}>
            <ClassifyPanel adminDS={adminDS} userId={user.id} userRecords={userRecords} setUserRecords={setUserRecords}/>
          </div>
        )}

        {page===P.USER_HIST && (
          <div style={{maxWidth:"960px",margin:"0 auto",padding:"2rem 1.5rem",animation:"fadeUp 0.4s ease"}}>
            <div style={{marginBottom:"1.5rem"}}>
              <h1 style={{fontSize:"1.7rem",fontWeight:"700",color:C.g5,margin:"0 0 0.2rem"}}>📋 Riwayat Klasifikasi Saya</h1>
              <p style={{color:"rgba(168,230,181,0.55)",fontSize:"13px",margin:0}}>
                Total: {my.length} · ✅ {my.filter(r=>r.status==="approved").length} disetujui · ⏳ {my.filter(r=>r.status==="pending").length} pending · ❌ {my.filter(r=>r.status==="rejected").length} ditolak
              </p>
            </div>
            {my.length===0 ? (
              <Card s={{textAlign:"center",padding:"3rem"}}>
                <div style={{fontSize:"52px",marginBottom:"0.75rem",opacity:"0.3"}}>📋</div>
                <p style={{color:"rgba(168,230,181,0.4)",fontSize:"14px",marginBottom:"1rem"}}>Belum ada riwayat.</p>
                <BtnAu onClick={()=>setPage(P.USER_CLASSIFY)} s={{padding:"10px 22px",fontSize:"14px"}}>Mulai Klasifikasi</BtnAu>
              </Card>
            ) : (
              <Card s={{overflowX:"auto"}}>
                <table style={{width:"100%",borderCollapse:"collapse",fontSize:"12px"}}>
                  <thead><tr>{["No","Waktu","Berat","Gula","Netting","Rusak","Kelas","P.Layak","P.Tidak","Status"].map(h=>(
                    <th key={h} style={{textAlign:"left",padding:"8px 8px",color:C.g4,borderBottom:"2px solid rgba(74,184,96,0.35)",fontWeight:"700",whiteSpace:"nowrap"}}>{h}</th>
                  ))}</tr></thead>
                  <tbody>{my.map((r,i)=>{
                    const nLabel = NETTING_OPTIONS.find(o=>o.value===r.netting)?.label||r.netting;
                    return (
                      <tr key={r.id} className="trH" style={{borderBottom:"1px solid rgba(74,184,96,0.08)"}}>
                        <td style={{padding:"6px 8px",color:"rgba(168,230,181,0.4)"}}>{i+1}</td>
                        <td style={{padding:"6px 8px",color:"rgba(168,230,181,0.5)",fontSize:"11px",whiteSpace:"nowrap"}}>{r.time}</td>
                        <td style={{padding:"6px 8px"}}>{r.berat}</td>
                        <td style={{padding:"6px 8px"}}>{r.gula}</td>
                        <td style={{padding:"6px 8px",fontSize:"11px",maxWidth:"100px"}}>{nLabel}</td>
                        <td style={{padding:"6px 8px"}}>{r.kerusakan}</td>
                        <td style={{padding:"6px 8px"}}><Badge ok={r.kelas==="Layak Ekspor"}>{r.kelas==="Layak Ekspor"?"Layak":"Tidak"}</Badge></td>
                        <td style={{padding:"6px 8px",color:"#2d8a45",fontWeight:"700"}}>{r.probLayak}%</td>
                        <td style={{padding:"6px 8px",color:"#dc2626",fontWeight:"700"}}>{r.probTidak}%</td>
                        <td style={{padding:"6px 8px"}}>
                          {r.status==="approved" && <span style={{fontSize:"11px",background:"rgba(74,184,96,0.2)",color:C.g4,padding:"2px 8px",borderRadius:"999px",fontWeight:"700"}}>✅ Disetujui</span>}
                          {r.status==="rejected" && <span style={{fontSize:"11px",background:"rgba(248,113,113,0.2)",color:"#f87171",padding:"2px 8px",borderRadius:"999px",fontWeight:"700"}}>❌ Ditolak</span>}
                          {r.status==="pending"  && <span style={{fontSize:"11px",background:"rgba(251,191,36,0.15)",color:"#fbbf24",padding:"2px 8px",borderRadius:"999px",fontWeight:"700"}}>⏳ Pending</span>}
                        </td>
                      </tr>
                    );
                  })}</tbody>
                </table>
              </Card>
            )}
          </div>
        )}
      </div>
    );
  }

  return null;
}
