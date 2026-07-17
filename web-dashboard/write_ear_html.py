import os

html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates', 'ear.html')

html = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ear Aesthetic Analysis</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root { --bg-color:#f8fafc; --sidebar-bg:#ffffff; --card-bg:#ffffff; --text-primary:#1e293b; --text-secondary:#64748b; --text-highlight:#94a3b8; --border-color:#f1f5f9; --bar-bg:#e2e8f0; --bar-fill:#94a3b8; --marker-color:#1e293b; --tooltip-bg:#64748b; }
        body { margin:0; font-family:'Inter',sans-serif; background:var(--bg-color); color:var(--text-primary); display:flex; height:100vh; overflow:hidden; }
        .sidebar { width:250px; background:var(--sidebar-bg); border-right:1px solid var(--border-color); padding:32px 0; overflow-y:auto; flex-shrink:0; }
        .sidebar-menu { list-style:none; padding:0; margin:0; font-size:14px; color:var(--text-secondary); }
        .sidebar-menu li { padding:12px 24px; cursor:pointer; display:flex; align-items:center; }
        .sidebar-menu li:hover { color:var(--text-primary); }
        .sidebar-menu li.has-children { font-weight:500; color:var(--text-primary); }
        .sidebar-menu li.has-children::before { content:"v"; margin-right:12px; font-size:10px; font-family:monospace; }
        .sidebar-menu li.collapsed::before { content:">"; }
        .sidebar-submenu { list-style:none; padding:0; margin:0; }
        .sidebar-submenu li { padding:10px 24px 10px 48px; margin:4px 12px; border-radius:8px; color:var(--text-secondary); }
        .sidebar-submenu li:hover { background:#f8fafc; color:var(--text-primary); }
        .sidebar-submenu li.active { background:#ffffff; color:var(--text-primary); box-shadow:0 1px 3px rgba(0,0,0,0.05); font-weight:500; }
        .main-content { flex-grow:1; padding:40px; overflow-y:auto; }
        .header h1 { font-size:32px; font-weight:600; margin:0 0 12px 0; letter-spacing:-0.5px; }
        .header h1 span { color:var(--text-highlight); }
        .header p { font-size:15px; color:var(--text-secondary); margin:0 0 32px 0; }
        .header p strong { color:var(--text-primary); font-weight:600; }
        .image-card { background:var(--card-bg); border:1px solid var(--border-color); border-radius:12px; padding:32px 40px; margin-bottom:24px; display:flex; justify-content:center; align-items:flex-start; gap:48px; box-shadow:0 2px 10px rgba(0,0,0,0.02); }
        .image-card img { max-width:360px; max-height:360px; width:auto; height:auto; object-fit:contain; border-radius:10px; }
        .bottom-split { display:grid; grid-template-columns:1fr 1fr; gap:24px; }
        .grid-header { font-size:18px; font-weight:500; margin-bottom:16px; }
        .metric-grid { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
        .class-card { background:var(--card-bg); border:1px solid var(--border-color); border-radius:12px; padding:24px 20px; position:relative; box-shadow:0 2px 10px rgba(0,0,0,0.02); transition:all 0.2s; cursor:default; }
        .class-card:hover { border-color:#cbd5e1; }
        .class-label { font-family:monospace; font-size:10px; color:var(--text-highlight); text-transform:uppercase; letter-spacing:1px; margin-bottom:40px; font-weight:600; }
        .class-value { font-size:22px; font-weight:500; color:var(--text-primary); }
        .class-card .tooltip { position:absolute; bottom:100%; left:50%; transform:translateX(-50%) translateY(-10px); background:var(--tooltip-bg); color:white; padding:16px; border-radius:8px; font-size:13px; line-height:1.4; width:200px; opacity:0; visibility:hidden; transition:all 0.2s; z-index:10; box-shadow:0 10px 15px -3px rgba(0,0,0,0.1); }
        .class-card .tooltip::after { content:''; position:absolute; top:100%; left:50%; transform:translateX(-50%); border-width:6px; border-style:solid; border-color:var(--tooltip-bg) transparent transparent transparent; }
        .class-card .tooltip .tooltip-title { font-family:monospace; font-size:10px; text-transform:uppercase; margin-bottom:8px; font-weight:600; color:#f8fafc; }
        .class-card:hover .tooltip { opacity:1; visibility:visible; transform:translateX(-50%) translateY(-5px); }
        .carousel-container { position:relative; background:var(--card-bg); border:1px solid var(--border-color); border-radius:12px; box-shadow:0 2px 10px rgba(0,0,0,0.02); overflow:hidden; min-height:350px; }
        .carousel-nav { position:absolute; top:24px; right:24px; display:flex; gap:8px; align-items:center; color:#cbd5e1; font-size:14px; z-index:20; user-select:none; }
        .carousel-nav .arrow { cursor:pointer; padding:0 4px; transition:color 0.2s; }
        .carousel-nav .arrow:hover { color:var(--text-primary); }
        .carousel-nav .dot { width:6px; height:6px; background:#e2e8f0; border-radius:50%; cursor:pointer; }
        .carousel-nav .dot.active { background:#94a3b8; }
        .range-card { padding:24px; position:absolute; top:0; left:0; width:100%; height:100%; box-sizing:border-box; transition:transform 0.4s ease, opacity 0.4s ease; opacity:0; pointer-events:none; background:var(--card-bg); }
        .range-card.active { opacity:1; pointer-events:auto; transform:translateX(0); }
        .range-card.prev { transform:translateX(-100%); }
        .range-card.next { transform:translateX(100%); }
        .range-card h3 { margin:0 0 8px 0; font-size:18px; font-weight:500; }
        .range-card h3 .highlight { color:var(--text-highlight); }
        .range-card p.desc { font-size:13px; color:var(--text-secondary); margin:0 0 24px 0; line-height:1.5; max-width:85%; }
        .range-box { background:#fafafa; border-radius:8px; padding:24px; border:1px solid #f1f5f9; }
        .range-box-header { display:flex; justify-content:space-between; font-family:monospace; font-size:10px; color:var(--text-highlight); text-transform:uppercase; letter-spacing:0.5px; margin-bottom:12px; font-weight:600; }
        .range-value { font-size:28px; font-weight:600; margin-bottom:40px; }
        .range-value span { font-size:14px; font-weight:500; margin-left:2px; color:var(--text-primary); }
        .slider-container { position:relative; height:40px; }
        .slider-track { position:absolute; bottom:20px; left:0; right:0; height:4px; background:var(--bar-bg); border-radius:2px; }
        .slider-fill { position:absolute; bottom:20px; height:4px; background:var(--bar-fill); border-radius:2px; }
        .slider-marker { position:absolute; bottom:24px; width:4px; height:4px; background:var(--marker-color); transform:translateX(-50%); transition:left 0.5s ease-out; }
        .slider-marker::after { content:''; position:absolute; top:4px; left:50%; transform:translateX(-50%); width:1px; height:12px; background:var(--bar-fill); }
        .slider-labels { position:absolute; bottom:0; left:0; right:0; display:flex; justify-content:space-between; font-size:12px; font-weight:600; }
        #results { display:none; }
        .metric-row { display:flex; justify-content:space-between; align-items:center; padding:8px 12px; border-radius:8px; background:#f8fafc; border:1px solid var(--border-color); font-size:13px; margin-bottom:6px; }
        .metric-row:hover { background:#f1f5f9; }
        .metric-label { color:var(--text-secondary); font-weight:500; }
        .metric-val { color:var(--text-primary); font-weight:600; }
        .not-uploaded-banner { background:#fffbeb; border:1px solid #fde68a; border-radius:12px; padding:32px; text-align:center; margin-bottom:24px; }
        .not-uploaded-banner h3 { margin:0 0 8px 0; font-size:18px; color:#92400e; }
        .not-uploaded-banner p { margin:0; color:#a16207; font-size:14px; }
        .not-uploaded-banner a { color:#1e293b; font-weight:600; text-decoration:underline; }
    </style>
</head>
<body>
<div class="sidebar">
    <ul class="sidebar-menu">
        <li class="collapsed">Introduction</li>
        <li class="collapsed">Facial Assessments</li>
        <li class="has-children">Features Analysis</li>
        <ul class="sidebar-submenu">
            <li onclick="window.location.href='/'">Dashboard</li>
            <li onclick="window.location.href='/eyebrows'">Eyebrows</li>
            <li onclick="window.location.href='/eyes'">Eyes</li>
            <li onclick="window.location.href='/nose'">Nose</li>
            <li onclick="window.location.href='/lips'">Lips</li>
            <li onclick="window.location.href='/cheeks'">Cheeks</li>
            <li onclick="window.location.href='/jaw'">Jaw</li>
            <li onclick="window.location.href='/chin'">Chin</li>
            <li onclick="window.location.href='/hair'">Hair</li>
            <li onclick="window.location.href='/smile'">Smile</li>
            <li onclick="window.location.href='/neck'">Neck</li>
            <li class="active" onclick="window.location.href='/ear'">Ear</li>
            <li>Skin</li>
        </ul>
        <li class="collapsed">Protocol</li>
    </ul>
</div>
<div class="main-content">
    <div class="header">
        <h1>An overview of your <span>ears</span></h1>
        <p>The ears contribute to <strong>facial symmetry and proportion</strong>. Ear size, shape, and positioning all play a role in overall facial harmony.</p>
    </div>
    <div id="notUploadedBanner" class="not-uploaded-banner" style="display:none;">
        <h3>&#9888;&#65039; Side Face Image Not Uploaded</h3>
        <p>Ear analysis requires a side-profile photo. Please go back to the <a href="/">Dashboard</a> and upload a side face image alongside your front face image.</p>
    </div>
    <div id="results">
        <div class="image-card">
            <div style="text-align:center;">
                <p style="font-size:11px;color:var(--text-highlight);margin-bottom:8px;font-weight:600;letter-spacing:1px;text-transform:uppercase;">Cropped Ear</p>
                <img id="earCroppedImg" src="" alt="Cropped Ear">
                <p style="margin-top:8px;font-size:12px;color:var(--text-secondary);">Masked Region</p>
            </div>
            <div style="text-align:center;">
                <p style="font-size:11px;color:var(--text-highlight);margin-bottom:8px;font-weight:600;letter-spacing:1px;text-transform:uppercase;">Full Image with Measurements</p>
                <img id="earOverlayImg" src="" alt="Ear with Lines">
                <p style="margin-top:8px;font-size:12px;color:var(--text-secondary);">Caliper Lines</p>
            </div>
        </div>
        <div class="bottom-split">
            <div>
                <div class="grid-header">Summary of your ears</div>
                <div class="metric-grid">
                    <div class="class-card">
                        <div class="tooltip"><div class="tooltip-title">EAR SIZE</div>Classified by ear height relative to typical population ranges (55-68 mm).</div>
                        <div class="class-label">EAR SIZE</div>
                        <div class="class-value" id="valEarSize">--</div>
                    </div>
                    <div class="class-card">
                        <div class="tooltip"><div class="tooltip-title">EAR PROMINENCE</div>How much the ear protrudes, estimated from its top width measurement.</div>
                        <div class="class-label">EAR PROMINENCE</div>
                        <div class="class-value" id="valEarProminence">--</div>
                    </div>
                    <div class="class-card">
                        <div class="tooltip"><div class="tooltip-title">EAR SHAPE</div>Derived from the width-to-height ratio of the segmented ear region.</div>
                        <div class="class-label">EAR SHAPE</div>
                        <div class="class-value" id="valEarShape">--</div>
                    </div>
                    <div class="class-card">
                        <div class="tooltip"><div class="tooltip-title">EAR POSITION</div>Vertical positioning of the ear relative to facial landmarks.</div>
                        <div class="class-label">EAR POSITION</div>
                        <div class="class-value" id="valEarPosition">--</div>
                    </div>
                </div>
            </div>
            <div class="carousel-container">
                <div class="carousel-nav">
                    <span class="arrow" onclick="prevSlide()">&#8249;</span>
                    <span class="dot active" onclick="goToSlide(0)"></span>
                    <span class="dot" onclick="goToSlide(1)"></span>
                    <span class="arrow" onclick="nextSlide()">&#8250;</span>
                </div>
                <div class="range-card active" id="slide-0">
                    <h3>Markedly <span class="highlight">greater</span> than typical range</h3>
                    <p class="desc">Your ears show increased vertical span, giving a more prominent ear outline that draws attention to the upper and lower ear landmarks.</p>
                    <div class="range-box">
                        <div class="range-box-header"><div>EAR LENGTH</div><div>ROBOFLOW SEGMENTATION</div></div>
                        <div class="range-value" id="valEarHeightMm">--<span> mm</span></div>
                        <div class="slider-container">
                            <div class="slider-track"></div>
                            <div class="slider-fill" style="left:15%;right:15%;"></div>
                            <div class="slider-marker" id="markerHeight" style="left:50%;"></div>
                            <div class="slider-labels" style="padding:0 10%;"><div>55.00 mm</div><div>68.00 mm</div></div>
                        </div>
                    </div>
                </div>
                <div class="range-card next" id="slide-1">
                    <h3>Ear <span class="highlight">Width</span></h3>
                    <p class="desc">Width measured at the upper portion of the ear from the segmented outline. Typical range 18-30 mm.</p>
                    <div class="range-box">
                        <div class="range-box-header"><div>EAR WIDTH (TOP)</div><div>ROBOFLOW SEGMENTATION</div></div>
                        <div class="range-value" id="valEarWidthMm">--<span> mm</span></div>
                        <div class="slider-container">
                            <div class="slider-track"></div>
                            <div class="slider-fill" style="left:15%;right:15%;"></div>
                            <div class="slider-marker" id="markerWidth" style="left:50%;"></div>
                            <div class="slider-labels" style="padding:0 10%;"><div>18.00 mm</div><div>30.00 mm</div></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <div style="margin-top:24px;background:var(--card-bg);border:1px solid var(--border-color);border-radius:12px;padding:24px;box-shadow:0 2px 10px rgba(0,0,0,0.02);">
            <div class="grid-header" style="margin-bottom:16px;">All Ear Metrics</div>
            <div id="earMetricsDetail" style="display:grid;grid-template-columns:1fr 1fr;gap:8px 32px;"></div>
        </div>
    </div>
</div>
<script>
    let currentSlide=0; const totalSlides=2;
    function updateCarousel(){for(let i=0;i<totalSlides;i++){const s=document.getElementById('slide-'+i);s.classList.remove('active','prev','next');if(i===currentSlide)s.classList.add('active');else if(i<currentSlide)s.classList.add('prev');else s.classList.add('next');}document.querySelectorAll('.carousel-nav .dot').forEach((d,i)=>d.classList.toggle('active',i===currentSlide));}
    function nextSlide(){currentSlide=(currentSlide+1)%totalSlides;updateCarousel();}
    function prevSlide(){currentSlide=(currentSlide-1+totalSlides)%totalSlides;updateCarousel();}
    function goToSlide(i){currentSlide=i;updateCarousel();}
    function setVal(id,v){const el=document.getElementById(id);if(el)el.innerText=(v&&v!=='N/A')?v:'N/A';}
    function setMark(id,val,min,max){if(!val||val==='N/A')return;const p=15+((parseFloat(val)-min)/(max-min))*70;document.getElementById(id).style.left=Math.max(5,Math.min(95,p))+'%';}
    window.addEventListener('DOMContentLoaded',function(){
        const dataStr=sessionStorage.getItem('faceData_ear');
        const sideUploaded=sessionStorage.getItem('sideImageUploaded')==='true';
        if(!sideUploaded||!dataStr){document.getElementById('notUploadedBanner').style.display='block';return;}
        const data=JSON.parse(dataStr);
        if(data.error){
            document.getElementById('notUploadedBanner').style.display='block';
            document.getElementById('notUploadedBanner').querySelector('h3').textContent='\u26a0\ufe0f Ear Analysis Failed';
            document.getElementById('notUploadedBanner').querySelector('p').textContent=data.error;
            return;
        }
        document.getElementById('results').style.display='block';
        setVal('valEarSize',data.ear_size);
        setVal('valEarProminence',data.ear_prominence);
        setVal('valEarShape',data.ear_shape);
        setVal('valEarPosition',data.ear_position);
        document.getElementById('valEarHeightMm').innerHTML=(data.ear_height_mm||'N/A')+'<span> mm</span>';
        document.getElementById('valEarWidthMm').innerHTML=(data.ear_width_mm||'N/A')+'<span> mm</span>';
        setMark('markerHeight',data.ear_height_mm,55,68);
        setMark('markerWidth',data.ear_width_mm,18,30);
        if(data.ear_cropped)document.getElementById('earCroppedImg').src=data.ear_cropped;
        if(data.ear_overlay)document.getElementById('earOverlayImg').src=data.ear_overlay;
        const det=document.getElementById('earMetricsDetail');
        const rows=[
            ['Ear Height (px)',data.ear_height_px||'N/A'],
            ['Ear Width (px)',data.ear_width_px||'N/A'],
            ['Ear Height (mm)',data.ear_height_mm?data.ear_height_mm+' mm':'N/A'],
            ['Ear Width (mm)',data.ear_width_mm?data.ear_width_mm+' mm':'N/A'],
            ['Ear Size',data.ear_size||'N/A'],
            ['Ear Prominence',data.ear_prominence||'N/A'],
            ['Ear Shape',data.ear_shape||'N/A'],
            ['Ear Position',data.ear_position||'N/A'],
        ];
        det.innerHTML=rows.map(function(r){return '<div class="metric-row"><span class="metric-label">'+r[0]+'</span><span class="metric-val">'+r[1]+'</span></div>';}).join('');
    });
</script>
</body>
</html>"""

with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html)

print('ear.html written OK to', os.path.abspath(html_path))
