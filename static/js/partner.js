/* ===========================================================================
   AIRD Emotion Partner — chat -> 4-screen analysis report.
   Screen 0: chat   1: result hero   2: distribution donut
   3: probability breakdown   4: radar + keywords + tip
   =========================================================================== */
(() => {
  const COLORS = {Normal:"#10b981",Stress:"#f59e0b",Anxiety:"#3b82f6",
    "Personality Disorder":"#8b5cf6",Bipolar:"#ec4899",Depression:"#647488",Suicidal:"#ef4444"};
  const EMOJI = {Normal:"🙂",Stress:"😟",Anxiety:"😰","Personality Disorder":"🎭",
    Bipolar:"🎢",Depression:"😔",Suicidal:"🆘"};
  const DESC = {
    Normal:"Your emotional state appears generally stable with mild variations.",
    Stress:"Signs of stress and pressure are showing up in what you wrote.",
    Anxiety:"Markers of worry and anxious tension are present.",
    "Personality Disorder":"Complex, shifting emotional patterns were detected.",
    Bipolar:"Patterns consistent with strong mood shifts were detected.",
    Depression:"Low-mood and withdrawal signals are present in your words.",
    Suicidal:"Language of serious distress was detected. Please reach out for support."};
  const RADAR_ORDER = ["Normal","Stress","Anxiety","Suicidal","Depression","Bipolar","Personality Disorder"];
  const RISK_COLOR = ["#10b981","#f59e0b","#3b82f6","#7c3aed","#ef4444"];

  const $ = s => document.querySelector(s);
  const charts = {};
  let step = 0, lastAnalysis = null, busy = false;

  // ---------- open / close ---------------------------------------------------
  const overlay=$("#overlay"), partner=$("#partner");
  function open(){overlay.classList.add("open");partner.classList.add("open");partner.setAttribute("aria-hidden","false");$("#feelInput")?.focus();}
  function close(){overlay.classList.remove("open");partner.classList.remove("open");partner.setAttribute("aria-hidden","true");}
  document.querySelectorAll("[data-open-partner]").forEach(b=>b.onclick=open);
  $("#closePartner").onclick=close; overlay.onclick=close;
  document.addEventListener("keydown",e=>{          // ESC closes the dialog
    if(e.key==="Escape" && partner.classList.contains("open")) close();
  });
  // Deep-link: opening /#partner reveals the Emotion Partner straight away.
  if(location.hash==="#partner") setTimeout(open,150);

  // ---------- chat input -----------------------------------------------------
  const input=$("#feelInput");
  input.addEventListener("input",()=>{input.style.height="auto";input.style.height=Math.min(120,input.scrollHeight)+"px";});
  input.addEventListener("keydown",e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();send();}});
  $("#sendBtn").onclick=send;

  function bubble(cls,html){const d=document.createElement("div");d.className="bubble "+cls;d.innerHTML=html;$("#chat").appendChild(d);$("#chatStage").scrollTop=1e9;return d;}

  const isRedditUrl = s => /reddit\.com\/.+?\/comments\/[a-z0-9]+/i.test(s)
                           || /^https?:\/\/\S+$/i.test(s);

  async function send(){
    if(busy) return;
    const raw=input.value.trim();
    if(!raw){input.focus();return;}
    const model=$("#modelSelect").value;
    const url=isRedditUrl(raw);
    busy=true; $("#sendBtn").disabled=true;
    bubble("me",escapeHtml(raw));
    input.value=""; input.style.height="auto";
    const msg=url?"Scraping the Reddit post and analyzing the author + comments"
                 :"Analyzing your emotional state";
    const wait=bubble("analyzing",`<span>${msg}</span><span class="dots"><i></i><i></i><i></i></span>`);
    const hint=setTimeout(()=>{wait.querySelector("span").textContent=
      url?"Fetching from Reddit + running the model…":"Loading the model (first run can take a moment)";},4000);

    try{
      const r=await fetch(url?"/api/analyze_url":"/api/analyze",
        {method:"POST",headers:{"Content-Type":"application/json"},
         body:JSON.stringify(url?{url:raw,model}:{text:raw,model})});
      const j=await r.json();
      clearTimeout(hint); wait.remove();
      if(!r.ok){bubble("bot",(j.error||"Something went wrong."));busy=false;$("#sendBtn").disabled=false;return;}
      lastAnalysis=j.analysis;
      if(url && j.post)
        bubble("bot",`Scraped <strong>r/${escapeHtml(j.post.subreddit)}</strong> — “${escapeHtml(j.post.title)}”. Analyzed the author and ${j.thread.analyzed} comments.`);
      showReport(j.analysis, (url&&j.post)?{post:j.post,thread:j.thread}:null);
      if(j.community && window.AIRD?.onNewSubmission) window.AIRD.onNewSubmission(j.community);
    }catch(err){
      clearTimeout(hint); wait.remove();
      bubble("bot","Network error — is the server running?");
    }
    busy=false; $("#sendBtn").disabled=false;
  }

  // ---------- report ---------------------------------------------------------
  function showReport(a, ctx){
    $("#chatStage").classList.add("hide"); $("#composer").classList.add("hide");
    $("#reportStage").classList.add("show");
    // optional Reddit thread card (link-analysis mode)
    const tc=$("#threadCard");
    if(ctx && ctx.post){
      $("#tcSource").textContent="r/"+ctx.post.subreddit+(ctx.post.author?" · u/"+ctx.post.author:"");
      $("#tcTitle").textContent=ctx.post.title;
      $("#tcMeta").textContent=`▲ ${ctx.post.score} · ${ctx.post.num_comments} comments · ${ctx.thread.analyzed} analyzed here`;
      // show the author's post body so classification is transparent
      const snip=ctx.post.snippet&&ctx.post.snippet!==ctx.post.title?ctx.post.snippet:null;
      const snipEl=document.getElementById("tcSnippet")||document.createElement("div");
      snipEl.id="tcSnippet";
      snipEl.style.cssText="margin:6px 0 4px;font-size:12px;color:#64748b;font-style:italic;line-height:1.5;border-left:3px solid #cbd5e1;padding-left:9px";
      snipEl.textContent=snip?"Author wrote: \""+snip+"\"":`(title-only post — no body text)`;
      const titleEl=$("#tcTitle"); titleEl.after(snipEl);
      const total=ctx.thread.distribution.reduce((s,d)=>s+d.count,0)||1;
      $("#tcBar").innerHTML=ctx.thread.analyzed?`<div class="seg-bar">`+ctx.thread.distribution.map(d=>
        `<span style="width:${100*d.count/total}%;background:${d.color}" title="${d.emotion}: ${d.count}"></span>`).join("")+`</div>`:"";
      $("#tcComments").innerHTML=(ctx.thread.comments||[]).map(c=>
        `<div class="tc-c"><span class="tag" style="background:${c.color}">${c.emotion}</span><span>${escapeHtml(c.text)}</span></div>`).join("");
      tc.classList.remove("hide");
    }else tc.classList.add("hide");
    // hero — when post title is too short, fall back to dominant thread mood
    const threadDist = ctx?.thread?.distribution;
    const useThreadMood = a.short_text && threadDist && threadDist.length > 0;
    const hero = useThreadMood
      ? {emotion: threadDist[0].emotion, color: threadDist[0].color,
         emoji: threadDist[0].emoji, confidence: null, risk_tier: a.risk_tier,
         risk_level: a.risk_level}
      : {emotion: a.top_emotion, color: COLORS[a.top_emotion]||a.top_color,
         emoji: EMOJI[a.top_emotion]||"•", confidence: a.confidence,
         risk_tier: a.risk_tier, risk_level: a.risk_level};
    const fc=$("#rFace"); fc.textContent=hero.emoji;
    fc.style.color=hero.color;
    $("#rEmo").textContent=hero.confidence!=null
      ? `${hero.emotion} · ${hero.confidence}%`
      : `${hero.emotion} · thread mood`;
    $("#rDesc").textContent=useThreadMood
      ? "Post title too short — showing dominant thread emotion."
      : (DESC[a.top_emotion]||"");
    const rb=$("#rRisk"); rb.textContent=`Risk: ${a.risk_tier}`;
    rb.style.background=RISK_COLOR[a.risk_level]||a.top_color;
    const existWarn=$("#shortTextWarn");
    if(existWarn) existWarn.remove();
    if(a.short_text){
      const w=document.createElement("div");
      w.id="shortTextWarn";
      w.style.cssText="margin:10px 0 0;padding:8px 12px;border-radius:8px;background:#fef9c3;border:1px solid #fde047;font-size:11.5px;color:#854d0e;line-height:1.45;text-align:center";
      w.textContent=useThreadMood
        ? "⚠ Post title too short for direct classification — emotion shown reflects the thread discussion."
        : "⚠ Text too short for reliable classification. Try adding more detail.";
      rb.after(w);
    }
    // donut + legend
    const ranked=a.ranked.filter(x=>x.pct>0);
    charts.pdonut?.destroy();
    charts.pdonut=new Chart($("#partnerDonut"),{type:"doughnut",
      data:{labels:ranked.map(x=>x.emotion),datasets:[{data:ranked.map(x=>x.pct),
        backgroundColor:ranked.map(x=>COLORS[x.emotion]),borderWidth:2,borderColor:"#fff"}]},
      options:{cutout:"70%",plugins:{legend:{display:false},
        tooltip:{callbacks:{label:c=>` ${c.label}: ${c.raw}%`}}}}});
    $("#partnerLegend").innerHTML=a.ranked.map(x=>`
      <div class="row"><span class="dot" style="background:${COLORS[x.emotion]}"></span>
        <span class="name">${x.emotion}</span>
        <span class="pct">${x.pct}%</span></div>`).join("");
    // probability bars
    $("#probBars").innerHTML=a.ranked.map(x=>`
      <div class="hbar"><div class="top"><span class="name">${x.emotion}</span>
        <span style="font-weight:700">${x.pct}%</span></div>
      <div class="track"><span class="fill" style="width:${x.pct}%;background:${COLORS[x.emotion]};border-radius:6px"></span></div></div>`).join("");
    // radar
    const vals=RADAR_ORDER.map(e=>a.probabilities[e]||0);
    charts.pradar?.destroy();
    charts.pradar=new Chart($("#partnerRadar"),{type:"radar",
      data:{labels:RADAR_ORDER.map(e=>(e==="Personality Disorder"?"Personality":e)),
        datasets:[{data:vals,borderColor:(COLORS[a.top_emotion]||a.top_color),
          backgroundColor:hexA(COLORS[a.top_emotion]||a.top_color,.18),
          pointBackgroundColor:(COLORS[a.top_emotion]||a.top_color),borderWidth:2}]},
      options:{plugins:{legend:{display:false}},
        scales:{r:{suggestedMin:0,suggestedMax:Math.max(20,Math.ceil(Math.max(...vals)/10)*10),
          ticks:{display:false,stepSize:20},pointLabels:{font:{size:10}},grid:{color:"#dfe4ee"}}}}});
    // keywords + tip
    $("#kwBox").innerHTML=(a.keywords&&a.keywords.length)
      ? a.keywords.map(k=>`<span class="k">${escapeHtml(k)}</span>`).join("")
      : `<span style="color:var(--muted);font-size:12px">No specific keywords detected</span>`;
    $("#tipText").textContent=a.tip||"";
    step=0; showStep(0); buildDots();
  }

  function buildDots(){
    $("#dots2").innerHTML=[0,1,2,3].map(i=>`<i class="${i===step?'on':''}"></i>`).join("");
  }
  function showStep(i){
    step=Math.max(0,Math.min(3,i));
    document.querySelectorAll(".step").forEach(s=>s.classList.toggle("show",+s.dataset.step===step));
    $("#prevStep").disabled=step===0; $("#nextStep").disabled=step===3;
    buildDots(); $("#reportStage").scrollTop=0;
    if(step===1) charts.pdonut?.resize();
    if(step===3) charts.pradar?.resize();
  }
  $("#prevStep").onclick=()=>showStep(step-1);
  $("#nextStep").onclick=()=>showStep(step+1);
  $("#againBtn").onclick=()=>{
    $("#reportStage").classList.remove("show");
    $("#chatStage").classList.remove("hide"); $("#composer").classList.remove("hide");
    input.focus(); $("#chatStage").scrollTop=1e9;
  };

  // ---------- utils ----------------------------------------------------------
  function escapeHtml(s){return (s||"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));}
  function hexA(hex,a){const m=hex.replace("#","");const n=parseInt(m.length===3?m.split("").map(c=>c+c).join(""):m,16);
    return `rgba(${(n>>16)&255},${(n>>8)&255},${n&255},${a})`;}
})();
