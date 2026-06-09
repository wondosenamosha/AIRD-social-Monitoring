/* ===========================================================================
   AIRD dashboard — data load + Reddit/Community/Combined views.
   Signature visuals are hand-built with D3 (arc donut, stacked-area timeline,
   stacked subreddit bars); lists/bars are semantic HTML/CSS.
   =========================================================================== */
(() => {
  const ALL_EMOTIONS = ["Normal","Stress","Anxiety","Personality Disorder",
                        "Bipolar","Depression","Suicidal"];
  const COLORS = {Normal:"#10b981",Stress:"#f59e0b",Anxiety:"#3b82f6",
    "Personality Disorder":"#8b5cf6",Bipolar:"#ec4899",Depression:"#647488",Suicidal:"#ef4444"};
  const EMOJI = {Normal:"🙂",Stress:"😟",Anxiety:"😰","Personality Disorder":"🎭",
    Bipolar:"🎢",Depression:"😔",Suicidal:"🆘"};
  const hexRgba=(hex,a)=>{const m=hex.replace("#","");const n=parseInt(m.length===3?m.split("").map(c=>c+c).join(""):m,16);return `rgba(${(n>>16)&255},${(n>>8)&255},${n&255},${a})`;};
  const KPI_DEF=[
    {label:"Total mentions",ico:`<path d="M18 20V10M12 20V4M6 20v-6"/>`,ibg:"var(--accent-soft)",ic:"var(--accent)",cls:""},
    {label:"Stable",ico:`<path d="M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20z"/><polyline points="9 12 11 14 15 10"/>`,ibg:"#dcfce7",ic:"var(--normal)",cls:"good"},
    {label:"At-risk",ico:`<path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>`,ibg:"#fef3c7",ic:"var(--stress)",cls:"warn"},
    {label:"High-risk",ico:`<path d="M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20z"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>`,ibg:"#fee2e2",ic:"var(--suicidal)",cls:"alert"},
    {label:"Avg confidence",ico:`<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>`,ibg:"var(--accent-soft)",ic:"var(--accent)",cls:""},
  ];
  const LB_COLORS={"Stacking Ensemble":"var(--accent)","MentalBERT":"#0891b2","Mental-RoBERTa":"#7c3aed","LightGBM":"#d97706","Logistic Regression":"#16a34a"};
  const MEDAL_STYLE=["background:#fef3c7;color:#b45309","background:#f1f5f9;color:#64748b","background:#fdf4ec;color:#c2773a"];

  const state = {reddit:null, community:null, source:"reddit", active:new Set(ALL_EMOTIONS)};
  const fmt = n => (n>=1000 ? (n/1000).toFixed(n>=10000?0:1)+"k" : ""+n);
  const $ = s => document.querySelector(s);

  // shared d3 tooltip
  const TT = document.createElement("div"); TT.className="tt"; document.body.appendChild(TT);
  const tipShow=(html,e)=>{TT.innerHTML=html;TT.style.opacity=1;tipMove(e);};
  const tipMove=e=>{TT.style.left=(e.clientX+14)+"px";TT.style.top=(e.clientY+14)+"px";};
  const tipHide=()=>{TT.style.opacity=0;};

  // ---------- data merge ----------------------------------------------------
  function distFromCounts(counts){
    const total = Object.values(counts).reduce((a,b)=>a+b,0) || 1;
    return ALL_EMOTIONS.map(e=>({emotion:e,count:counts[e]||0,
      pct:+(100*(counts[e]||0)/total).toFixed(1),color:COLORS[e],emoji:EMOJI[e]}))
      .sort((a,b)=>b.count-a.count);
  }
  function kpisFromDist(dist){
    const total = dist.reduce((a,d)=>a+d.count,0);
    const risk = {Normal:0,Stress:1,Anxiety:2,"Personality Disorder":2,Depression:2,Bipolar:3,Suicidal:4};
    let hr=0, ar=0, st=0;
    dist.forEach(d=>{const r=risk[d.emotion]; if(r>=4)hr+=d.count; else if(r===3)ar+=d.count; if(r<=1)st+=d.count;});
    return {total, high_risk:hr,
      high_risk_pct:total?+(100*hr/total).toFixed(1):0,
      at_risk_pct:total?+(100*ar/total).toFixed(1):0,
      stable_pct:total?+(100*st/total).toFixed(1):0};
  }
  function mergeCounts(a,b){const o={}; ALL_EMOTIONS.forEach(e=>o[e]=(a[e]||0)+(b[e]||0)); return o;}
  function distToCounts(dist){const o={}; dist.forEach(d=>{o[d.emotion]=d.count;}); return o;}

  function filteredConfAvg(buckets){
    const mp={"70-85%":77.5,"85-100%":92.5};
    const hi=(buckets||[]).filter(b=>mp[b.range]);
    const n=hi.reduce((s,b)=>s+b.count,0);
    return n?+(hi.reduce((s,b)=>s+mp[b.range]*b.count,0)/n).toFixed(1):0;
  }
  function currentView(){
    const R = state.reddit, C = state.community;
    if(state.source==="reddit"){
      const k = kpisFromDist(R.emotion_distribution);
      const conf = filteredConfAvg(R.confidence_buckets) || R.kpis.avg_confidence;
      return {...R, kpis:{...k, avg_confidence:conf, subreddit_count:R.kpis.subreddit_count},
        mentions:R.top_mentions, srcLabel:"Reddit corpus"};
    }
    if(state.source==="community"){
      const k = kpisFromDist(C.emotion_distribution);
      const conf = filteredConfAvg(C.confidence_buckets) || C.kpis.avg_confidence;
      return {meta:R.meta, kpis:{...k, avg_confidence:conf, subreddit_count:0},
        emotion_distribution:C.emotion_distribution, timeline:C.timeline,
        by_subreddit:R.by_subreddit, sources:null, topics:R.topics,
        leaderboard:R.leaderboard, mentions:C.recent, srcLabel:"Community (live)"};
    }
    const counts = mergeCounts(distToCounts(R.emotion_distribution), distToCounts(C.emotion_distribution));
    const dist = distFromCounts(counts);
    const k = kpisFromDist(dist);
    const rConf = filteredConfAvg(R.confidence_buckets) || R.kpis.avg_confidence;
    const cConf = filteredConfAvg(C.confidence_buckets) || C.kpis.avg_confidence;
    const conf = ((rConf*R.kpis.total)+(cConf*C.kpis.total))/Math.max(1,R.kpis.total+C.kpis.total);
    return {meta:R.meta, kpis:{...k, avg_confidence:+conf.toFixed(1), subreddit_count:R.kpis.subreddit_count},
      emotion_distribution:dist, timeline:R.timeline, by_subreddit:R.by_subreddit,
      sources:R.sources, topics:R.topics, leaderboard:R.leaderboard,
      mentions:[...C.recent, ...R.top_mentions], srcLabel:"Reddit + Community"};
  }

  // ---------- grid scaffold --------------------------------------------------
  function kpi(val,def,extra){return `<div class="kpi ${def.cls}" style="--kpi-ac:${def.ic}"><div class="ico" style="background:${def.ibg}"><svg viewBox="0 0 24 24" fill="none" stroke="${def.ic}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">${def.ico}</svg></div><div class="v">${val}</div>${extra||""}<div class="l">${def.label}</div></div>`;}
  function confKpi(c){
    const ic=c>=75?"var(--normal)":c>=55?"var(--stress)":"var(--suicidal)";
    const ibg=c>=75?"#dcfce7":c>=55?"#fef3c7":"#fee2e2";
    const tier=c>=75?"HIGH":c>=55?"MED":"LOW";
    const def={...KPI_DEF[4],ic,ibg,cls:c>=75?"good":c>=55?"warn":"alert"};
    const badge=`<span class="conf-tier" style="color:${ic}">${tier}</span>`;
    return kpi(c+"%",def,badge);
  }
  function gridHTML(v){
    return `
    <div class="kpis">
      ${kpi(fmt(v.kpis.total),KPI_DEF[0])}
      ${kpi(v.kpis.stable_pct+"%",KPI_DEF[1])}
      ${kpi(v.kpis.at_risk_pct+"%",KPI_DEF[2])}
      ${kpi(v.kpis.high_risk_pct+"%",KPI_DEF[3])}
      ${confKpi(v.kpis.avg_confidence||0)}
    </div>
    <div class="card">
      <h3>Emotion distribution <span class="pill" id="distSrc"></span></h3>
      <div class="donut-wrap">
        <div class="donut-box"><div id="donutEmotion" class="chart"></div>
          <div class="donut-center"><div class="big" id="donutTotal"></div><div class="lbl" id="donutTopLabel">mentions</div></div>
        </div>
        <div class="legend" id="emotionLegend"></div>
      </div>
    </div>
    <div class="card col-2">
      <h3>Mentions over time <span class="pill">stacked daily volume</span></h3>
      <div id="lineMentions" class="chart"></div>
    </div>
    <div class="card">
      <h3>Risk level by subreddit</h3>
      <div class="hbars" id="riskBars"></div>
    </div>
    <div class="card col-2">
      <h3>Emotion mix by subreddit</h3>
      <div id="barSubreddit" class="chart"></div>
    </div>
    <div class="card" id="mentions">
      <h3>Top mentions <span class="pill" id="mentionSrc"></span></h3>
      <div class="mentions" id="mentionList"></div>
    </div>
    <div class="card col-2">
      <h3>Topic cloud <span class="pill">colored by dominant emotion</span></h3>
      <div class="cloud" id="topicCloud"></div>
    </div>
    <div class="card" id="leaderboard">
      <h3>Model leaderboard <span class="pill">held-out test metrics</span></h3>
      <div class="lb" id="lbList"></div>
    </div>
    <div class="card">
      <h3>Sources</h3>
      <div class="hbars" id="sourceBars"></div>
    </div>
    <div class="card" id="community">
      <h3>Community pulse <span class="pill" id="commCount">live</span></h3>
      <div id="commFeed"></div>
    </div>`;
  }

  // ---------- render ---------------------------------------------------------
  function render(){
    const v = currentView();
    $("#grid").innerHTML = gridHTML(v);
    $("#distSrc").textContent = v.srcLabel;
    drawDonut(v); renderLegend(v); drawTimeline(v); renderRiskBars(v);
    drawSubredditBar(v); renderMentions(v); renderCloud(v); renderLeaderboard(v);
    renderSources(v); renderCommunity();
  }
  const filteredDist = v => v.emotion_distribution.filter(d=>state.active.has(d.emotion));

  // ---------- D3: donut ------------------------------------------------------
  function drawDonut(v){
    const el = $("#donutEmotion"); el.innerHTML="";
    const data = filteredDist(v).filter(x=>x.count>0);
    $("#donutTotal").textContent = fmt(data.reduce((a,x)=>a+x.count,0));
    const top0=data[0]; if(top0){const lbl=document.getElementById("donutTopLabel");if(lbl){lbl.textContent=top0.emotion;lbl.style.color=COLORS[top0.emotion];}}
    const W=158,H=158,R=Math.min(W,H)/2;
    const svg=d3.select(el).append("svg").attr("class","chart").attr("width",W).attr("height",H)
      .append("g").attr("transform",`translate(${W/2},${H/2})`);
    const pie=d3.pie().sort(null).value(d=>d.count).padAngle(.018);
    const arc=d3.arc().innerRadius(R*0.64).outerRadius(R-2).cornerRadius(3);
    svg.selectAll("path").data(pie(data)).join("path")
      .attr("d",arc).attr("fill",d=>COLORS[d.data.emotion]||d.data.color).attr("stroke","#fff").attr("stroke-width",2)
      .style("cursor","default")
      .on("mousemove",(e,d)=>tipShow(`<b>${d.data.emotion}</b> · ${fmt(d.data.count)} (${d.data.pct}%)`,e))
      .on("mouseleave",tipHide)
      .transition().duration(450).attrTween("d",function(d){
        const i=d3.interpolate({startAngle:d.startAngle,endAngle:d.startAngle},d);
        return t=>arc(i(t));});
  }
  function renderLegend(v){
    $("#emotionLegend").innerHTML = v.emotion_distribution.map(x=>`
      <div class="row" data-emo="${x.emotion}" style="${state.active.has(x.emotion)?'':'opacity:.35'}">
        <span class="dot" style="background:${COLORS[x.emotion]||x.color}"></span>
        <span class="name">${x.emotion}</span>
        <span class="val">${fmt(x.count)}</span><span class="pct">${x.pct}%</span></div>`).join("");
    $("#emotionLegend").querySelectorAll(".row").forEach(r=>r.onclick=()=>toggleEmotion(r.dataset.emo));
  }

  // ---------- D3: stacked-area timeline -------------------------------------
  function drawTimeline(v){
    const el=$("#lineMentions"); el.innerHTML="";
    const t=v.timeline||[];
    if(t.length<2){el.innerHTML='<div class="empty">Not enough days to chart a trend.</div>';return;}
    const emos=ALL_EMOTIONS.filter(e=>state.active.has(e));
    const W=el.clientWidth||640, H=170, m={t:8,r:10,b:22,l:34};
    const parse=d3.timeParse("%Y-%m-%d");
    const rows=t.map(p=>{const o={date:parse(p.date)}; emos.forEach(e=>o[e]=(p.counts&&p.counts[e])||0); return o;});
    const x=d3.scaleTime().domain(d3.extent(rows,d=>d.date)).range([m.l,W-m.r]);
    const stack=d3.stack().keys(emos)(rows);
    const yMax=d3.max(stack[stack.length-1],d=>d[1])||1;
    const y=d3.scaleLinear().domain([0,yMax]).nice().range([H-m.b,m.t]);
    const svg=d3.select(el).append("svg").attr("class","chart").attr("width","100%").attr("height",H)
      .attr("viewBox",`0 0 ${W} ${H}`);
    // y grid
    svg.append("g").attr("class","grid").selectAll("line").data(y.ticks(4)).join("line")
      .attr("x1",m.l).attr("x2",W-m.r).attr("y1",y).attr("y2",y);
    const area=d3.area().x(d=>x(d.data.date)).y0(d=>y(d[0])).y1(d=>y(d[1])).curve(d3.curveMonotoneX);
    svg.selectAll("path.layer").data(stack).join("path").attr("class","layer")
      .attr("fill",d=>COLORS[d.key]).attr("fill-opacity",.76).attr("d",area)
      .on("mousemove",(e,d)=>tipShow(`<b>${d.key}</b>`,e)).on("mouseleave",tipHide);
    // axes
    svg.append("g").attr("class","axis").attr("transform",`translate(0,${H-m.b})`)
      .call(d3.axisBottom(x).ticks(Math.min(7,rows.length)).tickFormat(d3.timeFormat("%b %d")).tickSizeOuter(0))
      .selectAll("text").style("font-size","10px");
    svg.append("g").attr("class","axis").attr("transform",`translate(${m.l},0)`)
      .call(d3.axisLeft(y).ticks(4).tickSize(0)).call(g=>g.select(".domain").remove())
      .selectAll("text").style("font-size","10px");
  }

  // ---------- D3: stacked subreddit bars ------------------------------------
  function drawSubredditBar(v){
    const el=$("#barSubreddit"); el.innerHTML="";
    if(!v.by_subreddit||!v.by_subreddit.length){el.innerHTML='<div class="empty">No subreddit data for this view.</div>';return;}
    const emos=ALL_EMOTIONS.filter(e=>state.active.has(e));
    const subs=v.by_subreddit;
    const W=el.clientWidth||640, H=190, m={t:8,r:10,b:26,l:34};
    const x=d3.scaleBand().domain(subs.map(s=>"r/"+s.subreddit)).range([m.l,W-m.r]).padding(.34);
    const rows=subs.map(s=>{const o={k:"r/"+s.subreddit}; emos.forEach(e=>o[e]=(s.counts&&s.counts[e])||0); return o;});
    const stack=d3.stack().keys(emos)(rows);
    const yMax=d3.max(rows,r=>emos.reduce((a,e)=>a+r[e],0))||1;
    const y=d3.scaleLinear().domain([0,yMax]).nice().range([H-m.b,m.t]);
    const svg=d3.select(el).append("svg").attr("class","chart").attr("width","100%").attr("height",H)
      .attr("viewBox",`0 0 ${W} ${H}`);
    svg.append("g").attr("class","grid").selectAll("line").data(y.ticks(4)).join("line")
      .attr("x1",m.l).attr("x2",W-m.r).attr("y1",y).attr("y2",y);
    svg.selectAll("g.layer").data(stack).join("g").attr("class","layer").attr("fill",d=>COLORS[d.key])
      .selectAll("rect").data(d=>d.map(p=>({...p,key:d.key}))).join("rect")
        .attr("x",d=>x(d.data.k)).attr("width",x.bandwidth())
        .attr("y",d=>y(d[1])).attr("height",d=>Math.max(0,y(d[0])-y(d[1]))).attr("rx",2)
        .on("mousemove",(e,d)=>tipShow(`<b>${d.key}</b> · ${d.data.k}: ${d[1]-d[0]}`,e)).on("mouseleave",tipHide);
    svg.append("g").attr("class","axis").attr("transform",`translate(0,${H-m.b})`)
      .call(d3.axisBottom(x).tickSizeOuter(0)).call(g=>g.select(".domain").remove())
      .selectAll("text").style("font-size","11px").style("fill","var(--ink-2)");
    svg.append("g").attr("class","axis").attr("transform",`translate(${m.l},0)`)
      .call(d3.axisLeft(y).ticks(4).tickSize(0)).call(g=>g.select(".domain").remove())
      .selectAll("text").style("font-size","10px");
  }

  // ---------- CSS-based panels ----------------------------------------------
  function renderRiskBars(v){
    if(!v.by_subreddit){$("#riskBars").innerHTML='<div class="empty">No subreddit data for this view.</div>';return;}
    $("#riskBars").innerHTML = v.by_subreddit.map(s=>`
      <div class="hbar"><div class="top"><span class="name">r/${s.subreddit}</span>
        <span>${fmt(s.total)}</span></div>
        <div class="stacked"><div class="seg-bar">
          <span style="width:${s.stable_pct}%;background:var(--normal)" title="Stable ${s.stable_pct}%"></span>
          <span style="width:${Math.max(0,s.at_risk_pct-s.high_risk_pct)}%;background:var(--stress)" title="Elevated"></span>
          <span style="width:${s.high_risk_pct}%;background:var(--suicidal)" title="Suicidal ${s.high_risk_pct}%"></span>
        </div></div></div>`).join("");
  }
  function renderMentions(v){
    $("#mentionSrc").textContent = v.srcLabel;
    const list = (v.mentions||[]).slice(0,30);
    if(!list.length){$("#mentionList").innerHTML='<div class="empty">No mentions yet.</div>';return;}
    $("#mentionList").innerHTML = list.map(m=>{
      const color = COLORS[m.emotion]||m.color;
      const highRisk = new Set(["Suicidal","Bipolar"]).has(m.emotion);
      const lowConf = highRisk ? m.confidence < 65 : m.confidence < 50;
      const where = m.subreddit?`r/${m.subreddit} · ${m.source}`:`community · ${m.model||""}`;
      const tagLabel = lowConf && highRisk ? "Uncertain" : m.emotion;
      return `<div class="mention${lowConf?' low-conf':''}"><div class="ava" style="background:${lowConf?'#cbd5e1':color}"></div>
        <div class="body"><div class="meta">
          <span class="tag" style="background:${lowConf?'#94a3b8':color}">${tagLabel}</span>
          <span>${where}</span><span>· ${m.date||timeAgo(m.ts)}</span>
          <span class="stat">· ${m.confidence}% conf${lowConf?' ⚠ low':''}${m.score!==undefined?` · ▲${m.score}`:""}</span>
        </div><div class="txt">${escapeHtml(m.text||"")}</div></div></div>`;}).join("");
  }
  function renderCloud(v){
    if(!v.topics){$("#topicCloud").innerHTML="";return;}
    const max = Math.max(...v.topics.map(t=>t.count));
    $("#topicCloud").innerHTML = v.topics.map(t=>{
      const size = 12 + Math.round(24*Math.sqrt(t.count/max));
      const c=COLORS[t.emotion]||t.color; return `<span style="font-size:${size}px;color:${c};background:${hexRgba(c,.1)}" title="${t.count} mentions · ${t.emotion}">${t.term}</span>`;
    }).join("");
  }
  function renderLeaderboard(v){
    if(!v.leaderboard){return;}
    const max = Math.max(...v.leaderboard.map(r=>r.f1_macro));
    $("#lbList").innerHTML = v.leaderboard.map((r,i)=>{
      const color=LB_COLORS[r.model]||"var(--accent)";
      const rStyle=i<3?` style="${MEDAL_STYLE[i]}"`:` style="color:var(--faint)"`;
      return `<div class="r ${i===0?'top':''}"><span class="rank"${rStyle}>${i<3?["#1","#2","#3"][i]:"#"+(i+1)}</span>
        <div><div class="nm">${r.model}${r.live?'<span class="live">LIVE</span>':''}</div>
          <div class="bar"><span style="width:${100*r.f1_macro/max}%;background:${color}"></span></div></div>
        <div class="sc">${r.f1_macro}<small>F1 · ${r.accuracy}% acc</small></div></div>`;
    }).join("");
  }
  function renderSources(v){
    if(!v.sources){$("#sourceBars").innerHTML='<div class="empty">Community submissions only.</div>';return;}
    const total = Object.values(v.sources).reduce((a,b)=>a+b,0)||1;
    const items=[["Reddit comments",v.sources.comment||0,"#ff5a1f"],["Reddit posts",v.sources.post||0,"var(--accent)"]];
    $("#sourceBars").innerHTML = items.map(([n,c,col])=>`
      <div class="hbar"><div class="top"><span class="name">${n}</span><span>${(100*c/total).toFixed(1)}%</span></div>
      <div class="track"><span class="fill" style="width:${100*c/total}%;background:${col}"></span></div></div>`).join("");
  }
  function renderCommunity(){
    const C = state.community;
    $("#commCount").textContent = C.kpis.total ? `${C.kpis.total} live` : "live";
    const feed = (C.recent||[]).slice(0,8);
    if(!feed.length){$("#commFeed").innerHTML='<div class="empty">No community submissions yet.<br>Open the Emotion Companion to add the first one →</div>';return;}
    $("#commFeed").innerHTML = `<div class="mentions" style="max-height:300px">`+feed.map(m=>{const cc=COLORS[m.emotion]||m.color;return `
      <div class="mention"><div class="ava" style="background:${cc}"></div>
      <div class="body"><div class="meta"><span class="tag" style="background:${cc}">${m.emotion}</span>
        <span>${m.model||""}</span><span class="stat">· ${m.confidence}% · ${timeAgo(m.ts)}</span></div>
        <div class="txt">${escapeHtml(m.text||"")}</div></div></div>`;}).join("")+`</div>`;
  }

  // ---------- chips ----------------------------------------------------------
  function toggleEmotion(e){
    state.active.has(e)?state.active.delete(e):state.active.add(e);
    if(state.active.size===0) state.active.add(e);
    renderChips(); render();
  }
  function renderChips(){
    $("#emotionChips").innerHTML = ALL_EMOTIONS.map(e=>`
      <span class="chip ${state.active.has(e)?'':'off'}" data-emo="${e}">
        <span class="dot" style="background:${COLORS[e]}"></span>${e}</span>`).join("");
    $("#emotionChips").querySelectorAll(".chip").forEach(c=>c.onclick=()=>toggleEmotion(c.dataset.emo));
  }

  // ---------- utils ----------------------------------------------------------
  function escapeHtml(s){return (s||"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));}
  function timeAgo(ts){if(!ts)return"";const s=(Date.now()-new Date(ts))/1000;
    if(s<60)return"just now";if(s<3600)return Math.floor(s/60)+"m ago";
    if(s<86400)return Math.floor(s/3600)+"h ago";return Math.floor(s/86400)+"d ago";}

  // ---------- boot -----------------------------------------------------------
  async function load(){
    const r = await fetch("/api/dashboard"); const j = await r.json();
    state.reddit = j.reddit; state.community = j.community;
    renderChips(); render();
  }
  $("#sourceSeg").querySelectorAll("button").forEach(b=>b.onclick=()=>{
    $("#sourceSeg").querySelectorAll("button").forEach(x=>x.classList.remove("active"));
    b.classList.add("active"); state.source=b.dataset.src; render();
  });
  document.querySelectorAll("[data-scroll]").forEach(a=>a.onclick=()=>{
    const id=a.dataset.scroll;
    document.querySelectorAll(".nav a").forEach(n=>n.classList.remove("active")); a.classList.add("active");
    (id==="top"?$("#top"):document.getElementById(id))?.scrollIntoView({behavior:"smooth",block:"start"});
  });

  window.AIRD = window.AIRD || {};
  window.AIRD.onNewSubmission = (community)=>{ state.community = community; render(); };

  async function pollCommunity(){
    try{
      const r = await fetch("/api/community"); if(!r.ok) return;
      const c = await r.json();
      const changed = !state.community || c.kpis.total !== state.community.kpis.total;
      state.community = c;
      if(!changed) return;
      if(state.source === "reddit") renderCommunity(); else render();
    }catch(e){}
  }
  setInterval(pollCommunity, 20000);

  // redraw D3 charts on resize (debounced)
  let rt; window.addEventListener("resize",()=>{clearTimeout(rt);rt=setTimeout(render,200);});

  load();
})();
