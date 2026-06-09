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
    drawDonut(v); renderLegend(v); drawTimeline(v); drawRiskDonuts(v);
    drawSubredditBar(v); renderMentions(v); renderCloud(v); renderLeaderboard(v);
    drawSources(v); renderCommunity();
  }
  const filteredDist = v => v.emotion_distribution.filter(d=>state.active.has(d.emotion));

  // ---------- D3: donut ------------------------------------------------------
  function drawDonut(v){
    const el = $("#donutEmotion"); el.innerHTML="";
    const data = filteredDist(v).filter(x=>x.count>0);
    $("#donutTotal").textContent = fmt(data.reduce((a,x)=>a+x.count,0));
    const top0=data[0]; if(top0){const lbl=document.getElementById("donutTopLabel");if(lbl){lbl.textContent=top0.emotion;lbl.style.color=COLORS[top0.emotion];}}
    const W=172,H=172,R=Math.min(W,H)/2;
    const svg=d3.select(el).append("svg").attr("class","chart").attr("width",W).attr("height",H)
      .append("g").attr("transform",`translate(${W/2},${H/2})`);
    const pie=d3.pie().sort(null).value(d=>d.count).padAngle(.022);
    const arc=d3.arc().innerRadius(R*0.60).outerRadius(R-2).cornerRadius(4);
    svg.selectAll("path").data(pie(data)).join("path")
      .attr("fill",d=>COLORS[d.data.emotion]||d.data.color).attr("stroke","#fff").attr("stroke-width",2)
      .style("cursor","default").attr("opacity",0)
      .on("mousemove",(e,d)=>tipShow(`<b>${d.data.emotion}</b> · ${fmt(d.data.count)} (${d.data.pct}%)`,e))
      .on("mouseleave",tipHide)
      .transition().duration(600).delay((_,i)=>i*60).ease(d3.easeCubicOut)
      .attr("opacity",1)
      .attrTween("d",function(d){
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
    const W=el.clientWidth||640, H=210, m={t:8,r:10,b:22,l:34};
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
    // clip-path reveal: wipe left → right
    const clipId="tc"+Date.now();
    svg.append("defs").append("clipPath").attr("id",clipId)
      .append("rect").attr("x",m.l).attr("y",m.t).attr("height",H-m.t-m.b).attr("width",0)
      .transition().duration(1100).ease(d3.easeCubicInOut).attr("width",W-m.l-m.r);
    svg.selectAll("path.layer").data(stack).join("path").attr("class","layer")
      .attr("fill",d=>COLORS[d.key]).attr("fill-opacity",.74).attr("d",area)
      .attr("clip-path",`url(#${clipId})`)
      .on("mousemove",(e,d)=>tipShow(`<b>${d.key}</b>`,e)).on("mouseleave",tipHide);
    // axes
    svg.append("g").attr("class","axis").attr("transform",`translate(0,${H-m.b})`)
      .call(d3.axisBottom(x).ticks(Math.min(7,rows.length)).tickFormat(d3.timeFormat("%b %d")).tickSizeOuter(0))
      .selectAll("text").style("font-size","10px");
    svg.append("g").attr("class","axis").attr("transform",`translate(${m.l},0)`)
      .call(d3.axisLeft(y).ticks(4).tickSize(0)).call(g=>g.select(".domain").remove())
      .selectAll("text").style("font-size","10px");
  }

  // ---------- D3: emotion mix horizontal bars (replaces vertical stacked) ----
  function drawSubredditBar(v){
    const el=$("#barSubreddit"); el.innerHTML="";
    if(!v.by_subreddit||!v.by_subreddit.length){el.innerHTML='<div class="empty">No subreddit data for this view.</div>';return;}
    const emos=ALL_EMOTIONS.filter(e=>state.active.has(e));
    const subs=v.by_subreddit;
    const W=el.clientWidth||560, BAR=14, GAP=20, PAD_L=110, PAD_R=52, PAD_T=6, HEADER_H=22;
    const H=PAD_T+(subs.length*(HEADER_H+emos.length*(BAR+4)+GAP));
    const maxTotal=d3.max(subs,s=>emos.reduce((a,e)=>a+(s.counts?.[e]||0),0))||1;
    const xScale=d3.scaleLinear().domain([0,maxTotal]).range([0,W-PAD_L-PAD_R]);
    const svg=d3.select(el).append("svg").attr("class","chart").attr("width","100%").attr("height",H)
      .attr("viewBox",`0 0 ${W} ${H}`);
    // shimmer gradient def
    const shimId="shim"+Date.now();
    const defs=svg.append("defs");
    const lg=defs.append("linearGradient").attr("id",shimId).attr("x1","0%").attr("x2","100%");
    lg.append("stop").attr("offset","0%").attr("stop-color","white").attr("stop-opacity",0);
    lg.append("stop").attr("offset","50%").attr("stop-color","white").attr("stop-opacity",.22);
    lg.append("stop").attr("offset","100%").attr("stop-color","white").attr("stop-opacity",0);

    subs.forEach((s,si)=>{
      const groupY=PAD_T+si*(HEADER_H+emos.length*(BAR+4)+GAP);
      // subreddit header row — above the bars, no overlap with emotion labels
      svg.append("text").attr("x",2).attr("y",groupY+14)
        .attr("font-size","11.5").attr("font-weight","700")
        .attr("fill","#334155").text("r/"+s.subreddit);
      const rowTotal=emos.reduce((a,e)=>a+(s.counts?.[e]||0),0);
      svg.append("text").attr("x",W-PAD_R).attr("y",groupY+14)
        .attr("font-size","11").attr("font-weight","600")
        .attr("fill","#94a3b8").attr("text-anchor","end").text(fmt(rowTotal));

      emos.forEach((e,ei)=>{
        const count=s.counts?.[e]||0;
        const barW=xScale(count);
        const y=groupY+HEADER_H+ei*(BAR+4);
        const color=COLORS[e];
        // track
        svg.append("rect").attr("x",PAD_L).attr("y",y).attr("width",W-PAD_L-PAD_R)
          .attr("height",BAR).attr("rx",BAR/2).attr("fill","#eaecf2");
        // fill — animated from 0
        const fill=svg.append("rect").attr("x",PAD_L).attr("y",y).attr("width",0)
          .attr("height",BAR).attr("rx",BAR/2).attr("fill",color)
          .on("mousemove",ev=>tipShow(`<b>${e}</b> · r/${s.subreddit}: ${fmt(count)}`,ev))
          .on("mouseleave",tipHide);
        fill.transition().duration(700).delay(si*120+ei*45).ease(d3.easeCubicOut)
          .attr("width",Math.max(barW,count>0?BAR:0));
        // shimmer overlay on fill (repeating)
        if(count>0){
          const shim=svg.append("rect").attr("x",PAD_L).attr("y",y)
            .attr("width",0).attr("height",BAR).attr("rx",BAR/2)
            .attr("fill",`url(#${shimId})`).attr("pointer-events","none");
          shim.transition().duration(700).delay(si*120+ei*45)
            .attr("width",Math.max(barW,BAR))
            .on("end",function(){d3.select(this).attr("class","bar-shim");});
        }
        // emotion label
        svg.append("text").attr("x",PAD_L-6).attr("y",y+BAR/2).attr("dy","0.35em")
          .attr("text-anchor","end").attr("font-size","10.5").attr("fill","var(--muted)")
          .attr("font-weight","500").text(e==="Personality Disorder"?"P. Disorder":e);
      });
    });
  }

  // ---------- D3: risk by subreddit — animated mini donuts -----------------
  function drawRiskDonuts(v){
    const el=$("#riskBars"); el.innerHTML="";
    if(!v.by_subreddit){el.innerHTML='<div class="empty">No subreddit data for this view.</div>';return;}
    const subs=v.by_subreddit;
    const W=el.clientWidth||320, COLS=subs.length, CW=Math.floor(W/COLS);
    const R=Math.min(CW/2-14,54), ri=R*0.57;
    const cy=R+8, H=R*2+56;
    const arcFn=d3.arc().innerRadius(ri).outerRadius(R).cornerRadius(3);
    const pieFn=d3.pie().sort(null).value(d=>d.val).padAngle(0.03);
    const svg=d3.select(el).append("svg").attr("width","100%").attr("height",H)
      .attr("viewBox",`0 0 ${W} ${H}`);
    subs.forEach((s,si)=>{
      const ox=si*CW+CW/2;
      const elevated=Math.max(0,100-s.stable_pct-s.at_risk_pct);
      const atRisk=Math.max(0,s.at_risk_pct-s.high_risk_pct);
      const data=[
        {label:"Stable",val:s.stable_pct,color:"#10b981"},
        {label:"Moderate",val:elevated,color:"#94a3b8"},
        {label:"At-risk (Bipolar)",val:atRisk,color:"#ec4899"},
        {label:"High-risk (Suicidal)",val:s.high_risk_pct,color:"#ef4444"},
      ].filter(d=>d.val>0.1);
      const g=svg.append("g").attr("transform",`translate(${ox},${cy})`);
      g.selectAll("path").data(pieFn(data)).join("path")
        .attr("fill",d=>d.data.color).attr("stroke","#fff").attr("stroke-width",1.5)
        .attr("opacity",0)
        .on("mousemove",(e,d)=>tipShow(`<b>${d.data.label}</b>: ${d.data.val.toFixed(1)}%`,e))
        .on("mouseleave",tipHide)
        .transition().duration(700).delay(si*130)
        .attr("opacity",1)
        .attrTween("d",function(d){
          const i=d3.interpolate({startAngle:d.startAngle,endAngle:d.startAngle},d);
          return t=>arcFn(i(t));
        });
      g.append("text").attr("text-anchor","middle").attr("dy","-0.15em")
        .attr("font-size","13").attr("font-weight","800").attr("fill","#1e293b")
        .text(fmt(s.total));
      g.append("text").attr("text-anchor","middle").attr("dy","1em")
        .attr("font-size","9").attr("font-weight","500").attr("fill","#94a3b8")
        .text("mentions");
      svg.append("text").attr("x",ox).attr("y",cy+R+18)
        .attr("text-anchor","middle").attr("font-size","11").attr("font-weight","700")
        .attr("fill","#334155").text("r/"+s.subreddit);
      svg.append("text").attr("x",ox).attr("y",cy+R+33)
        .attr("text-anchor","middle").attr("font-size","10").attr("font-weight","600")
        .attr("fill",s.high_risk_pct>10?"#ef4444":"#94a3b8")
        .text(`${s.high_risk_pct}% crisis`);
    });
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
      const f1disp=(r.f1_macro>1?r.f1_macro/100:r.f1_macro).toFixed(3);
      return `<div class="r ${i===0?'top':''}"><span class="rank"${rStyle}>${i<3?["#1","#2","#3"][i]:"#"+(i+1)}</span>
        <div><div class="nm">${r.model}${r.live?'<span class="live">LIVE</span>':''}</div>
          <div class="bar"><span style="width:${100*r.f1_macro/max}%;background:${color}"></span></div></div>
        <div class="sc">${f1disp}<small>F1 · ${r.accuracy}% acc</small></div></div>`;
    }).join("");
  }
  function drawSources(v){
    const el=$("#sourceBars"); el.innerHTML="";
    if(!v.sources){el.innerHTML='<div class="empty">Community submissions only.</div>';return;}
    const total=Object.values(v.sources).reduce((a,b)=>a+b,0)||1;
    const items=[
      {label:"Comments",count:v.sources.comment||0,color:"#ff5a1f"},
      {label:"Posts",count:v.sources.post||0,color:"#0ea5e9"},
    ];
    const W=el.clientWidth||280, H=162;
    const cx=W/2, cy=H-20;
    const R=Math.min(cx-20,cy-6), ri=R*0.57;
    const arcFn=d3.arc().innerRadius(ri).outerRadius(R).cornerRadius(5);
    let startA=-Math.PI/2;
    const segments=items.map(d=>{
      const span=(d.count/total)*Math.PI;
      const seg={...d,startAngle:startA,endAngle:startA+span};
      startA+=span; return seg;
    });
    const svg=d3.select(el).append("svg").attr("width","100%").attr("height",H)
      .attr("viewBox",`0 0 ${W} ${H}`);
    const g=svg.append("g").attr("transform",`translate(${cx},${cy})`);
    g.append("path").attr("d",arcFn({startAngle:-Math.PI/2,endAngle:Math.PI/2})).attr("fill","#f1f5f9");
    segments.forEach((seg,i)=>{
      g.append("path").attr("fill",seg.color).attr("opacity",0)
        .on("mousemove",e=>tipShow(`<b>${seg.label}</b>: ${fmt(seg.count)} · ${(seg.count/total*100).toFixed(1)}%`,e))
        .on("mouseleave",tipHide)
        .transition().duration(850).delay(i*90)
        .attr("opacity",1)
        .attrTween("d",()=>{
          const interp=d3.interpolate(
            {startAngle:seg.startAngle,endAngle:seg.startAngle},
            {startAngle:seg.startAngle,endAngle:seg.endAngle});
          return t=>arcFn(interp(t));
        });
    });
    g.append("text").attr("text-anchor","middle").attr("dy","-0.3em")
      .attr("font-size","18").attr("font-weight","800").attr("fill","#1e293b").text(fmt(total));
    g.append("text").attr("text-anchor","middle").attr("dy","0.9em")
      .attr("font-size","9.5").attr("fill","#94a3b8").text("total items");
    const legY=cy+18;
    items.forEach((d,i)=>{
      const lx=i===0?cx-68:cx+8;
      svg.append("circle").attr("cx",lx).attr("cy",legY).attr("r",5).attr("fill",d.color);
      svg.append("text").attr("x",lx+10).attr("y",legY).attr("dy","0.35em")
        .attr("font-size","10.5").attr("fill","#334155")
        .text(`${d.label} · ${(d.count/total*100).toFixed(0)}%`);
    });
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

  // ---------- sidebar toggle ---------------------------------------------------
  (function(){
    const sidebar = document.querySelector(".sidebar");
    const btn = $("#sidebarToggle");
    if(!sidebar||!btn) return;
    const COLLAPSED_KEY = "aird_sidebar_collapsed";
    if(localStorage.getItem(COLLAPSED_KEY)==="1") sidebar.classList.add("collapsed");
    btn.onclick = () => {
      const isNowCollapsed = sidebar.classList.toggle("collapsed");
      localStorage.setItem(COLLAPSED_KEY, isNowCollapsed ? "1" : "0");
    };
  })();

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
