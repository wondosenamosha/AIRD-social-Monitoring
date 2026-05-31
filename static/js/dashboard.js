/* ===========================================================================
   AIRD dashboard — data load, Reddit/Community/Combined views, charts.
   =========================================================================== */
(() => {
  const ALL_EMOTIONS = ["Normal","Stress","Anxiety","Personality Disorder",
                        "Bipolar","Depression","Suicidal"];
  const COLORS = {Normal:"#0c3547",Stress:"#10656d",Anxiety:"#598f91",
    "Personality Disorder":"#93b071",Bipolar:"#ede2cc",Depression:"#edae93",Suicidal:"#dd6670"};

  const state = {reddit:null, community:null, source:"reddit",
                 active:new Set(ALL_EMOTIONS)};
  const charts = {};
  const fmt = n => (n>=1000 ? (n/1000).toFixed(n>=10000?0:1)+"k" : ""+n);
  const $ = s => document.querySelector(s);

  // ---------- data merge ----------------------------------------------------
  function distFromCounts(counts){
    const total = Object.values(counts).reduce((a,b)=>a+b,0) || 1;
    return ALL_EMOTIONS.map(e=>({emotion:e,count:counts[e]||0,
      pct:+(100*(counts[e]||0)/total).toFixed(1),color:COLORS[e],emoji:EMOJI[e]}))
      .sort((a,b)=>b.count-a.count);
  }
  function kpisFromDist(dist){
    const total = dist.reduce((a,d)=>a+d.count,0);
    const risk = {Normal:0,Stress:1,Anxiety:2,"Personality Disorder":2,Bipolar:3,Depression:3,Suicidal:4};
    let hr=0, ar=0, st=0;
    dist.forEach(d=>{const r=risk[d.emotion]; if(r>=3)hr+=d.count; if(r>=2)ar+=d.count; if(r===0)st+=d.count;});
    return {total, high_risk:hr,
      high_risk_pct:total?+(100*hr/total).toFixed(1):0,
      at_risk_pct:total?+(100*ar/total).toFixed(1):0,
      stable_pct:total?+(100*st/total).toFixed(1):0};
  }
  function mergeCounts(a,b){const o={}; ALL_EMOTIONS.forEach(e=>o[e]=(a[e]||0)+(b[e]||0)); return o;}
  function distToCounts(dist){const o={}; dist.forEach(d=>{o[d.emotion]=d.count;}); return o;}

  function currentView(){
    const R = state.reddit, C = state.community;
    if(state.source==="reddit") return {...R, mentions:R.top_mentions, srcLabel:"Reddit corpus"};
    if(state.source==="community"){
      return {meta:R.meta, kpis:{...C.kpis, subreddit_count:0},
        emotion_distribution:C.emotion_distribution, timeline:C.timeline,
        by_subreddit:R.by_subreddit, sources:null, topics:R.topics,
        leaderboard:R.leaderboard, mentions:C.recent, srcLabel:"Community (live)"};
    }
    // combined
    const counts = mergeCounts(distToCounts(R.emotion_distribution), distToCounts(C.emotion_distribution));
    const dist = distFromCounts(counts);
    const k = kpisFromDist(dist);
    const conf = ((R.kpis.avg_confidence*R.kpis.total)+(C.kpis.avg_confidence*C.kpis.total))/
                 Math.max(1,R.kpis.total+C.kpis.total);
    return {meta:R.meta, kpis:{...k, avg_confidence:+conf.toFixed(1), subreddit_count:R.kpis.subreddit_count},
      emotion_distribution:dist, timeline:R.timeline, by_subreddit:R.by_subreddit,
      sources:R.sources, topics:R.topics, leaderboard:R.leaderboard,
      mentions:[...C.recent, ...R.top_mentions], srcLabel:"Reddit + Community"};
  }

  // ---------- card builders --------------------------------------------------
  function gridHTML(v){
    return `
    <div class="kpis">
      ${kpi(fmt(v.kpis.total),"Total mentions")}
      ${kpi(v.kpis.stable_pct+"%","Stable")}
      ${kpi(v.kpis.at_risk_pct+"%","At-risk")}
      ${kpi(v.kpis.high_risk_pct+"%","High-risk","alert")}
      ${kpi((v.kpis.avg_confidence||0)+"%","Avg confidence")}
    </div>

    <div class="card">
      <h3>Emotion distribution <span class="pill" id="distSrc"></span></h3>
      <div class="donut-wrap">
        <div class="donut-box"><canvas id="donutEmotion"></canvas>
          <div class="donut-center"><div class="big" id="donutTotal"></div><div class="lbl">mentions</div></div>
        </div>
        <div class="legend" id="emotionLegend"></div>
      </div>
    </div>

    <div class="card col-2">
      <h3>Mentions over time <span class="pill">daily volume</span></h3>
      <canvas id="lineMentions" height="118"></canvas>
    </div>

    <div class="card">
      <h3>Risk level by subreddit</h3>
      <div class="hbars" id="riskBars"></div>
    </div>

    <div class="card col-2">
      <h3>Emotion mix by subreddit</h3>
      <canvas id="barSubreddit" height="120"></canvas>
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
      <h3>Track B leaderboard <span class="pill">published metrics</span></h3>
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
  const kpi=(v,l,cls="")=>`<div class="kpi ${cls}"><div class="v">${v}</div><div class="l">${l}</div></div>`;

  // ---------- renderers ------------------------------------------------------
  function render(){
    const v = currentView();
    $("#grid").innerHTML = gridHTML(v);
    $("#distSrc").textContent = v.srcLabel;
    renderDonut(v); renderLegend(v); renderTimeline(v); renderRiskBars(v);
    renderSubredditBar(v); renderMentions(v); renderCloud(v); renderLeaderboard(v);
    renderSources(v); renderCommunity();
  }

  function filteredDist(v){ return v.emotion_distribution.filter(d=>state.active.has(d.emotion)); }

  function renderDonut(v){
    const d = filteredDist(v).filter(x=>x.count>0);
    $("#donutTotal").textContent = fmt(d.reduce((a,x)=>a+x.count,0));
    charts.donut?.destroy();
    charts.donut = new Chart($("#donutEmotion"),{type:"doughnut",
      data:{labels:d.map(x=>x.emotion),datasets:[{data:d.map(x=>x.count),
        backgroundColor:d.map(x=>x.color),borderWidth:2,borderColor:"#fff"}]},
      options:{cutout:"68%",plugins:{legend:{display:false},
        tooltip:{callbacks:{label:c=>` ${c.label}: ${c.raw} (${d[c.dataIndex].pct}%)`}}}}});
  }
  function renderLegend(v){
    $("#emotionLegend").innerHTML = v.emotion_distribution.map(x=>`
      <div class="row" style="${state.active.has(x.emotion)?'':'opacity:.35'}">
        <span class="dot" style="background:${x.color}"></span>
        <span class="name">${x.emotion}</span>
        <span class="val">${fmt(x.count)}</span><span class="pct">${x.pct}%</span>
      </div>`).join("");
  }
  function renderTimeline(v){
    const t = v.timeline; if(!t||!t.length){$("#lineMentions").parentElement.querySelector("canvas").style.opacity=.3;return;}
    charts.line?.destroy();
    charts.line = new Chart($("#lineMentions"),{type:"line",
      data:{labels:t.map(p=>p.date),datasets:[
        {label:"Total",data:t.map(p=>p.total),borderColor:"#3b82f6",backgroundColor:"rgba(59,130,246,.12)",fill:true,tension:.35,pointRadius:0,borderWidth:2},
        {label:"At-risk",data:t.map(p=>p.at_risk),borderColor:"#fbbf24",fill:false,tension:.35,pointRadius:0,borderWidth:2},
        {label:"High-risk",data:t.map(p=>p.high_risk),borderColor:"#f87171",fill:false,tension:.35,pointRadius:0,borderWidth:2}]},
      options:{plugins:{legend:{position:"bottom",labels:{boxWidth:12,font:{size:11}}}},
        scales:{x:{ticks:{maxTicksLimit:8,font:{size:10}},grid:{display:false}},
          y:{beginAtZero:true,ticks:{font:{size:10}},grid:{color:"#f0f2f7"}}}}});
  }
  function renderRiskBars(v){
    if(!v.by_subreddit){$("#riskBars").innerHTML='<div class="empty">No subreddit data for this view.</div>';return;}
    $("#riskBars").innerHTML = v.by_subreddit.map(s=>`
      <div class="hbar"><div class="top"><span class="name">r/${s.subreddit}</span>
        <span style="color:var(--muted)">${fmt(s.total)}</span></div>
        <div class="stacked"><div class="seg-bar">
          <span style="width:${s.stable_pct}%;background:var(--normal)" title="Stable ${s.stable_pct}%"></span>
          <span style="width:${Math.max(0,s.at_risk_pct-s.high_risk_pct)}%;background:var(--stress)" title="Elevated"></span>
          <span style="width:${s.high_risk_pct}%;background:var(--suicidal)" title="High-risk ${s.high_risk_pct}%"></span>
        </div></div></div>`).join("");
  }
  function renderSubredditBar(v){
    if(!v.by_subreddit){return;}
    const emos = ALL_EMOTIONS.filter(e=>state.active.has(e));
    charts.bar?.destroy();
    charts.bar = new Chart($("#barSubreddit"),{type:"bar",
      data:{labels:v.by_subreddit.map(s=>"r/"+s.subreddit),
        datasets:emos.map(e=>({label:e,data:v.by_subreddit.map(s=>s.counts[e]||0),
          backgroundColor:COLORS[e],stack:"s",borderRadius:3}))},
      options:{plugins:{legend:{position:"bottom",labels:{boxWidth:11,font:{size:10}}}},
        scales:{x:{stacked:true,grid:{display:false},ticks:{font:{size:11}}},
          y:{stacked:true,beginAtZero:true,ticks:{font:{size:10}},grid:{color:"#f0f2f7"}}}}});
  }
  function renderMentions(v){
    $("#mentionSrc").textContent = v.srcLabel;
    const list = (v.mentions||[]).slice(0,30);
    if(!list.length){$("#mentionList").innerHTML='<div class="empty">No mentions yet.</div>';return;}
    $("#mentionList").innerHTML = list.map(m=>{
      const color = m.color||COLORS[m.emotion];
      const where = m.subreddit?`r/${m.subreddit} · ${m.source}`:`community · ${m.model||""}`;
      return `<div class="mention"><div class="ava" style="background:${color}"></div>
        <div class="body"><div class="meta">
          <span class="tag" style="background:${color}">${m.emotion}</span>
          <span>${where}</span><span>· ${m.date||timeAgo(m.ts)}</span>
          <span class="stat">· ${m.confidence}% conf${m.score!==undefined?` · ▲${m.score}`:""}</span>
        </div><div class="txt">${escapeHtml(m.text||"")}</div></div></div>`;}).join("");
  }
  function renderCloud(v){
    if(!v.topics){$("#topicCloud").innerHTML="";return;}
    const max = Math.max(...v.topics.map(t=>t.count));
    $("#topicCloud").innerHTML = v.topics.map(t=>{
      const size = 12 + Math.round(26*Math.sqrt(t.count/max));
      return `<span style="font-size:${size}px;color:${t.color}" title="${t.count} mentions · ${t.emotion}">${t.term}</span>`;
    }).join("");
  }
  function renderLeaderboard(v){
    if(!v.leaderboard){return;}
    const max = Math.max(...v.leaderboard.map(r=>r.f1_macro));
    $("#lbList").innerHTML = v.leaderboard.map((r,i)=>`
      <div><div class="r"><span class="rank">#${i+1}</span>
        <div><div class="nm">${r.model}${r.live?'<span class="live">LIVE</span>':''}</div>
          <div class="bar"><span style="width:${100*r.f1_macro/max}%"></span></div></div>
        <div class="sc">${r.f1_macro}<small>F1 · ${r.accuracy}% acc</small></div></div></div>`).join("");
  }
  function renderSources(v){
    if(!v.sources){$("#sourceBars").innerHTML='<div class="empty">Community submissions only.</div>';return;}
    const total = Object.values(v.sources).reduce((a,b)=>a+b,0)||1;
    const items=[["Reddit comments",v.sources.comment||0,"#FF4500"],["Reddit posts",v.sources.post||0,"#3b82f6"]];
    $("#sourceBars").innerHTML = items.map(([n,c,col])=>`
      <div class="hbar"><div class="top"><span class="name">${n}</span><span>${(100*c/total).toFixed(1)}%</span></div>
      <div class="track"><span class="fill" style="width:${100*c/total}%;background:${col};border-radius:6px"></span></div></div>`).join("");
  }
  function renderCommunity(){
    const C = state.community;
    $("#commCount").textContent = C.kpis.total ? `${C.kpis.total} live` : "live";
    const feed = (C.recent||[]).slice(0,8);
    if(!feed.length){$("#commFeed").innerHTML='<div class="empty">No community submissions yet.<br>Open the Emotion Partner to add the first one →</div>';return;}
    $("#commFeed").innerHTML = `<div class="mentions" style="max-height:300px">`+feed.map(m=>`
      <div class="mention"><div class="ava" style="background:${m.color}"></div>
      <div class="body"><div class="meta"><span class="tag" style="background:${m.color}">${m.emotion}</span>
        <span>${m.model||""}</span><span class="stat">· ${m.confidence}% · ${timeAgo(m.ts)}</span></div>
        <div class="txt">${escapeHtml(m.text||"")}</div></div></div>`).join("")+`</div>`;
  }

  // ---------- chips + toggles ------------------------------------------------
  function renderChips(){
    $("#emotionChips").innerHTML = ALL_EMOTIONS.map(e=>`
      <span class="chip ${state.active.has(e)?'':'off'}" data-emo="${e}">
        <span class="dot" style="background:${COLORS[e]}"></span>${e}</span>`).join("");
    $("#emotionChips").querySelectorAll(".chip").forEach(c=>c.onclick=()=>{
      const e=c.dataset.emo;
      state.active.has(e)?state.active.delete(e):state.active.add(e);
      if(state.active.size===0) state.active.add(e);
      renderChips(); render();
    });
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
    const id=a.dataset.scroll; (id==="top"?$("#top"):document.getElementById(id))?.scrollIntoView({behavior:"smooth"});
  });

  // expose for partner.js to push live updates
  window.AIRD = window.AIRD || {};
  window.AIRD.onNewSubmission = (community)=>{ state.community = community; render(); };

  // live: poll community aggregates so the dashboard stays current across
  // sessions. Re-render only when the count changes (avoids chart flicker);
  // on the Reddit view just refresh the pulse card in place.
  async function pollCommunity(){
    try{
      const r = await fetch("/api/community"); if(!r.ok) return;
      const c = await r.json();
      const changed = !state.community || c.kpis.total !== state.community.kpis.total;
      state.community = c;
      if(!changed) return;
      if(state.source === "reddit") renderCommunity(); else render();
    }catch(e){ /* transient network errors are fine */ }
  }
  setInterval(pollCommunity, 20000);

  load();
})();
