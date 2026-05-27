/* Position Planner — IIFE isolated (no globals leak). Common to US & KR. */
(function () {
  'use strict';
  const QUOTE_API = "https://g5-22-trader.blackmoneytica.workers.dev";
  const KR_DATA   = "https://kr-v25.pages.dev/data/daily.json";
  const MAE_MULT  = 2.7;   // typical underwater depth ≈ 2.7 × ATR (backtest)

  const $ = (id) => document.getElementById(id);
  const isKR = (t) => /\.(KS|KQ)$/i.test(t);

  function fmtCur(v, kr) {
    if (v == null || !isFinite(v)) return "-";
    return kr ? "₩" + Math.round(v).toLocaleString("ko-KR")
              : "$" + v.toLocaleString("en-US", { maximumFractionDigits: 2 });
  }
  function fmtPx(v, kr) {
    if (v == null || !isFinite(v)) return "-";
    return kr ? Math.round(v).toLocaleString("ko-KR") : v.toFixed(2);
  }

  async function fetchQuote(t) {
    const r = await fetch(`${QUOTE_API}/api/quote?t=${encodeURIComponent(t)}`);
    const j = await r.json();
    if (!r.ok || j.error) throw new Error(j.error || ("조회 실패 " + r.status));
    return j;
  }

  async function fetchRegime(kr) {
    try {
      if (kr) {
        const j = await (await fetch(KR_DATA + "?t=" + Date.now())).json();
        const zone = (j.zone || "").toUpperCase();
        const gate = (j.macro_gate || "").toLowerCase();
        const off = zone.includes("ELEVATED") || zone.includes("PANIC") ||
                    gate === "caution" || gate === "crisis";
        return { off, label: `KR ${j.zone || "?"}`, detail: `VKOSPI proxy ${j.vkospi_proxy ?? "?"} · macro ${j.macro_gate || "?"}` };
      } else {
        const j = await (await fetch(`${QUOTE_API}/api/last`)).json();
        const m = (j.signal && j.signal.market) || {};
        const vix = m.vix, belowEma = m.spy_ab50 === false, dd = m.spy_dd_60;
        const off = (vix != null && vix > 25) || belowEma || (dd != null && dd < -10);
        return { off, label: `US ${off ? "RISK-OFF" : "RISK-ON"}`,
                 detail: `VIX ${vix != null ? vix.toFixed(1) : "?"} · SPY ${belowEma ? "<EMA50" : ">EMA50"}` };
      }
    } catch (e) {
      return { off: false, label: "국면 불명(강세 가정)", detail: e.message };
    }
  }

  function setStatus(msg, err) {
    const el = $("pl-status"); el.textContent = msg || ""; el.className = "pl-status" + (err ? " err" : "");
  }

  async function run() {
    const t = ($("pl-ticker").value || "").trim().toUpperCase();
    if (!t) { setStatus("티커를 입력해줘", true); return; }
    const kr = isKR(t);
    const capital = parseFloat($("pl-capital").value);
    if (!capital || capital <= 0) { setStatus("자본을 입력해줘", true); return; }
    const tolVal = parseFloat($("pl-tol").value);
    const tolMode = $("pl-tol-mode").value;
    if (!(tolVal > 0)) { setStatus("견딜 손실을 입력해줘", true); return; }

    $("pl-fetch").disabled = true;
    setStatus("조회 중…");
    try {
      const [q, reg] = await Promise.all([fetchQuote(t), fetchRegime(kr)]);
      const atrPct = q.atr_pct;
      if (!(atrPct > 0)) throw new Error("ATR 계산 불가");
      const entry = parseFloat($("pl-entry").value) || q.price;

      const tolCash = tolMode === "pct" ? capital * tolVal / 100 : tolVal;
      const typicalMAE = MAE_MULT * atrPct;                 // fraction
      const posPain = tolCash / typicalMAE;                 // size so typical underwater = tolerance
      const shares = Math.max(0, Math.floor(posPain / entry));
      const actualPos = shares * entry;
      const underwaterCash = typicalMAE * actualPos;

      const stopMult = reg.off ? 1.5 : 3.0;
      const stopPrice = entry * (1 - stopMult * atrPct);
      const stopRisk = (entry - stopPrice) * shares;
      const atrUnits3pct = 0.03 / atrPct;                   // -3% expressed in ATRs

      // alt: size by stop distance
      const posStop = tolCash / (stopMult * atrPct);
      const sharesStop = Math.max(0, Math.floor(posStop / entry));

      // ---- render
      $("pl-result").style.display = "block";
      const rb = $("pl-regime-badge");
      rb.textContent = reg.label; rb.className = "pl-badge " + (reg.off ? "off" : "on");
      $("pl-quote-badge").textContent = `${q.ticker} ${fmtPx(q.price, kr)} · ATR ${(atrPct * 100).toFixed(1)}%` + (q.from_cache ? " (캐시)" : "");

      $("pl-shares").textContent = shares.toLocaleString() + "주";
      $("pl-position").textContent = `${fmtCur(actualPos, kr)} (자본의 ${(actualPos / capital * 100).toFixed(0)}%)`;
      $("pl-stop").textContent = fmtPx(stopPrice, kr);
      $("pl-stop-tag").textContent = reg.off ? `약세 −1.5×ATR (트레일)` : `강세 −3×ATR`;
      $("pl-stop-tag").style.color = reg.off ? "#fca5a5" : "#6ee7b7";
      $("pl-stop-risk").textContent = `손절 시 −${fmtCur(stopRisk, kr)} (${(stopMult * atrPct * 100).toFixed(1)}%)`;

      $("pl-pain").innerHTML = `이 종목은 보통 진입 후 <strong>−${(typicalMAE * 100).toFixed(1)}%</strong> 까지 물려 (2.7×ATR). ` +
        `위 사이즈면 그게 약 <strong>${fmtCur(underwaterCash, kr)}</strong> — 네가 정한 "견딜 손실"과 같아. ` +
        `견딜 만하면 진입, 아니면 견딜 손실을 줄여(사이즈↓).`;

      $("pl-3pct-note").innerHTML = `참고: <strong>−3% 고정 손절은 이 종목엔 ${atrUnits3pct.toFixed(2)}×ATR</strong> ` +
        `— 하루 변동(1 ATR=${(atrPct * 100).toFixed(1)}%)의 ${(0.03 / atrPct * 100).toFixed(0)}% 수준이라 노이즈에 털리고 다시 오름(whipsaw). 그래서 고정% 대신 ATR 기반.`;

      $("pl-detail").innerHTML = `
        <tr><td>현재가 / 진입가</td><td>${fmtPx(q.price, kr)} / ${fmtPx(entry, kr)}</td></tr>
        <tr><td>ATR(14)</td><td>${fmtPx(q.atr14, kr)} (${(atrPct * 100).toFixed(2)}%)</td></tr>
        <tr><td>견딜 손실</td><td>${fmtCur(tolCash, kr)}${tolMode === "pct" ? ` (자본 ${tolVal}%)` : ""}</td></tr>
        <tr><td>추천 사이즈 (물림 기준)</td><td>${shares.toLocaleString()}주 · ${fmtCur(actualPos, kr)}</td></tr>
        <tr><td>대안 사이즈 (손절 기준)</td><td>${sharesStop.toLocaleString()}주 · ${fmtCur(sharesStop * entry, kr)}</td></tr>
        <tr><td>52주 범위</td><td>${fmtPx(q.low_52w, kr)} ~ ${fmtPx(q.high_52w, kr)}</td></tr>
        <tr><td>시장 국면</td><td>${reg.label} · ${reg.detail}</td></tr>`;
      setStatus(`완료 · ${q.date} 기준`);
    } catch (e) {
      setStatus("실패: " + e.message, true);
    } finally {
      $("pl-fetch").disabled = false;
    }
  }

  function init() {
    $("pl-fetch").addEventListener("click", run);
    $("pl-ticker").addEventListener("keypress", (e) => { if (e.key === "Enter") run(); });
    $("pl-entry").addEventListener("keypress", (e) => { if (e.key === "Enter") run(); });
    // sensible default capital per first use
    if (!$("pl-capital").value) $("pl-capital").value = 10000000;
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
