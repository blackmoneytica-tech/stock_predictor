// KR V25-full Dashboard fetcher
const SECTOR_NAMES = {
  Semi: '반도체', Tech: 'Tech', Game: '게임', Auto: '자동차',
  Battery: '2차전지', Chem: '화학', Oil: '정유', DefShip: '방산·조선',
  Finance: '금융', Bio: '바이오', Consumer: '소비재', Util: '인프라',
  Logistics: '물류', Construct: '건설', Leisure: '레저', Other: '기타'
};

const ZONE_CLASS = {
  'CALM (CASH)': 'calm',
  'NORMAL': 'normal',
  'ELEVATED': 'elevated',
  'PANIC (MAX BUY)': 'panic',
};

const ACTION_CLASS = {
  cash: 'cash', normal: 'normal', elevated: 'elevated', panic: 'panic',
};

async function loadDaily() {
  try {
    const cacheBust = '?t=' + Date.now();
    const resp = await fetch('data/daily.json' + cacheBust);
    if (!resp.ok) throw new Error('Daily JSON fetch failed: ' + resp.status);
    const data = await resp.json();
    renderDaily(data);
    return data;
  } catch (e) {
    document.getElementById('zone-value').textContent = 'ERROR';
    console.error(e);
    return null;
  }
}

async function loadMonthly() {
  try {
    const cacheBust = '?t=' + Date.now();
    const resp = await fetch('data/monthly.json' + cacheBust);
    if (!resp.ok) throw new Error('Monthly fetch failed');
    const data = await resp.json();
    renderMonthly(data);
    return data;
  } catch (e) {
    document.getElementById('picks-body').innerHTML =
      '<tr><td colspan="5" class="loading">Picks data 없음 — 다음 rebal 대기</td></tr>';
    console.warn(e);
    return null;
  }
}

async function loadHistory() {
  try {
    const cacheBust = '?t=' + Date.now();
    const resp = await fetch('data/history.json' + cacheBust);
    if (!resp.ok) throw new Error('History fetch failed');
    const data = await resp.json();
    renderHistory(data);
  } catch (e) {
    document.getElementById('zone-history').innerHTML =
      '<div class="loading">History 데이터 없음</div>';
    console.warn(e);
  }
}

function fmtPct(v, digits = 1) {
  if (v === null || v === undefined) return '-';
  const sign = v >= 0 ? '+' : '';
  return sign + (v * 100).toFixed(digits) + '%';
}

function fmtNum(v, digits = 2) {
  if (v === null || v === undefined) return '-';
  return Number(v).toFixed(digits);
}

function renderDaily(d) {
  document.getElementById('last-update').textContent =
    `🕐 ${d.timestamp} (KST) · 데이터 기준: ${d.market_date}`;

  document.getElementById('ks200-close').textContent =
    d.ks200_close ? Number(d.ks200_close).toLocaleString('ko-KR', {maximumFractionDigits: 2}) : '-';
  document.getElementById('vkospi-proxy').textContent = fmtNum(d.vkospi_proxy);
  document.getElementById('zone-value').textContent = d.zone || '-';
  document.getElementById('final-lev').textContent = d.final_lev !== null ? d.final_lev + 'x' : '-';
  document.getElementById('macro-gate').textContent = (d.macro_gate || 'normal').toUpperCase();
  document.getElementById('usdkrw').textContent = d.usdkrw ? Math.round(d.usdkrw) : '-';
  document.getElementById('vix').textContent = fmtNum(d.us_vix);
  // Strict-Panic: lagged DD가 action 기준 (전일 종가까지의 60d high)
  const ddLagged = d.ks200_dd_60d_lagged !== undefined ? d.ks200_dd_60d_lagged : d.ks200_dd_60d;
  document.getElementById('ks200-dd60').textContent = ddLagged !== null ? fmtPct(ddLagged) + ' (lagged)' : '-';

  const zoneBanner = document.getElementById('zone-banner');
  zoneBanner.className = 'zone-banner ' + (ZONE_CLASS[d.zone] || '');

  const actionEl = document.getElementById('action-msg');
  let actionText = '';
  let actionClass = '';
  if (d.final_lev === 0) {
    actionText = '💤 CASH (KODEX MMF / 예금)';
    actionClass = 'cash';
  } else if (d.final_lev === 1.0) {
    actionText = '✅ 정상 운영 (lev 1x, 종목 7개 균등)';
    actionClass = 'normal';
  } else if (d.final_lev === 1.5) {
    actionText = '⚡ Elevated (lev 1.5x, 신용 50%)';
    actionClass = 'elevated';
  } else if (d.final_lev >= 2.0) {
    actionText = '🔥 PANIC BUY (lev 2x, max 신용)';
    actionClass = 'panic';
  }

  // H-B Peak Exit 상태 추가
  let hbText = '';
  if (d.hb_triggered) {
    const stocks = (d.hb_triggered_stocks || []).join(', ');
    hbText = `\n🚨 H-B PEAK EXIT 발동! 다음 거래일 portfolio 1/3 매도\n트리거: ${stocks}`;
  } else if (d.hb_cooldown_active) {
    hbText = `\n⏸ H-B 쿨다운 (${d.hb_cooldown_days_left}일 남음)`;
  }

  // Deployed_pct 표시 (H-B 1/3 매도 누적 시)
  const dep = d.deployed_pct !== undefined ? d.deployed_pct : 1.0;
  if (dep < 1.0) {
    hbText += `\n💼 Deployed: ${(dep*100).toFixed(0)}% / Effective lev: ${d.effective_lev ? d.effective_lev.toFixed(2) : '?'}x`;
  }

  actionEl.textContent = actionText + hbText;
  actionEl.className = 'action ' + actionClass;
}

function renderMonthly(d) {
  document.getElementById('last-rebal').textContent = d.last_rebal || '-';
  document.getElementById('next-rebal').textContent = d.next_rebal || '21일 후';
  document.getElementById('rebal-zone').textContent =
    `${d.zone || '-'} (w=${d.squeeze_weight || '-'})`;

  const tbody = document.getElementById('picks-body');
  if (!d.picks || d.picks.length === 0) {
    tbody.innerHTML = '<tr><td colspan="5" class="loading">Picks 없음</td></tr>';
    return;
  }
  tbody.innerHTML = d.picks.map((p, i) => {
    const momCls = p.mom120 >= 0 ? 'mom-positive' : 'mom-negative';
    const statCls = p.status === 'BUY' ? 'status-buy' :
                    p.status === 'HOLD' ? 'status-hold' : 'status-sell';
    const statEmoji = p.status === 'BUY' ? '🟢' :
                      p.status === 'HOLD' ? '🔄' : '🔴';
    return `<tr>
      <td>${i+1}</td>
      <td><span class="ticker">${p.code}</span><span class="name">${p.name}</span></td>
      <td>${SECTOR_NAMES[p.sector] || p.sector}</td>
      <td class="${momCls}">${fmtPct(p.mom120, 1)}</td>
      <td>${renderFlowCell(p)}</td>
      <td class="${statCls}">${statEmoji} ${p.status}</td>
    </tr>`;
  }).join('');

  // Flow legend / 업데이트 시각
  const legend = document.getElementById('flow-legend');
  if (legend) {
    const anyBottom = d.picks.some(p => p.flow_bottom);
    let txt = '수급 = 외국인/기관 5일 누적 순매수 (거래대금 대비 %). 외=외국인, 기=기관.';
    if (anyBottom) {
      txt += ' ⚠️ = universe Bottom 20% (강한 매도 압력 — 단기 약세 주의).';
    }
    if (d.flow_updated) txt += ` · 수급 갱신: ${d.flow_updated}`;
    legend.textContent = txt;
  }
}

function renderFlowCell(p) {
  const combo = p.combo_5d_pct;
  if (combo === null || combo === undefined) return '<span class="flow-na">-</span>';
  const frgn = p.frgn_5d_pct;
  const inst = p.inst_5d_pct;
  const comboCls = combo >= 0 ? 'flow-pos' : 'flow-neg';
  const bottomMark = p.flow_bottom ? ' <span class="flow-warn">⚠️</span>' : '';
  const fStr = frgn !== null && frgn !== undefined ?
    `<span class="${frgn >= 0 ? 'flow-pos' : 'flow-neg'}">외${fmtPct(frgn, 0)}</span>` : '';
  const iStr = inst !== null && inst !== undefined ?
    `<span class="${inst >= 0 ? 'flow-pos' : 'flow-neg'}">기${fmtPct(inst, 0)}</span>` : '';
  return `<span class="flow-combo ${comboCls}">${fmtPct(combo, 0)}</span>${bottomMark}
    <span class="flow-detail">${fStr} ${iStr}</span>`;
}

function renderHistory(d) {
  if (!d.history || d.history.length === 0) {
    document.getElementById('zone-history').innerHTML =
      '<div class="loading">History 데이터 없음</div>';
    return;
  }
  const last30 = d.history.slice(-30);
  const html = last30.map(h => {
    const cls = (h.zone_label || 'normal').toLowerCase();
    return `<div class="bar ${cls}" data-tooltip="${h.date} · ${h.zone} · lev ${h.lev}x"></div>`;
  }).join('');
  document.getElementById('zone-history').innerHTML = html;
}

// ============================================================
// ⚡ Action Plan — 내일 정확한 액션 + 종목별 주 수 계산
// ============================================================
const CAPITAL_STORAGE_KEY = 'kr_v25_capital_krw';

function getCapital() {
  const v = parseFloat(localStorage.getItem(CAPITAL_STORAGE_KEY) || '10000000');
  return isNaN(v) ? 10000000 : v;
}

function saveCapital(v) {
  localStorage.setItem(CAPITAL_STORAGE_KEY, String(v));
}

function fmtKRW(v) {
  if (v === null || v === undefined || isNaN(v)) return '-';
  return Math.round(v).toLocaleString('ko-KR');
}

function determineAction(daily, monthly) {
  // 우선순위: H-B 발동 > Cash > Rebal day (매도/매수) > 보유 유지
  if (daily?.hb_triggered) {
    return {
      emoji: '🚨',
      text: 'H-B 익절 발동 — Portfolio 1/3 매도',
      sub: `트리거: ${(daily.hb_triggered_stocks || []).join(', ')} · 5일 쿨다운 시작`,
      class: 'hb',
    };
  }
  if (daily?.final_lev === 0) {
    return {
      emoji: '💤',
      text: 'CASH (전량 매도, MMF/예금 대기)',
      sub: 'Macro Crisis gate 발동 — 시장 회복까지 보유 X',
      class: 'cash',
    };
  }
  // Rebal day 체크 (next_rebal_date가 내일이거나 오늘이면 rebal 임박)
  const sells = monthly?.sells || [];
  const buys = (monthly?.picks || []).filter(p => p.status === 'BUY');
  // 최근 rebal이 오늘이면 = 방금 rebal됨
  const today = daily?.market_date;
  const lastRebal = monthly?.last_rebal;
  const isRebalDay = today === lastRebal && (sells.length > 0 || buys.length > 0);

  if (isRebalDay) {
    return {
      emoji: '🔄',
      text: `월간 Rebal — 매도 ${sells.length}종, 매수 ${buys.length}종`,
      sub: `Top-7 갱신. 매도 먼저 → 매수 진행 (현금 회수 후 진입)`,
      class: 'rebal',
    };
  }
  // 보유 유지
  const lev = daily?.final_lev || 1.0;
  const levText = lev === 1.5 ? '신용 50%' : lev === 2.0 ? '신용 100% (PANIC)' : '현금';
  return {
    emoji: '✅',
    text: `보유 유지 (Lev ${lev}x, ${levText})`,
    sub: monthly?.next_rebal_date ? `다음 rebal: ${monthly.next_rebal_date}` : '추가 액션 없음',
    class: 'hold',
  };
}

function renderActionPlan(daily, monthly) {
  if (!daily || !monthly) return;

  // Date
  const today = new Date(daily.market_date);
  const tomorrow = new Date(today.getTime() + 24*60*60*1000);
  const dStr = `${tomorrow.getFullYear()}-${String(tomorrow.getMonth()+1).padStart(2,'0')}-${String(tomorrow.getDate()).padStart(2,'0')}`;
  document.getElementById('action-date').textContent = `(${dStr})`;

  // Headline
  const action = determineAction(daily, monthly);
  document.getElementById('headline-emoji').textContent = action.emoji;
  document.getElementById('headline-text').textContent = action.text;
  document.getElementById('headline-sub').textContent = action.sub;
  const headline = document.getElementById('action-headline');
  headline.className = 'action-headline ' + action.class;

  // Capital input
  const capInput = document.getElementById('capital');
  const cap = getCapital();
  capInput.value = cap;

  // Compute order table
  renderOrderTable(daily, monthly, cap);

  // Sell list (rebal day)
  const sells = monthly.sells || [];
  const sellList = document.getElementById('sell-list');
  const sellBody = document.getElementById('sell-body');
  if (sells.length > 0) {
    sellList.style.display = 'block';
    sellBody.innerHTML = sells.map(s =>
      `<li><strong>${s.code}</strong> ${s.name} <small>(${s.sector})</small> — 전량 매도</li>`
    ).join('');
  } else {
    sellList.style.display = 'none';
  }
}

function renderOrderTable(daily, monthly, capital) {
  const picks = monthly.picks || [];
  if (picks.length === 0) {
    document.getElementById('order-body').innerHTML =
      '<tr><td colspan="6" class="loading">Picks 없음</td></tr>';
    return;
  }

  const lev = daily.final_lev || 1.0;
  const deployed = daily.deployed_pct !== undefined ? daily.deployed_pct : 1.0;
  const effLev = lev * deployed;
  const totalDeploy = capital * effLev;
  const perStock = totalDeploy / picks.length;

  // Capital summary
  const summary = document.getElementById('capital-summary');
  summary.innerHTML = `
    <div>레버리지: <strong>${lev}x</strong>${deployed < 1.0 ? ` × deployed <strong>${(deployed*100).toFixed(0)}%</strong> = 실효 <strong>${effLev.toFixed(2)}x</strong>` : ''}</div>
    <div>총 운용금: <strong>${fmtKRW(totalDeploy)}원</strong> (자본 ${fmtKRW(capital)} × ${effLev.toFixed(2)})</div>
    <div>종목당 배분: <strong>${fmtKRW(perStock)}원</strong> (1/${picks.length})</div>
  `;

  // Order table rows
  const rows = picks.map((p, i) => {
    const close = p.close;
    const shares = close && close > 0 ? Math.floor(perStock / close) : 0;
    const actualKRW = shares * (close || 0);
    const closeStr = close ? close.toLocaleString('ko-KR') : '-';
    const statCls = p.status === 'BUY' ? 'status-buy' :
                    p.status === 'HOLD' ? 'status-hold' : 'status-sell';
    const statEmoji = p.status === 'BUY' ? '🟢 매수' :
                      p.status === 'HOLD' ? '🔄 보유' : '🔴 매도';
    const flowWarn = p.flow_bottom ? ' <span class="flow-warn" title="외국인+기관 강한 매도 — 단기 약세 주의">⚠️</span>' : '';
    return `<tr>
      <td>${i+1}</td>
      <td><span class="ticker">${p.code}</span><span class="name">${p.name}</span>${flowWarn}</td>
      <td class="num">${closeStr}</td>
      <td class="num">${fmtKRW(actualKRW)}</td>
      <td class="num shares">${shares}주</td>
      <td class="${statCls}">${statEmoji}</td>
    </tr>`;
  }).join('');
  document.getElementById('order-body').innerHTML = rows;
}

function bindCapitalInput() {
  const input = document.getElementById('capital');
  const btn = document.getElementById('capital-save');
  const save = async () => {
    const v = parseFloat(input.value);
    if (!isNaN(v) && v > 0) {
      saveCapital(v);
      // Re-render with new capital
      const [daily, monthly] = await Promise.all([
        fetch('data/daily.json?t=' + Date.now()).then(r => r.json()),
        fetch('data/monthly.json?t=' + Date.now()).then(r => r.json()),
      ]);
      renderActionPlan(daily, monthly);
    }
  };
  btn.addEventListener('click', save);
  input.addEventListener('keypress', e => { if (e.key === 'Enter') save(); });
}

// Init
(async () => {
  bindCapitalInput();
  const [daily, monthly] = await Promise.all([loadDaily(), loadMonthly(), loadHistory()]);
  if (daily && monthly) renderActionPlan(daily, monthly);
})();

// Auto-refresh every 5 minutes
setInterval(async () => {
  const [daily, monthly] = await Promise.all([loadDaily(), loadMonthly(), loadHistory()]);
  if (daily && monthly) renderActionPlan(daily, monthly);
}, 5 * 60 * 1000);
