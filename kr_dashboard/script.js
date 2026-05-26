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
    if (d.strict_panic_fallback) {
      actionText = '⚡ Strong-Bull Elevated (lev 1.5x)\n신고가+변동성, 진짜 panic 아님 (Strict-Panic fallback)';
      actionClass = 'elevated';
    } else {
      actionText = '⚡ Elevated (lev 1.5x, 신용 50%)';
      actionClass = 'elevated';
    }
  } else if (d.final_lev >= 2.0) {
    if (d.strict_panic_real) {
      actionText = '🔥 REAL PANIC BUY (lev 2x)\nproxy≥30 AND lagged DD≤-10%';
    } else {
      actionText = '🔥 PANIC BUY (lev 2x, max 신용)';
    }
    actionClass = 'panic';
  }
  actionEl.textContent = actionText;
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
      <td class="${statCls}">${statEmoji} ${p.status}</td>
    </tr>`;
  }).join('');
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

// Init
(async () => {
  await Promise.all([loadDaily(), loadMonthly(), loadHistory()]);
})();

// Auto-refresh every 5 minutes
setInterval(() => {
  loadDaily();
  loadMonthly();
  loadHistory();
}, 5 * 60 * 1000);
