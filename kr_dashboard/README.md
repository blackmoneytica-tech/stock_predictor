# 🇰🇷 KR V25-full Champion Dashboard

한국 KOSPI200 V25-full 전략 운영 대시보드 + Telegram 알림.

## 백테스트 검증

- 기간: 2014-03-04 ~ 2026-05-26 (12.2년)
- Total: +19,796% / CAGR 54%
- Sharpe: 1.38
- Max DD: -41%
- WF 6/6 alpha 양수 (mean +232.4pp)
- IS Sh 1.33 → OOS Sh 1.74 (Δ +0.41)

## Strategy 요약

```
score = mom120 × 100
      + ((1 - squeeze_ratio) × 100 + 30_if_breakout) × w  [if sq ≤ 0.8]
        where w = 0.7/0.5/0.3 by zone (normal/elevated/panic)
      + (dist_from_52w_low × 100 × 0.06 + ret_5d × 50 × 0.2)  [if dist > 0.30]

leverage:
  zone_lev = 0/1/1.5/2x by VKOSPI proxy (15/22.5/30 thresholds)
  if macro_gate == 'crisis': lev = 0
  elif macro_gate == 'caution': lev = min(lev, 1.0)
  final = dd_multistage(lev, capital_dd)   # -30%/-45%/-55%/-65% throttle

universe: PIT KOSPI200 시총 top 50 (분기별 갱신)
rebal: monthly (21d), top-7 균등, sector cap 3
```

## 운영 자동화

### GitHub Actions
- **`.github/workflows/kr_daily.yml`** — 매일 KST 16:00 (월-금): zone + lev 계산 + Telegram
- **`.github/workflows/kr_monthly_rebal.yml`** — 매월 마지막 금요일: top-7 picking + Telegram

### JSON Output
GitHub Actions가 `kr_dashboard/data/` 폴더에 갱신:
- `daily.json` — 오늘 zone/lev/macro
- `monthly.json` — 마지막 rebal picks
- `history.json` — 최근 1년 zone 이력

### Cloudflare Pages 배포
1. Cloudflare Dashboard → Pages → "Create a project"
2. Connect to Git (이 repo 선택)
3. Build settings:
   - Production branch: `main` (또는 default branch)
   - Build command: 비움 (정적 사이트)
   - Build output: `kr_dashboard`
4. Deploy → 도메인 자동 발급 (예: `kr-v25.pages.dev`)
5. (선택) Custom domain 연결

배포 후 사이트는 `kr_dashboard/data/*.json`을 fetch해서 표시.
GitHub Actions가 매일 JSON 갱신 → Cloudflare Pages 자동 재빌드 → 사이트 업데이트.

## Telegram 새 봇 생성 가이드

### 1. BotFather로 봇 만들기
1. Telegram에서 `@BotFather` 검색
2. `/newbot` 입력
3. 봇 이름 (display name): 예) `KR V25 Champion`
4. 봇 username: 예) `kr_v25_champion_bot` (반드시 `_bot` 끝)
5. **token 받기** (예: `1234567890:ABCdef...`)

### 2. Chat ID 알아내기
1. 방금 만든 봇 username 검색해서 대화 시작
2. 아무 메시지 (`/start` 또는 "hi") 보내기
3. 브라우저에서 열기: `https://api.telegram.org/bot<TOKEN>/getUpdates`
4. JSON 응답에서 `chat.id` 찾기 (예: `123456789`)

### 3. GitHub Secrets 등록
```
Repository → Settings → Secrets and variables → Actions → New repository secret
```
- `KR_TG_BOT_TOKEN` = 봇 token (위 1번)
- `KR_TG_CHAT_ID` = chat id (위 2번)

### 4. 테스트
```
Actions 탭 → KR V25-full Daily → Run workflow
```
정상이면 Telegram에 메시지 도착.

## 로컬 테스트

```bash
# Daily 테스트 (Telegram 토큰 환경변수 설정 시 알림 전송)
python scripts/kr_publish_json.py --mode daily

# 월간 picking 테스트
python scripts/kr_publish_json.py --mode rebal

# 둘 다
python scripts/kr_publish_json.py --mode both

# Telegram 환경변수 (선택)
export TG_BOT_TOKEN='...'
export TG_CHAT_ID='...'
```

## 데이터 흐름

```
GitHub Actions cron (KST 16:00)
  ↓ scripts/kr_publish_json.py 실행
  ↓ FDR/yfinance에서 KS200, USDKRW, ^VIX, 50종목 fetch
  ↓ V25-full picking + zone + macro gate 계산
  ↓ kr_dashboard/data/*.json 갱신
  ↓ git commit & push
  ↓
Cloudflare Pages (auto rebuild)
  ↓ 사이트 업데이트
  ↓ 브라우저 fetch (5분마다 auto refresh)
  ↓
Telegram (병행)
  ↓ KR V25-full Daily 메시지
  ↓ Zone + Lev + Action + 권고
```

## 한계 + 주의사항

- KRX 인증 미승인이라 VKOSPI는 KS200 realized vol × 1.25 proxy 사용
- proxy 노이즈 ~30-50% (zone 임계값에서 분류 오류 가능)
- 백테스트 12.2년 (미국 18y의 67%)
- 강세장 win6 (2024-2026) 의존성 일부 존재
- look-ahead bias 모두 제거 (모든 market signal shift(1))
- V27 KS200 60d DD throttle은 retract됨 (정보 표시만)
- H-B Peak Exit (V30)는 검증 완료지만 미통합 (boundary: 강세장 alpha 깎임 risk)

## 다음 단계 (옵션)
- [ ] 1년 forward test (실 운영 vs backtest 차이 측정)
- [ ] KRX OpenAPI 활용 신청 승인 시 VKOSPI 실데이터로 zone 재교정
- [ ] H-B 통합 (DD 큰 사건 발생 후 재검토)
