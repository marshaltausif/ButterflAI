#!/bin/bash
set -e
G='\033[0;32m'; Y='\033[0;33m'; C='\033[0;36m'; R='\033[0;31m'; N='\033[0m'
log(){ echo -e "${C}[ButterflAI]${N} $1"; }
ok() { echo -e "${G}[✓]${N} $1"; }
warn(){ echo -e "${Y}[!]${N} $1"; }

echo -e "${Y}\n  🦋  ButterflAI Setup\n  ═══════════════════\n${N}"

node -v >/dev/null 2>&1 || { echo -e "${R}[✗] Node.js 18+ required${N}"; exit 1; }
ok "Node.js $(node -v)"

log "Installing npm packages…"
npm install --silent && ok "Dependencies installed"

if [ ! -f .env.local ]; then
  cp .env.local.example .env.local
  if command -v openssl &>/dev/null; then
    SEC=$(openssl rand -base64 32)
    sed -i.bak "s/your_secret_here/$SEC/" .env.local 2>/dev/null || sed -i '' "s/your_secret_here/$SEC/" .env.local
    ok "NEXTAUTH_SECRET generated"
  fi
  warn ".env.local created — fill in your API keys"
else
  ok ".env.local already exists"
fi

echo ""
echo -e "  ${C}Required keys (.env.local):${N}"
echo "  • GOOGLE_CLIENT_ID / SECRET → console.cloud.google.com (enable Drive API)"
echo "  • GEMINI_API_KEY            → aistudio.google.com (FREE)"
echo "  • GROQ_API_KEY              → console.groq.com (FREE)"
echo "  • NEXT_PUBLIC_SUPABASE_URL  → supabase.com (run supabase_schema.sql first)"
echo "  • MODAL_TOKEN_ID / SECRET   → modal.com (modal token new)"
echo ""
echo -e "  ${Y}Then:${N}"
echo "  modal deploy modal/butterflai_worker.py   # deploy GPU worker once"
echo "  npm run dev                               # start locally"
echo "  npx vercel --prod                         # deploy to Vercel"
echo ""
echo -e "  ${G}🦋 butterflai.app${N}\n"
