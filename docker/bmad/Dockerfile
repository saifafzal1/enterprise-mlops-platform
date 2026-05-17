FROM node:20-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip curl libnss3 libatk1.0-0 libatk-bridge2.0-0 \
    libcups2 libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 \
    libxrandr2 libgbm1 libasound2 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY package.json ./
RUN npm install

RUN npx playwright install chromium --with-deps

COPY validator.js ./

EXPOSE 8001
HEALTHCHECK --interval=15s --timeout=5s CMD curl -f http://localhost:8001/health || exit 1
CMD ["node", "validator.js"]
