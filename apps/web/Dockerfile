FROM node:24.12.0-alpine AS build

WORKDIR /app
RUN npm install --global npm@11.10.1
COPY package.json package-lock.json ./
RUN npm ci --ignore-scripts
COPY . .
RUN npm run build

FROM nginx:1.27-alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist/web/browser /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=10s --timeout=3s --retries=5 \
  CMD wget -q -O - http://127.0.0.1/healthz >/dev/null || exit 1
