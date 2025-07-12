
###############################################
# Étape 1 : Build front (Node + pnpm)
###############################################
FROM node:24.4-slim AS builder

RUN corepack enable && corepack prepare pnpm@9 --activate

WORKDIR /app

COPY package.json pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile

COPY . .
RUN pnpm run build

### Étape 2 : Serve statique via nginx
FROM nginx:alpine AS runner

COPY --from=builder /app/build /usr/share/nginx/html


CMD ["nginx", "-g", "daemon off;"]
