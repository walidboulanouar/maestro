# Maestro — multi-stage build → small runtime image
FROM node:22-slim AS build
WORKDIR /app
COPY package.json package-lock.json* .npmrc ./
RUN npm install --no-audit --no-fund
COPY tsconfig.json ./
COPY src ./src
RUN npm run build

FROM node:22-slim AS runtime
WORKDIR /app
ENV NODE_ENV=production
COPY package.json package-lock.json* .npmrc ./
RUN npm install --omit=dev --no-audit --no-fund
COPY --from=build /app/dist ./dist
EXPOSE 8080
# Runs on the built-in mock provider unless a provider key is supplied at runtime.
ENTRYPOINT ["node", "dist/cli.js"]
CMD ["serve"]
