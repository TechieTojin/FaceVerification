# FaceVerification 3.0 — web UI served by nginx.
# Build from the repository root (needs models/face_landmarker.task: run `python scripts/fetch_models.py` first):
#   docker build -f frontend/Dockerfile -t faceverification-web .
FROM node:22-alpine AS build
WORKDIR /src/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
COPY models/face_landmarker.task /src/models/face_landmarker.task
RUN npm run build

FROM nginx:1.27-alpine
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /src/frontend/dist /usr/share/nginx/html
EXPOSE 8080
