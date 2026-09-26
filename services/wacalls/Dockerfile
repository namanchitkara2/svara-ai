# WaCalls sidecar: pinned WaCallsNative + wake-agent patches.
FROM node:22-alpine AS web
RUN apk add --no-cache git
ARG PIN=026b889bef9e649600ced1992fb2228fe7954b77
RUN git clone https://github.com/jobasfernandes/WaCallsNative /src && cd /src && git checkout $PIN
COPY patches /patches
RUN cd /src && for p in /patches/*.patch; do git apply "$p"; done
RUN cd /src/client && npm ci && npm run build

FROM golang:1.26-alpine AS build
COPY --from=web /src /src
WORKDIR /src
RUN rm -rf internal/app/webui/dist && cp -r client/dist internal/app/webui/dist \
 && CGO_ENABLED=0 go build -trimpath -ldflags "-s -w" -o /out/wacalls ./cmd/server

FROM alpine:3.21
RUN apk add --no-cache ca-certificates && adduser -D -u 10001 wacalls
COPY --from=build /out/wacalls /usr/local/bin/wacalls
USER wacalls
VOLUME /data
# Loopback only: the wake-agent container shares this network namespace, nothing is published.
ENTRYPOINT ["wacalls", "-addr", "127.0.0.1:8787", "-db", "/data/wacalls.db", "-static", ""]
