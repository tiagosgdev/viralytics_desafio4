FROM prosody/prosody:latest

USER root

# Generate a self-signed TLS certificate for the "localhost" virtual host.
# Prosody looks for /etc/prosody/certs/<vhost>.crt + .key by convention.
# SPADE connects with verify_security=False so the self-signed cert is accepted.
RUN mkdir -p /etc/prosody/certs /var/run/prosody && \
    openssl req -x509 -newkey rsa:2048 \
        -keyout /etc/prosody/certs/localhost.key \
        -out  /etc/prosody/certs/localhost.crt \
        -days 3650 -nodes \
        -subj "/CN=localhost" 2>/dev/null && \
    chown -R prosody:prosody /etc/prosody/certs /var/run/prosody && \
    chmod 600 /etc/prosody/certs/localhost.key && \
    # Resolve the prosody binary path at build time (avoids PATH issues at runtime).
    # Bypass the default entrypoint's setpriv call: setpriv/initgroups fails on
    # Docker Desktop for Windows (WSL2 kernel restriction). Already running as
    # prosody user, so privilege drop is unnecessary.
    PROSODY_BIN=$(which prosody) && \
    printf '#!/bin/sh\nexec %s\n' "$PROSODY_BIN" > /start.sh && \
    chmod +x /start.sh && \
    chown prosody:prosody /start.sh

USER prosody

ENTRYPOINT ["/start.sh"]
