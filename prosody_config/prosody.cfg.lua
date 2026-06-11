-- Prosody XMPP configuration for Viralytics multi-agent system (development)
-- TLS is provided via a self-signed cert generated at image build time.
-- SPADE agents connect with verify_security=False so the cert is accepted.

daemonize = false
pidfile = "/var/run/prosody/prosody.pid"

interfaces = { "*" }

-- TLS cert is at /etc/prosody/certs/localhost.{crt,key} (generated in Dockerfile).
-- Prosody picks them up automatically for the VirtualHost below.
-- We don't REQUIRE encryption so the service still starts even if cert loading
-- fails in a misconfigured environment, but STARTTLS will be offered when available.
c2s_require_encryption = false

-- Allow agents to self-register their JIDs on first run (auto_register=True in SPADE)
allow_registration = true

authentication = "internal_plain"

log = {
    info = "*console";
}

modules_enabled = {
    "roster";
    "saslauth";
    "tls";           -- handles STARTTLS negotiation
    "dialback";
    "disco";
    "private";
    "version";
    "uptime";
    "time";
    "ping";
    "register";
    "posix";
}

-- The virtual host domain must match the @domain in agent JIDs:
--   orchestrator@localhost, body@localhost, etc.
VirtualHost "localhost"
    ssl = {
        key         = "/etc/prosody/certs/localhost.key";
        certificate = "/etc/prosody/certs/localhost.crt";
    }
