# Reaching Eris from your phone (Samsung S24 Ultra) — WiFi and 5G

Eris's server already binds to `0.0.0.0:8001`, so it's reachable from other
devices — you just need a network path to your Alienware and (off-LAN) a tunnel.

## On the same WiFi (works today, no setup)
1. On the Alienware, find its LAN IP: `ipconfig` → the IPv4 like `192.168.1.42`.
2. On the phone (same WiFi): open `http://192.168.1.42:8001/`.
That's it on your home network. (If it doesn't load, allow `python.exe` through
Windows Firewall for Private networks.)

## From anywhere — 5G / Verizon (the professional way: Tailscale)
Tailscale is a free mesh VPN. Your phone and PC join a private network and talk
by a stable private IP from anywhere — encrypted, no port-forwarding, nothing
exposed to the public internet.

1. **PC:** install Tailscale (https://tailscale.com/download/windows), sign in.
2. **Phone:** install the Tailscale app from the Play Store, sign in with the
   **same account**.
3. On the PC, note its Tailscale IP (`100.x.y.z`) from the Tailscale tray menu.
4. On the phone (on 5G, WiFi off to prove it): open `http://100.x.y.z:8001/`.

Eris now works from your phone on Verizon 5G, your couch, or a coffee shop — same
URL everywhere. To make it a tap-to-open "app", open it in Chrome on the phone →
menu → **Add to Home screen** (installs it as a PWA-style icon).

### Optional: a "MagicDNS" name instead of the number
In the Tailscale admin console enable MagicDNS; then the PC is reachable as
`http://alienware:8001/` (or whatever you name it) instead of the `100.x` IP.

## Lock it down with a token (recommended once it's reachable off-LAN)
By default there's no password — fine on your own tailnet (only your devices),
but add a token so only you can use her even if the port is ever reachable:

1. In `start_eris.bat` (or the environment), set a secret before launch:
   ```bat
   set ERIS_AUTH_TOKEN=pick-a-long-random-string
   ```
2. Restart Eris. Now every request needs that token.
3. On the phone, visit once with the token in the URL:
   `http://100.x.y.z:8001/?token=pick-a-long-random-string`
   It sets a cookie, so you stay logged in afterward; bookmark/Add-to-Home-screen
   that URL. (With no `ERIS_AUTH_TOKEN` set, nothing changes — open access on LAN.)

## Alternatives
- **Cloudflare Tunnel** (`cloudflared`): gives a public HTTPS URL; pair it with
  `ERIS_AUTH_TOKEN` since it's internet-facing.
- **ngrok**: quickest temporary public URL for testing; also use the token.
- **Port forwarding** on your router: works but exposes the port to the whole
  internet — only with `ERIS_AUTH_TOKEN` set, and Tailscale is strictly better.

Recommendation: **Tailscale + `ERIS_AUTH_TOKEN`.** Private, encrypted, free, and
it just works from 5G.
