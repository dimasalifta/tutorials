---

# FRP Server (frps) Setup on Ubuntu

Manual installation recommended for production.

---

## 1️⃣ Download Binary

Check system architecture:

```bash
uname -m
```

Common outputs:

* `x86_64` → `amd64`
* `aarch64` → `arm64`

Download the latest release from [FRP GitHub Releases](https://github.com/fatedier/frp/releases).

Example for `amd64`:

```bash
wget https://github.com/fatedier/frp/releases/download/v0.67.0/frp_0.67.0_linux_amd64.tar.gz
```

Extract:

```bash
tar -xzf frp_0.67.0_linux_amd64.tar.gz
cd frp_0.67.0_linux_amd64
```

Copy binaries to system path:

```bash
sudo mv frps /usr/local/bin/
sudo chmod +x /usr/local/bin/frps
```

---

## 2️⃣ Setup FRP Server Configuration

Create config directory:

```bash
sudo mkdir -p /etc/frp
```

Create config file:

```bash
sudo nano /etc/frp/frps.toml
```

Example minimal config:

```toml
bindPort = 7000

auth.method = "token"
auth.token = "ganti_dengan_token_kuat"

webServer.port = 7500
webServer.addr = "0.0.0.0"
webServer.user = "admin"
webServer.password = "admin123"
```

Explanation:

* `7000` → control channel
* `7500` → dashboard monitoring

---

## 3️⃣ Setup FRP Systemd Service

Create service file:

```bash
sudo nano /etc/systemd/system/frps.service
```

Content:

```ini
[Unit]
Description=FRP Server
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/frps -c /etc/frp/frps.toml
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Reload systemd and start service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable frps
sudo systemctl start frps
```

Check status:

```bash
sudo systemctl status frps
```

---
