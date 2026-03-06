---

# FRP Client (frpc) Setup on Ubuntu

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
wget https://github.com/fatedier/frp/releases/download/v0.56.0/frp_0.56.0_linux_amd64.tar.gz
```

Extract:

```bash
tar -xzf frp_0.56.0_linux_amd64.tar.gz
cd frp_0.56.0_linux_amd64
```

Copy binaries to system path:

```bash
sudo mv frpc /usr/local/bin/
sudo chmod +x /usr/local/bin/frpc
```

---

## 2️⃣ Setup FRP Client Configuration

Create config directory:

```bash
sudo mkdir -p /etc/frp
```

Create config file:

```bash
sudo nano /etc/frp/frpc.toml
```

Example config to expose SSH:

```toml
serverAddr = "IP_PUBLIC_SERVER"
serverPort = 7000

auth.method = "token"
auth.token = "ganti_dengan_token_kuat"

[[proxies]]
name = "ssh"
type = "tcp"
localIP = "127.0.0.1"
localPort = 22
remotePort = 6000
```

Now you can access SSH via:

```bash
ssh user@SERVER_IP -p 6000
```

---

## 2️⃣ Setup FRP Client Systemd Service

Create service file:

```bash
sudo nano /etc/systemd/system/frpc.service
```

Content:

```ini
[Unit]
Description=FRP Client
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/frpc -c /etc/frp/frpc.toml
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable frpc
sudo systemctl start frpc
```
