---
layout: single
title: "ssh to dev-container"
date: 2026-06-10
categories: [ssh, dev-container]
tags: [ssh]
---

1. **Check SSH Service Status** [in dev-container]
   - Verified if SSH service was already running in the container
   - Found no SSH daemon running initially

2. **Install OpenSSH Server** [in dev-container]
   - Used `sudo apt install -y openssh-server` to install SSH server package
   - System automatically generated SSH host keys during installation

3. **Start SSH Service** [in dev-container]
   - Executed `sudo service ssh start` to launch SSH daemon
   - Confirmed SSH was listening on port 22 using `netstat -tlnp`

4. **Configure SSH Security Settings** [in dev-container]
   - Modified `/etc/ssh/sshd_config` to enable root login: `PermitRootLogin yes`
   - Enabled password authentication: `PasswordAuthentication yes`

5. **Set Root Password** [in dev-container]
   - Set root user password to "123456" using `echo "root:123456" | sudo chpasswd`
   - Cleared and reset password when authentication issues occurred

6. **Restart SSH Service** [in dev-container]
   - Applied configuration changes with `sudo service ssh restart`
   - Ensured all settings took effect

7. **Regenerate Host Keys** [in dev-container]
   - Removed old host keys with `sudo rm -f /etc/ssh/ssh_host_*`
   - Generated new keys using `sudo ssh-keygen -A`
   - Resolved host key verification conflicts

8. **Test SSH Connectivity** [in dev-container]
   - Tested internal SSH connection using `sshpass -p "123456" ssh -o StrictHostKeyChecking=no root@localhost`
   - Verified successful authentication and command execution
   - Confirmed external SSH connection works through port forwarding

9. **Setup Port Forwarding and Provide Connection Instructions** [in dev-container + on local machine]
   - Map the current dev container's port 22 to the host's port 65345(or other), enabling external applications to connect to the SSH service inside the container
   - Configured port forwarding: `localhost:65345(or other)` → `container:22` in VS Code/Cursor
   - Created `~/.ssh/config` file on the host machine and added connection settings
   - Guided user through VS Code Remote SSH extension setup or terminal connection

**SSH Configuration** (`~/.ssh/config`):
After completing the above process, you need to manually add this content to your SSH config file, then connect through kirocode

```bash
Host dev-container
  HostName localhost
  User root
  Port 65345
  PasswordAuthentication yes
  StrictHostKeyChecking no
```
