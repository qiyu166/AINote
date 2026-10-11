# 推送到 GitHub（qiyu166/AINote）

仓库已在本地建好：两个提交（上游基线 + 你的二创），分支 `main`，**没有配 remote**。

---

## 第 1 步：在 GitHub 上创建空仓库

浏览器打开 https://github.com/new

| 字段 | 填什么 |
|---|---|
| Repository name | `AINote` |
| Description | `本地优先的视频笔记工具：发芽/拷问/自测三种思考模式 + Obsidian 双链集成（基于 video-to-note 二创）` |
| Public / Private | 按需 |
| **Initialize this repository with** | ⚠️ **全部不要勾**（不要 README、不要 .gitignore、不要 LICENSE） |

> 勾了会产生一个和本地冲突的提交，push 会被拒。

---

## 第 2 步：关联并推送

```bash
cd <本仓库所在目录>

git remote add origin https://github.com/qiyu166/AINote.git
git push -u origin main
```

推送时会要求认证：

- **推荐**：用 Personal Access Token 当密码（GitHub 已不支持账号密码）
  生成地址：https://github.com/settings/tokens → Fine-grained token → 勾 `Contents: Read and write`
- 也可以用 GitHub Desktop / 已配好的凭证助手

---

## 第 3 步：验证

推送后打开 https://github.com/qiyu166/AINote ，应看到：

- 93 个文件
- 两个提交（`chore: 导入上游…` + `feat: AINote 二创…`）
- README 自动渲染出 AINote 的介绍与功能对比表
- LICENSE 里两条版权声明（like-attract + qiyu166）

---

## ⚠️ 登录/推不上去时：hosts 问题

本机 `/etc/hosts` 里有 **Steam++（Watt Toolkit）** 写入的一批劫持条目，把 `github.com`、
`api.github.com`、`raw.githubusercontent.com` 等 20+ 域名指向了 `127.0.0.1`：

```
# Steam++ Start
127.0.0.1 github.com
127.0.0.1 api.github.com
...
# Steam++ End
```

**如果 git push 报连接失败**，二选一：

**方案 A（推荐）**：启动 Steam++，让它接管 github 加速（这是它原本的用途）

**方案 B**：把 github 相关行注释掉，恢复直连

```bash
sudo cp /etc/hosts /etc/hosts.bak-$(date +%Y%m%d)
sudo sed -i '' 's|^127\.0\.0\.1 \(.*github.*\)$|#127.0.0.1 \1|' /etc/hosts
# 恢复：sudo cp /etc/hosts.bak-YYYYMMDD /etc/hosts
```

> 注意：`huggingface.co` 也被同样方式劫持（第 28 行）。若之后模型下载失败，
> 用同样的思路处理——先启动 Steam++，或注释掉那一行。

---

## 推完之后的建议

1. **加 Topics**（仓库页右上 ⚙️）：`obsidian` `video-to-note` `bilibili` `whisper` `llm` `fastapi`
2. **Release**：需要 Windows 便携版的话，上游的 `.github/workflows/release.yml` 还在，打 tag 即可触发
3. **致谢置顶**：README 底部已写明基于 like-attract/video-to-note 二创，MIT 要求保留——已经做到了

---

## 关于隐私（已核对）

本地仓库 93 个文件，**不含**：

- `.env`（你本机配置，含 Obsidian 路径）
- `workspace/`（106MB 任务数据）
- `llm_keys.json` / `bili_credentials.json`（你的 API Key 与 B 站凭据）
- `.venv` / `.venv-dy`（444MB）
- `.backup-before-review-fixes/`、`:memory:.ses`

`.gitignore` 已覆盖以上全部。若之后新增文件，推之前跑一次：

```bash
git status --short          # 确认没有意外文件
git ls-files | grep -Ei "\.env$|keys|credential|workspace/"
```
