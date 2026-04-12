# Tiancheng's Blog

基于 [Hugo](https://gohugo.io/) + [Stack](https://github.com/CaiJimmy/hugo-theme-stack) 主题的个人博客。

🌐 **在线预览**: [https://n571e.github.io](https://n571e.github.io)

## 快速开始

### 环境要求

- [Hugo Extended](https://gohugo.io/installation/) (v0.120+)
- [Go](https://go.dev/) (v1.20+)

### 本地开发

```bash
# 启动本地预览服务器
hugo server

# 构建生产版本
hugo --minify
```

### 写一篇新文章

```bash
hugo new content post/文章名/index.md
```

然后编辑生成的 Markdown 文件，添加内容即可。

## 部署

推送到 `main` 分支后，GitHub Actions 会自动构建并部署到 GitHub Pages。

## 目录结构

```
├── config/_default/   # 配置文件
├── content/
│   ├── post/          # 博客文章
│   └── page/          # 独立页面 (关于/归档/搜索)
├── static/img/        # 静态资源
└── .github/workflows/ # CI/CD 配置
```

## License

MIT
