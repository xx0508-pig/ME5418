# seed43/44 顺序训练队列

实际项目仍位于原 Documents/Codex 目录；本聊天指定的 /home/saga/Projects/ME5418_RL_group 在启动时不存在。

脚本：scripts/train_seeds_43_44.py。先 seed43 后 seed44，各按 11维k1、11维k4、9维k1、9维k4、6维k1 顺序训练。每组500000步、fixed_v1归一化、摩擦0.3–0.9、ent_coef=0.01、每50000步保存检查点。使用新版默认CPU策略训练和EGL渲染。

后台服务名称：me5418-seeds43-44。关闭终端不影响运行，但不要关机、休眠或退出桌面登录。训练期间不要更新/修改代码。

查看服务及队列进度：
```bash
systemctl --user status me5418-seeds43-44 --no-pager
journalctl --user -u me5418-seeds43-44 -f
```

每组详细训练日志、status.json 和源码快照在 outputs/training_suites/seeds43_44_queue/。模型在 outputs/runs/formal_norm_*_seed43/44/。

停止整个队列（包括当前训练）：
```bash
systemctl --user stop me5418-seeds43-44
```
停止服务后 status.json 可能仍显示 running，应以 systemctl 状态为准。

重新启动前检查中断模型。脚本自动跳过配置匹配的已完成模型，但遇到未完成目录会停止保护，不会自动从检查点恢复，也不会删除重训。需要先确定如何恢复那一组。没有未完成目录时可在项目根目录前台运行：
```bash
bash scripts/local.sh -m scripts.train_seeds_43_44
```

本地尚有未提交的部署改动，未配置 Git 提交身份。本队列不伪造干净提交，而在开始前保存源码归档、每文件SHA256、Git提交号及dirty状态。后续正式评估前仍应提交和统一代码版本。未向 GitHub 推送。
