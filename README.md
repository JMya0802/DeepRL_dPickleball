# AIT306_FinalProject

我的思路: 
1. 改变Reward
   - Win point
   - Lose point
   - Hit the ball
   - Return the ball

2. 创新结合:Curriculum Learning + ELO + Self-Play
   - 给Agent一个由简单到难的课程
   - 一开始Agent要学会打球，中期训练为了避免训练不稳定引入ELO算法:
     - 当有一边的Agent开始越来越弱的时候 ELO的分数会越来越低 这时候把分数低的对手换成历史的Agent进行对打 之后ELO分数变高的时候才继续对打
