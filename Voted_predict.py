import pandas as pd
import numpy as np
import re
import sys

# 解决 Windows 终端打印编码问题
if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def predict_votes_from_your_model(data_path, beta_path='Posterior_Beta_Occupation.csv'):
    # 1. 加载数据
    df = pd.read_csv(data_path)
    try:
        # 读取 Beyasian.py 生成的后验系数
        beta_df = pd.read_csv(beta_path)
        beta_lookup = dict(zip(beta_df['Feature'], beta_df['Posterior_Beta']))
        cert_lookup = dict(zip(beta_df['Feature'], beta_df['Certainty']))
    except FileNotFoundError:
        print(f"Error: Cannot find {beta_path}. Please run Beyasian.py first.")
        return

    # 2. 预处理淘汰信息
    def get_elim_week(res_str):
        if pd.isna(res_str): return 99
        res_str = str(res_str).lower()
        if "winner" in res_str or "runner-up" in res_str or "third" in res_str:
            return 99
        match = re.search(r'eliminated week (\d+)', res_str)
        return int(match.group(1)) if match else 99
    
    df['elim_week'] = df['results'].apply(get_elim_week)
    judge_cols = [c for c in df.columns if 'judge' in c]
    
    final_output_list = []
    TOTAL_VOTES = 10000000 

    # 3. 按赛季循环
    for season_val, s_group in df.groupby('season'):
        # 判定规则 (1,2,28+ 为 Rank 制，3-27 为 Percentage 制)
        is_rank_rule = True if (season_val <= 2 or season_val >= 28) else False
        
        for w in range(1, 14): # 考虑到部分赛季可能有13周
            w_cols = [c for c in judge_cols if f'week{w}_' in c]
            if not w_cols: continue
            
            # 计算当周评委得分
            # 注意：这里需要处理选手可能没有分（NaN）的情况，用0填充
            s_group[f'w{w}_score'] = s_group[w_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1).fillna(0)
            
            # 筛选当周在场的选手
            active = s_group[s_group[f'w{w}_score'] > 0].copy()
            if len(active) < 2: continue

            # --- 映射核心逻辑：基础权重来自你的模型 Beta ---
            active['model_weight'] = active['celebrity_industry'].map(
                lambda x: np.exp(beta_lookup.get(x, 0))
            )
            
            # 先给一个基于模型的初始预测值，防止 KeyError
            active['final_votes'] = (active['model_weight'] / active['model_weight'].sum()) * TOTAL_VOTES

            # 4. 淘汰约束修正
            eliminated_this_week = active[active['elim_week'] == w]
            if not eliminated_this_week.empty:
                actual_loser_idx = eliminated_this_week.index[0]
                
                # 迭代微调：确保计算出的淘汰者符合历史
                for _ in range(20):
                    active['final_votes'] = (active['model_weight'] / active['model_weight'].sum()) * TOTAL_VOTES
                    
                    if is_rank_rule:
                        j_rank = active[f'w{w}_score'].rank(ascending=False)
                        v_rank = active['final_votes'].rank(ascending=False)
                        combined_rank = j_rank + v_rank
                        pred_loser_idx = combined_rank.idxmax()
                    else:
                        j_pct = active[f'w{w}_score'] / active[f'w{w}_score'].sum()
                        v_pct = active['final_votes'] / TOTAL_VOTES
                        combined_pct = j_pct + v_pct
                        pred_loser_idx = combined_pct.idxmin()

                    if pred_loser_idx == actual_loser_idx:
                        break
                    
                    # 调整：降低实际淘汰者的粉丝吸引力权重
                    active.loc[actual_loser_idx, 'model_weight'] *= 0.92
                    active.loc[pred_loser_idx, 'model_weight'] *= 1.08

            # 5. 存储结果
            for _, row in active.iterrows():
                final_output_list.append({
                    'Season': int(season_val),
                    'Week': w,
                    'Celebrity': row['celebrity_name'],
                    'Industry': row['celebrity_industry'],
                    'Judge_Total': row[f'w{w}_score'],
                    'Estimated_Votes': int(row['final_votes']),
                    'Fan_Percent': f"{(row['final_votes']/TOTAL_VOTES)*100:.2f}%",
                    'Certainty_From_Model': cert_lookup.get(row['celebrity_industry'], 0.5)
                })

    # 6. 保存数据 (使用 utf-8-sig 确保 Excel 打开不乱码)
    output_df = pd.DataFrame(final_output_list)
    output_df.to_csv('Weekly_Votes_Based_On_Your_Model.csv', index=False, encoding='utf-8-sig')
    print("Success: Generated Weekly_Votes_Based_On_Your_Model.csv based on your Bayesian model.")

if __name__ == "__main__":
    predict_votes_from_your_model('2026_MCM_Problem_C_Data.csv')