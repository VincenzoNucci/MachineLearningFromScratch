import os
import json
import pandas as pd

total_runs = 0
ascension_runs = 0
characters_dict = {
    'IRONCLAD':1,
    'THE_SILENT':2,
    'DEFECT':3,
    'WATCHER':4
}
columns = ['gold_spent','floor_reached','items_purged_num','campfires_num','total_cards','total_relics','potions_used_num','total_damage_taken','character_chosen','items_purchased_num','campfire_rested_num','campfire_upgraded_num','victory']
columns_dict = { i:columns[i] for i in range(len(columns)) }

for root, directiories, filenames in os.walk(os.path.join(os.getcwd(),'sts_runs','ascension20')):
    total_runs = len(filenames)
    df_item = []
    for filename in filenames:
        try:
            with open(os.path.join(os.getcwd(),'sts_runs','ascension20',filename),'r', encoding='utf-8') as g:
                run = json.load(g)
            if run['is_ascension_mode'] and run['ascension_level'] == 20:
                ascension_runs += 1
                gold_spent = run['gold_per_floor'][0] - run['gold_per_floor'][-1]
                gold_spent = 0
                floor_reached = run['floor_reached']
                items_purged_num = len(run['items_purged'])
                campfires_num = len(run['campfire_choices'])
                total_cards = len(run['master_deck'])
                total_relics = len(run['relics'])
                # potions_obtained_num = len(run['potions_obtained'])
                potions_used_num = len(run['potions_floor_usage'])
                total_damage_taken = 0
                for damage in run['damage_taken']:
                    total_damage_taken += damage['damage']
                total_damage_taken = 0
                character_chosen = characters_dict.get(run['character_chosen'])
                items_purchased_num = len(run['items_purchased'])
                campfire_rested_num = run['campfire_rested']
                campfire_upgraded_num = run['campfire_upgraded']
                victory = int(run['victory'])
                df_item.append([gold_spent, floor_reached, items_purged_num, campfires_num, total_cards, total_relics, potions_used_num, total_damage_taken, character_chosen,items_purchased_num, campfire_rested_num, campfire_upgraded_num, victory])
        except KeyError:
            continue
        except UnicodeDecodeError:
            continue
df = pd.DataFrame(df_item,columns=columns)
print(df.head())
df.to_csv('sts_data.csv')
print('ascension runs:', ascension_runs)