import os
import json
from pymongo import MongoClient
import pandas as pd
from copy import copy
import numpy as np

client = MongoClient()
total_runs = 0
ascension_runs = 0
with open('cards.json','r') as g:
    cards = json.load(g)
with open('relics.json','r') as g:
    relics = json.load(g)
characters_dict = {
    'IRONCLAD':1,
    'THE_SILENT':2,
    'DEFECT':3,
    'WATCHER':4
}
# cards_columns = [str(card['Name']).replace(' ','') for card in cards]
# relics_columns = [str(relic['Name']).replace(' ','') for relic in relics]
cards_columns = list(cards.keys())
relics_columns = list(relics.keys()) + ['Wing Boots']

for card_name in copy(cards_columns):
    cards_columns.append(card_name+'+1')
columns = [
    'gold_spent',
    'floor_reached',
    'items_purged_num',
    'campfires_num',
    'total_cards',
    'total_relics',
    'potions_used_num',
    'total_damage_taken',
    'character_chosen',
    'items_purchased_num',
    'campfire_rested_num',
    'campfire_upgraded_num',
    'victory'] + cards_columns + relics_columns
columns_dict = { i:columns[i] for i in range(len(columns)) }

db = client['sts']
collection = db['runs']

total_runs = collection.find().count
df_item = []

for run in collection.find():
    try:
        row = { column:0 for column in columns }
        skip = False
        for card in run['master_deck']:
            if card not in cards_columns:
                skip = True
                break
        for relic in run['relics']:
            if relic not in relics_columns:
                skip = True
                break
        if skip:
            continue
        # Deprecated Content
        if set(['Dodecahedron','Toxic Egg 2', 'Frozen Egg 2','WingedGreaves','GremlinMask','ReplayTheSpireMod:DimensionalGlitch','conspire:Flyswatter','aspiration:Nostalgia','Seething Fire','HollowMod:PaleOreRelic','BranchBurnRelic','King of Hearts','hubris:Stopwatch','Replay:WaspNest']).intersection(run['relics']) or set(['Underhanded Strike','Underhanded Strike+1','Crippling Poison','Crippling Poison+1','Venomology','Night Terror','Night Terror+1','Ghostly','Ghostly+1','Replay:Poison Smokescreen+1','hubris:Disease','BasicCrystalCard','BasicCrystalCard+1','Fluid Movement','Fluid Movement+1','ReplayTheSpireMod:Splinters','ReplayTheSpireMod:Splinters+1','ReplayTheSpireMod:Calculation Training','ReplayTheSpireMod:Calculation Training+1','ReplayTheSpireMod:LeadingStrike','ReplayTheSpireMod:LeadingStrike+1','ReplayTheSpireMod:Muscle Training','ReplayTheSpireMod:Muscle Training+1','ReplayTheSpireMod:Strike From Hell','strawberrySpire:CrashLanding','strawberrySpire:Spite','conspire:GhostlyStrike','Hallucinations','Strike_Witch','Pot Of Greed','Hemogenesis','Defend_R+2','Sneak Up']).intersection(run['master_deck']):
            continue
        # Malformed Run
        if 'is_ascension_mode' not in run.keys() or 'campfire_choices' not in run.keys():
            continue
        if run['is_ascension_mode']:
            ascension_runs += 1
            gold_spent = 0
            prev_gold = run['gold_per_floor'][0] # initial gold
            for gold in run['gold_per_floor']:
                if prev_gold - gold > 0:
                    gold_spent += prev_gold - gold
                    prev_gold = gold
            row['gold_spent'] = gold_spent
            row['floor_reached'] = run['floor_reached']
            row['items_purged_num'] = len(run['items_purged'])
            row['campfires_num'] = len(run['campfire_choices'])
            row['total_cards'] = len(run['master_deck'])
            for card in run['master_deck']:
                try:
                    if 'Searing Blow' in card and card not in row.keys():
                        row[card] = 1
                    else:
                        row[card] += 1
                except KeyError as e: # ignore unrecognized cards (deprecated or wrong name)
                    e.with_traceback()
            total_relics = len(run['relics'])
            for relic in run['relics']:
                try:
                    row[relic] += 1
                except KeyError as e: # ignore unrecognized relics (deprecated or wrong name)
                    e.with_traceback()
            # potions_obtained_num = len(run['potions_obtained'])
            row['potions_used_num'] = len(run['potions_floor_usage'])
            total_damage_taken = 0
            for damage in run['damage_taken']:
                total_damage_taken += damage['damage']
            row['total_damage_taken'] = total_damage_taken
            row['character_chosen'] = characters_dict.get(run['character_chosen'])
            row['items_purchased_num'] = len(run['items_purchased'])
            row['campfire_rested_num'] = run['campfire_rested']
            row['campfire_upgraded_num'] = run['campfire_upgraded']
            row['victory'] = int(run['victory'])
            df_item.append(row)
    except KeyError as k:
        print(k)
        exit()        
    except UnicodeDecodeError:
        print('error')
df = pd.DataFrame(df_item)
print(df.head())
wins = df.loc[df['victory'] == 1]
some_loss_df = df.loc[df['victory'] == 0]
np.random.seed(0)
indxes = np.random.randint(0,len(some_loss_df),13821)
losses = df.iloc[indxes]
print(losses.head())
df = wins.append(losses)
df.to_csv('sts_data.csv',index=False)
print('ascension runs:', ascension_runs)