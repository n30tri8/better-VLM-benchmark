@echo off

python few_shot_template_generator.py --type "color" --input_file "../VL-commonsense/mine-data/db/color/single/train.jsonl" --target_distribution "black,blue,brown,gray,green,orange,pink,purple,red,silver,white,yellow"
python few_shot_template_generator.py --type "color" --input_file "../VL-commonsense/mine-data/db/wiki-color/single/train.jsonl" --target_distribution "black,blue,brown,gray,green,orange,pink,purple,red,silver,white,yellow" --wiki
python few_shot_template_generator.py --type "material" --input_file "../VL-commonsense/mine-data/db/material/single/train.jsonl" --target_distribution "bronze,ceramic,cloth,concrete,cotton,denim,glass,gold,iron,jade,leather,metal,paper,plastic,rubber,stone,tin,wood"
python few_shot_template_generator.py --type "material" --input_file "../VL-commonsense/mine-data/db/wiki-material/single/train.jsonl" --target_distribution "bronze,ceramic,cloth,concrete,cotton,denim,glass,gold,iron,jade,leather,metal,paper,plastic,rubber,stone,tin,wood" --wiki
python few_shot_template_generator.py --type "shape" --input_file "../VL-commonsense/mine-data/db/shape/single/train.jsonl" --target_distribution "cross,heart,octagon,oval,polygon,rectangle,rhombus,round,semicircle,square,star,triangle"
python few_shot_template_generator.py --type "shape" --input_file "../VL-commonsense/mine-data/db/wiki-shape/single/train.jsonl" --target_distribution "cross,heart,octagon,oval,polygon,rectangle,rhombus,round,semicircle,square,star,triangle" --wiki
