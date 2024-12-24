@echo off

python few_shot_template_generator.py --output_file color-t.txt --train_dataset color.txt --target_distribution "black,blue,brown,gray,green,orange,pink,purple,red,silver,white,yellow"
python few_shot_template_generator.py --output_file wiki-color-t.txt --train_dataset wiki-color.txt --target_distribution "black,blue,brown,gray,green,orange,pink,purple,red,silver,white,yellow"
python few_shot_template_generator.py --output_file material-t.txt --train_dataset material.txt --target_distribution "bronze,ceramic,cloth,concrete,cotton,denim,glass,gold,iron,jade,leather,metal,paper,plastic,rubber,stone,tin,wood"
python few_shot_template_generator.py --output_file wiki-material-t.txt --train_dataset wiki-material.txt --target_distribution "bronze,ceramic,cloth,concrete,cotton,denim,glass,gold,iron,jade,leather,metal,paper,plastic,rubber,stone,tin,wood"
python few_shot_template_generator.py --output_file shape-t.txt --train_dataset shape.txt --target_distribution "cross,heart,octagon,oval,polygon,rectangle,rhombus,round,semicircle,square,star,triangle"
python few_shot_template_generator.py --output_file wiki-shape-t.txt --train_dataset wiki-shape.txt --target_distribution "cross,heart,octagon,oval,polygon,rectangle,rhombus,round,semicircle,square,star,triangle"





