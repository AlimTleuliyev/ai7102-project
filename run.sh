pip install -r requirements.txt

# English (Dundee) preprocessing
python preprocess/DC/add_annotation.py
python preprocess/DC/filter.py
python preprocess/DC/data_points4modeling.py
bash preprocess/DC/modify_context.sh

# English surprisal calculation - Wiki-LMs
bash experiments/en_surprisal.sh
python experiments/convert_scores.py --dir surprisals/DC --corpus dundee
python experiments/dundee.py surprisals/DC/
python experiments/aggregate.py --dir surprisals/DC --file likelihood.txt > surprisals/DC/aggregated.txt

# Vanilla model (ablation study)
bash experiments/en_surprisal_vanilla.sh
python experiments/convert_scores.py --dir surprisals/DC-vanilla --corpus dundee
python experiments/dundee.py surprisals/DC-vanilla/
python experiments/aggregate.py --dir surprisals/DC-vanilla --file likelihood.txt > surprisals/DC-vanilla/aggregated.txt

# GPT-2 models
bash experiments/en_surprisal_hf.sh
python experiments/convert_scores.py --dir surprisals/DC-hf --corpus dundee
python experiments/dundee.py surprisals/DC-hf/
python experiments/aggregate.py --dir surprisals/DC-hf --file likelihood.txt > surprisals/DC-hf/aggregated.txt
