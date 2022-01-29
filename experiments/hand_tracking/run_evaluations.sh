
python ./experiments/hand_tracking/helpers/dnbp_quantitative.py ./experiments/hand_tracking/dnbp_config_file.json 16 estimation
python ./experiments/hand_tracking/helpers/dnbp_quantitative.py ./experiments/hand_tracking/dnbp_config_file.json 16 tracking
python ./experiments/hand_tracking/helpers/plot_quantitative.py

python ./experiments/hand_tracking/helpers/dnbp_qualitative.py ./experiments/hand_tracking/dnbp_config_file.json 16