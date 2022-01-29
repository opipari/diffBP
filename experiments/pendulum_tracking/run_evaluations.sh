
END=95
for ((i=5;i<=END;i+=10)); do
	python ./experiments/pendulum_tracking/helpers/lstm_quantitative.py ./experiments/pendulum_tracking/lstm_config_file.json 74 ./data/pendulum/Test/${i}_percent/ ${i}
	python ./experiments/pendulum_tracking/helpers/dnbp_quantitative.py ./experiments/pendulum_tracking/dnbp_config_file.json 74 ./data/pendulum/Test/${i}_percent/ ${i}
done


python ./experiments/pendulum_tracking/helpers/plot_quantitative.py
python ./experiments/pendulum_tracking/helpers/plot_dnbp_histograms.py
python ./experiments/pendulum_tracking/helpers/plot_dataset_histograms.py
