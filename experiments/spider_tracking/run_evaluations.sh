
END=95
for ((i=5;i<=END;i+=10)); do
	python ./experiments/spider_tracking/helpers/lstm_quantitative.py ./experiments/spider_tracking/lstm_config_file.json 199 ./data/spider/Test/${i}_percent/ ${i}
	python ./experiments/spider_tracking/helpers/dnbp_quantitative.py ./experiments/spider_tracking/dnbp_config_file.json 74 ./data/spider/Test/${i}_percent/ ${i}
done


python ./experiments/spider_tracking/helpers/plot_quantitative.py
python ./experiments/spider_tracking/helpers/plot_dnbp_histograms.py
python ./experiments/spider_tracking/helpers/plot_dataset_histograms.py
