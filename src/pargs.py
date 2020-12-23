import argparse
import torch

def pargs():
	parser = argparse.ArgumentParser(description='K2NL')

	# mode
	parser.add_argument("-selector", action="store_true")
	parser.add_argument("-generator", action="store_true")
	parser.add_argument("-josa", action="store_true")

	# content selector
	parser.add_argument("-content_mode", type=str)
	parser.add_argument("-content_epoch", default=100, type=int)
	parser.add_argument("-content_bsz", default=256, type=int)
	parser.add_argument("-content_hsz", default=300, type=int)
	parser.add_argument("-content_eval_interval", default=5, type=int)
	parser.add_argument("-content_model_save", default="../saved/temp.pt")
	parser.add_argument("-content_output_save", default="../output/content_output.tsv")

	# josa
	parser.add_argument("-josa_model", default="gru")
	parser.add_argument("-josa_mode", type=str)
	parser.add_argument("-josa_esz", default=300, type=int)
	parser.add_argument("-josa_bsz", default=10, type=int)
	parser.add_argument("-josa_hsz", default=300, type=int)
	parser.add_argument("-josa_lsz", default=1, type=int)
	parser.add_argument("-josa_epoch", default=100, type=int)
	parser.add_argument("-josa_eval_interval", default=5, type=int)
	parser.add_argument("-josa_save", default="../saved/temp.pt")
	parser.add_argument("-josa_predict_save", default="../output/josa_output.tsv")

	# data
	parser.add_argument("-datadir", default="../data/")
	parser.add_argument("-josa_train_path", default="josa/preprocessed/train.josa.tsv")
	parser.add_argument("-josa_dev_path", default="josa/preprocessed/dev.josa.tsv")
	parser.add_argument("-josa_test_path", default="josa/preprocessed/test.josa.tsv")
	parser.add_argument("-nlg_train_path", default="preprocess/summary_base/summary_base.train.tsv")
	parser.add_argument("-nlg_dev_path", default="preprocess/summary_base/summary_base.val.tsv")
	parser.add_argument("-nlg_test_path", default="preprocess/summary_base/summary_base.test.tsv")

	# device
	parser.add_argument("-gpu", default=0, type=int)

	args = parser.parse_args()
	if args.gpu == -1:
		args.device = torch.device("cpu")
	else:
		args.device = torch.device(f"cuda:{args.gpu}")

	return args
