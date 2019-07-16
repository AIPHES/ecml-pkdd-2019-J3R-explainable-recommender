import argparse
import os, sys
from subprocess import check_output

def mkdirp(path):
    """Checks if a path exists otherwise creates it
    Each line in the filename should contain a list of URLs separated by comma.
    Args:
        path: The path to check or create
    """
    if path == '':
        return
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

#./bin/librec rec -exec -conf test_conf/PMF.conf -D dfs.data.dir=./data -D data.input.path=$domain/${domain}_train.csv -D dfs.result.dir=./result/valid -D data.testset.path=$domain/test/${domain}_valid.csv
#./bin/librec rec -exec -conf test_conf/PMF.conf -D dfs.data.dir=./data -D data.input.path=$domain/${domain}_train.csv -D dfs.result.dir=./result/test -D data.testset.path=$domain/test/${domain}_test.csv

def run(args):
    models = ['NMF','SVD++',"PMF"]
    factors = ['10', '20', '30', '40', '50', '60','70', '80', '90', '100']
    #domains= ["music", 'toys', 'electronics', 'kindle', 'tv']
    domains= ["yelp"]

    for factor in factors:
        for domain in domains:
            for data_type in ["valid", "test"]:
                if not os.path.isdir(args.logs_path):
                    mkdirp(args.logs_path)
                if not os.path.isdir(args.results_path):
                    mkdirp(args.results_path)

                for model in models:
                    train_arg = "-D data.input.path=%s/%s_train.csv" % (domain, domain)
                    test_arg = "-D data.testset.path=%s/test/%s_%s.csv" % (domain, domain, data_type)
                    factor_arg = "-D rec.factor.number=%s" % (factor)
                    log_arg = "2> %s/%s_%s_%s_%s.log" % (args.logs_path, domain, model, data_type, factor)
                    results_arg= "-D dfs.result.dir=%s/%s_%s" % (args.results_path, data_type, factor)
                    command = "./bin/librec rec -exec -conf test_conf/%s.conf -D dfs.data.dir=%s %s %s %s %s %s" % (model, args.data_dir, results_arg, train_arg, test_arg, factor_arg, log_arg)
                    print(command)
                    try:
                        check_output(command, shell=True)
                    except:
                        continue


def get_args():
    ''' This function parses and return arguments passed in'''

    parser = argparse.ArgumentParser(description='Run librec baselines')

    parser.add_argument('--data_dir', type=str, help='Data Directory', required=False,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))

    parser.add_argument('--results_path', type=str, help='Data Directory', required=False,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))

    parser.add_argument('--logs_path', type=str, help='Data Directory', required=False,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"))


    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = get_args()
    run(args)

